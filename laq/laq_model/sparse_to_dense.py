"""
Sparse-to-Dense Projection Module.

This module maps sparse point cloud features [N, D] to a dense 2D feature map [H, W, D]
using learnable spatial queries and cross-attention.

Similar to DETR's object queries, but for spatial feature reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for spatial queries."""
    
    def __init__(self, dim: int, height: int, width: int):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, height * width, dim) * 0.02)
    
    def forward(self) -> torch.Tensor:
        """Returns: [1, H*W, D]"""
        return self.pos_embed


class CrossAttention(nn.Module):
    """Cross-attention between spatial queries and point features."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,  # [B, Q, D] - spatial queries
        keys: torch.Tensor,     # [B, N, D] - point features
        values: torch.Tensor,   # [B, N, D] - point features
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, D] spatial position queries
            keys: [B, N, D] point cloud features
            values: [B, N, D] point cloud features
            
        Returns:
            [B, Q, D] attended features for each query position
        """
        B, Q, D = queries.shape
        _, N, _ = keys.shape
        
        # Project Q, K, V
        q = self.q_proj(queries).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: [B, heads, Q, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output: [B, heads, Q, head_dim] -> [B, Q, D]
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Q, D)
        out = self.out_proj(out)
        
        return out


class SparseToDenseProjection(nn.Module):
    """
    Learnable module to project sparse point cloud features to dense 2D feature map.
    
    Uses learnable spatial queries with cross-attention to aggregate point features.
    
    Architecture:
        1. Learnable 2D position queries [H, W, D]
        2. Cross-attention: queries attend to point cloud features
        3. FFN for feature refinement
        4. Reshape to [H, W, D]
    """
    
    def __init__(
        self,
        point_dim: int = 512,      # Concerto output dimension
        output_dim: int = 512,     # Output feature dimension
        height: int = 14,          # Output spatial height
        width: int = 14,           # Output spatial width
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
    ):
        super().__init__()
        self.point_dim = point_dim
        self.output_dim = output_dim
        self.height = height
        self.width = width
        self.num_queries = height * width
        
        # Project point features to output dim if different
        self.input_proj = nn.Linear(point_dim, output_dim) if point_dim != output_dim else nn.Identity()
        
        # Learnable spatial queries
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_queries, output_dim) * 0.02)
        
        # 2D positional encoding for queries
        self.query_pos = PositionalEncoding2D(output_dim, height, width)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossAttention(output_dim, num_heads, dropout),
                'cross_attn_norm': nn.LayerNorm(output_dim),
                'ffn': nn.Sequential(
                    nn.Linear(output_dim, int(output_dim * ffn_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(output_dim * ffn_ratio), output_dim),
                    nn.Dropout(dropout),
                ),
                'ffn_norm': nn.LayerNorm(output_dim),
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        point_features: torch.Tensor,  # [N, D] single sample or [B, N, D] batch
    ) -> torch.Tensor:
        """
        Project sparse point features to dense 2D feature map.
        
        Args:
            point_features: [N, D] or [B, N, D] point cloud features from Concerto
            
        Returns:
            [H, W, D] or [B, H, W, D] dense feature map
        """
        # Handle single sample
        if point_features.dim() == 2:
            point_features = point_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D_in = point_features.shape
        device = point_features.device
        
        # Project point features
        point_features = self.input_proj(point_features)  # [B, N, D]
        
        # Initialize queries with learnable spatial embeddings + positional encoding
        queries = self.spatial_queries.expand(B, -1, -1) + self.query_pos()  # [B, Q, D]
        queries = queries.to(device)
        
        # Apply cross-attention layers
        for layer in self.layers:
            # Cross-attention: queries attend to point features
            attn_out = layer['cross_attn'](queries, point_features, point_features)
            queries = queries + attn_out
            queries = layer['cross_attn_norm'](queries)
            
            # FFN
            ffn_out = layer['ffn'](queries)
            queries = queries + ffn_out
            queries = layer['ffn_norm'](queries)
        
        # Final normalization
        features = self.output_norm(queries)  # [B, Q, D]
        
        # Reshape to 2D: [B, H*W, D] -> [B, H, W, D]
        features = features.view(B, self.height, self.width, self.output_dim)
        
        if squeeze_output:
            features = features.squeeze(0)  # [H, W, D]
        
        return features


# Test
if __name__ == "__main__":
    print("Testing SparseToDenseProjection...")
    
    # Create module
    module = SparseToDenseProjection(
        point_dim=512,
        output_dim=512,
        height=14,
        width=14,
        num_heads=8,
        num_layers=2,
    )
    
    # Test with single sample
    point_features = torch.randn(8192, 512)  # [N, D]
    output = module(point_features)
    print(f"Input: {point_features.shape} -> Output: {output.shape}")
    assert output.shape == (14, 14, 512), f"Expected (14, 14, 512), got {output.shape}"
    
    # Test with batch
    point_features_batch = torch.randn(2, 8192, 512)  # [B, N, D]
    output_batch = module(point_features_batch)
    print(f"Input: {point_features_batch.shape} -> Output: {output_batch.shape}")
    assert output_batch.shape == (2, 14, 14, 512), f"Expected (2, 14, 14, 512), got {output_batch.shape}"
    
    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    print(f"Parameters: {num_params:,}")
    
    print("✓ All tests passed!")
