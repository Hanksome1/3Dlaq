"""
Product Quantization VAE (PQ-VAE) for Latent Action Quantization.

Product Quantization divides the embedding space into M sub-spaces, each with 
its own smaller codebook. This effectively prevents index collapse by:
1. Forcing the model to use all M sub-quantizers
2. Creating a much larger effective codebook (K^M combinations from M×K codes)
3. Using dual-decoding strategy for balanced sub-space utilization

Based on: "PQ-VAE: Learning Efficient Representation with Product Quantized VAE"
https://arxiv.org/abs/2305.14565
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ProductQuantizer(nn.Module):
    """
    Product Quantizer with M sub-quantizers.
    
    The embedding dimension is split into M groups, each quantized independently.
    This gives an effective codebook size of K^M from only M×K codebook entries.
    
    Args:
        num_embeddings: Number of codes per sub-quantizer (K)
        embedding_dim: Total embedding dimension (D), must be divisible by num_groups
        num_groups: Number of sub-quantizers (M)
        commitment_cost: Weight for commitment loss
    """
    
    def __init__(
        self,
        num_embeddings: int = 64,
        embedding_dim: int = 32,
        num_groups: int = 4,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        assert embedding_dim % num_groups == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_groups ({num_groups})"
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.commitment_cost = commitment_cost
        self.group_dim = embedding_dim // num_groups
        
        # M separate codebooks, each of size [K, D/M]
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_embeddings, self.group_dim) * 0.1)
            for _ in range(num_groups)
        ])
        
        # EMA tracking for each sub-codebook
        for i in range(num_groups):
            self.register_buffer(f'ema_cluster_size_{i}', torch.zeros(num_embeddings))
            self.register_buffer(f'ema_sum_{i}', torch.zeros(num_embeddings, self.group_dim))
        
        self.ema_decay = 0.99
        self.eps = 1e-5
    
    def quantize_single_group(
        self, 
        x: torch.Tensor, 
        group_idx: int,
        training: bool = True
    ) -> tuple:
        """
        Quantize a single group using its sub-codebook.
        
        Args:
            x: [B*N, D/M] input for this group
            group_idx: Which sub-codebook to use
            training: Whether to apply EMA updates
            
        Returns:
            quantized: [B*N, D/M] quantized output
            indices: [B*N] codebook indices
            commitment_loss: Scalar
        """
        codebook = self.codebooks[group_idx]  # [K, D/M]
        
        # Compute distances
        distances = (
            torch.sum(x ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(x, codebook.t())
            + torch.sum(codebook ** 2, dim=1)
        )
        
        # Find nearest codes
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, codebook)
        
        # EMA codebook updates during training
        if training:
            encodings = F.one_hot(indices, self.num_embeddings).float()
            cluster_size = encodings.sum(dim=0)
            embedding_sum = torch.matmul(encodings.t(), x)
            
            ema_cluster = getattr(self, f'ema_cluster_size_{group_idx}')
            ema_sum = getattr(self, f'ema_sum_{group_idx}')
            
            ema_cluster.data.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            ema_sum.data.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)
            
            # Update codebook
            n = ema_cluster.sum()
            cluster_size_normalized = (
                (ema_cluster + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            self.codebooks[group_idx].data.copy_(
                ema_sum / cluster_size_normalized.unsqueeze(1)
            )
            
            # Reset unused codes: find codes with very low usage and reinitialize
            usage_threshold = max(1.0, n / (self.num_embeddings * 10))
            unused_mask = ema_cluster < usage_threshold
            num_unused = unused_mask.sum().item()
            if num_unused > 0 and x.shape[0] >= num_unused:
                # Sample random vectors from current batch
                random_indices = torch.randperm(x.shape[0], device=x.device)[:int(num_unused)]
                random_vectors = x[random_indices]
                # Reset unused codes
                self.codebooks[group_idx].data[unused_mask] = random_vectors + 0.01 * torch.randn_like(random_vectors)
                # Reset EMA stats for these codes
                ema_cluster.data[unused_mask] = 1.0
                ema_sum.data[unused_mask] = random_vectors
        
        # Commitment loss (stronger to prevent collapse)
        commitment_loss = F.mse_loss(quantized.detach(), x) * 2.0
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, commitment_loss
    
    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Forward pass with product quantization.
        
        Args:
            inputs: [B, N, D] input tensor
            
        Returns:
            quantized: [B, N, D] quantized output
            commitment_loss: Total commitment loss
            indices: [B, N, M] indices from each sub-quantizer
            perplexity: Average perplexity across sub-quantizers
        """
        input_shape = inputs.shape
        B, N, D = input_shape
        
        flat_input = inputs.reshape(-1, D)  # [B*N, D]
        
        # Split into groups
        groups = flat_input.chunk(self.num_groups, dim=-1)  # M tensors of [B*N, D/M]
        
        quantized_groups = []
        all_indices = []
        total_commitment_loss = 0.0
        total_perplexity = 0.0
        
        for i, group in enumerate(groups):
            q, idx, loss = self.quantize_single_group(
                group, i, training=self.training
            )
            quantized_groups.append(q)
            all_indices.append(idx)
            total_commitment_loss = total_commitment_loss + loss
            
            # Compute perplexity for this group
            encodings = F.one_hot(idx, self.num_embeddings).float()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            total_perplexity = total_perplexity + perplexity
        
        # Concatenate quantized groups
        quantized = torch.cat(quantized_groups, dim=-1)  # [B*N, D]
        quantized = quantized.reshape(input_shape)
        
        # Stack indices: [B*N, M] -> [B, N, M]
        indices = torch.stack(all_indices, dim=-1)
        indices = indices.reshape(B, N, self.num_groups)
        
        # Average losses and perplexity
        commitment_loss = total_commitment_loss / self.num_groups
        perplexity = total_perplexity / self.num_groups
        
        return quantized, commitment_loss, indices, perplexity
    
    @property
    def effective_codebook_size(self) -> int:
        """Effective number of unique codes (K^M)."""
        return self.num_embeddings ** self.num_groups


class LatentPQVAE(nn.Module):
    """
    Latent PQ-VAE for Concerto features with dual-decoding.
    
    Uses Product Quantization to prevent codebook collapse and
    dual-decoding strategy for balanced sub-space utilization.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 64,
        num_embeddings: int = 64,
        num_groups: int = 4,
        code_seq_len: int = 4,
        feature_size: tuple = (14, 14),
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_groups = num_groups
        self.code_seq_len = code_seq_len
        self.feature_size = feature_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Spatial aggregation
        self.aggregation = self._build_aggregation(code_seq_len)
        
        # Product Quantizer
        self.vq = ProductQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_groups=num_groups,
            commitment_cost=commitment_cost,
        )
        
        # Dual decoders for balanced training
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, input_dim),
        )
        
        # Secondary decoder for encoded (pre-quantization) features
        self.encoder_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, input_dim),
        )
    
    def _build_aggregation(self, code_seq_len: int) -> nn.Module:
        if code_seq_len == 1:
            return nn.AdaptiveAvgPool2d((1, 1))
        else:
            grid_size = int(math.sqrt(code_seq_len))
            return nn.AdaptiveAvgPool2d((grid_size, grid_size))
    
    @property
    def codebooks(self):
        """Return flattened codebook for compatibility."""
        # Return first group's codebook for spread loss computation
        return self.vq.codebooks[0]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [B, H, W, D] to [B, code_seq_len, embedding_dim]."""
        B, H, W, D = x.shape
        
        # Add input noise for regularization
        if self.training:
            x = x + 0.01 * torch.randn_like(x)
        
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        x = self.aggregation(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        return x
    
    def decode(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Decode [B, code_seq_len, embedding_dim] to [B, H, W, input_dim]."""
        B = x.shape[0]
        
        x = self.decoder(x)
        
        h = w = int(math.sqrt(self.code_seq_len))
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        
        return x
    
    def forward(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        return_decoded: bool = True,
    ) -> tuple:
        """
        Compute action codes from feature difference with dual decoding.
        
        Returns: (decoded, perplexity, commitment_loss, indices)
        """
        B, H, W, D = features_t0.shape
        
        # Feature difference
        delta = features_t1 - features_t0
        
        # Encode
        encoded = self.encode(delta)
        
        # Quantize
        quantized, commitment_loss, indices, perplexity = self.vq(encoded)
        
        # Dual decoding loss (for training stability)
        if self.training:
            # Decode from quantized
            decoded_q = self.decode(quantized, H, W)
            
            # Decode from encoded (bypassing quantization)
            encoded_flat = encoded.reshape(B, -1, self.embedding_dim)
            decoded_e = self.encoder_decoder(encoded_flat)
            h = w = int(math.sqrt(self.code_seq_len))
            decoded_e = rearrange(decoded_e, 'b (h w) d -> b d h w', h=h, w=w)
            decoded_e = F.interpolate(decoded_e, size=(H, W), mode='bilinear', align_corners=False)
            decoded_e = decoded_e.permute(0, 2, 3, 1)
            
            # Add dual-decoding consistency loss to commitment_loss
            dual_loss = F.mse_loss(decoded_q, decoded_e.detach())
            commitment_loss = commitment_loss + 0.1 * dual_loss
        
        if return_decoded:
            decoded = self.decode(quantized, H, W)
            # Flatten indices for compatibility: [B, N, M] -> [B, N]
            # Use first group's indices (or could combine them)
            flat_indices = indices[..., 0]
            return decoded, perplexity, commitment_loss, flat_indices
        
        flat_indices = indices[..., 0]
        return quantized, perplexity, commitment_loss, flat_indices
    
    def inference(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        user_action_token: torch.Tensor = None,
    ) -> tuple:
        B, H, W, D = features_t0.shape
        
        if user_action_token is not None:
            # Use user-specified action (from first group's codebook)
            quantized = F.embedding(user_action_token, self.vq.codebooks[0])
            # Repeat for all groups (simple approach)
            quantized = quantized.repeat(1, 1, self.num_groups)
            decoded = self.decode(quantized, H, W)
            return decoded, user_action_token
        
        delta = features_t1 - features_t0
        encoded = self.encode(delta)
        quantized, _, indices, _ = self.vq(encoded)
        decoded = self.decode(quantized, H, W)
        
        return decoded, indices[..., 0]
    
    def replace_unused_codebooks(self, num_batches: int):
        """No-op for API compatibility - PQ-VAE uses EMA updates."""
        pass
