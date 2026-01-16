"""
Latent Space NSVQ for Concerto-LAQ.

Modified vector quantization module that operates on Concerto latent features
instead of raw pixel patches. Uses NSVQ (Noise Substitution Vector Quantization)
for gradient-friendly codebook learning.

Based on: https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
from einops import rearrange


class LatentSpaceNSVQ(nn.Module):
    """
    NSVQ adapted for Concerto latent space features.
    
    Unlike the original NSVQ which uses CNN encoders to downsample spatial features,
    this version operates directly on latent space features with attention-based pooling.
    
    Args:
        input_dim: Input feature dimension (from Concerto)
        embedding_dim: Codebook embedding dimension
        num_embeddings: Number of codebook entries (action vocabulary size)
        code_seq_len: Number of action tokens to predict (1, 4, 16, etc.)
        feature_size: Spatial size of input features (H, W)
        discarding_threshold: Threshold for replacing unused codebooks
        initialization: Codebook initialization method ('normal' or 'uniform')
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 32,
        num_embeddings: int = 8,
        code_seq_len: int = 4,
        feature_size: tuple = (8, 8),
        discarding_threshold: float = 0.1,
        initialization: str = 'normal',
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.code_seq_len = code_seq_len
        self.feature_size = feature_size
        self.discarding_threshold = discarding_threshold
        self.device = device
        self.eps = 1e-12
        
        # Initialize codebooks
        if initialization == 'normal':
            codebooks = torch.randn(num_embeddings, embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / num_embeddings, 1 / num_embeddings).sample(
                [num_embeddings, embedding_dim]
            ).to(device)
        else:
            raise ValueError(f"initialization should be 'normal' or 'uniform', got {initialization}")
        
        self.codebooks = nn.Parameter(codebooks, requires_grad=True)
        
        # Usage counter for codebook replacement
        self.register_buffer('codebooks_used', torch.zeros(num_embeddings, dtype=torch.int32, device=device))
        
        # Projection layers
        self.project_in = nn.Linear(input_dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, input_dim)
        
        # Spatial aggregation for producing code_seq_len tokens
        self._setup_spatial_aggregation(code_seq_len, feature_size)
    
    def _setup_spatial_aggregation(self, code_seq_len: int, feature_size: tuple):
        """
        Set up spatial aggregation to reduce features to code_seq_len tokens.
        
        Uses adaptive pooling or attention-based aggregation.
        """
        H, W = feature_size
        total_features = H * W
        
        if code_seq_len == 1:
            # Global average pooling
            self.aggregation = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
            )
            self.aggregation_type = "pool"
        elif code_seq_len <= 4:
            # Small grid pooling
            grid_h = int(math.sqrt(code_seq_len))
            grid_w = code_seq_len // grid_h
            self.aggregation = nn.AdaptiveAvgPool2d((grid_h, grid_w))
            self.aggregation_type = "pool"
        elif code_seq_len <= 16:
            # Medium grid pooling
            grid_size = int(math.sqrt(code_seq_len))
            self.aggregation = nn.AdaptiveAvgPool2d((grid_size, grid_size))
            self.aggregation_type = "pool"
        else:
            # For larger code_seq_len, use learned projection
            self.aggregation = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(total_features * self.embedding_dim, code_seq_len * self.embedding_dim),
            )
            self.aggregation_type = "linear"
        
        # Decoder upsampling (reverse of aggregation)
        if code_seq_len == 1:
            self.upsample = lambda x, h, w: x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        else:
            self.upsample = lambda x, h, w: F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial features to code_seq_len tokens.
        
        Args:
            features: [B, H, W, D] or [B, N, D] latent features
            
        Returns:
            encoded: [B, code_seq_len, embedding_dim]
        """
        B = features.shape[0]
        
        # Handle different input shapes
        if features.ndim == 4:
            # [B, H, W, D] -> [B, D, H, W] for pooling
            features = features.permute(0, 3, 1, 2)
        elif features.ndim == 3:
            # [B, N, D] -> reshape to spatial
            N, D = features.shape[1:]
            H = W = int(math.sqrt(N))
            features = features.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        # Project to embedding dimension
        # [B, D, H, W] -> [B, H, W, D] -> project -> [B, D', H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        features = self.project_in(features)  # [B, H, W, embedding_dim]
        features = features.permute(0, 3, 1, 2)  # [B, embedding_dim, H, W]
        
        # Aggregate spatially
        if self.aggregation_type == "pool":
            encoded = self.aggregation(features)  # [B, embedding_dim, h, w]
            encoded = rearrange(encoded, 'b d h w -> b (h w) d')
        else:
            encoded = self.aggregation(features)  # [B, code_seq_len * embedding_dim]
            encoded = encoded.reshape(B, self.code_seq_len, self.embedding_dim)
        
        return encoded  # [B, code_seq_len, embedding_dim]
    
    def decode(self, quantized: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Decode quantized tokens back to spatial features.
        
        Args:
            quantized: [B, code_seq_len, embedding_dim]
            target_h, target_w: Target spatial dimensions
            
        Returns:
            decoded: [B, target_h, target_w, input_dim]
        """
        B = quantized.shape[0]
        
        # Reshape to spatial for upsampling
        if self.code_seq_len == 1:
            spatial = quantized.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, embedding_dim, 1, 1]
        else:
            h = w = int(math.sqrt(self.code_seq_len))
            spatial = rearrange(quantized, 'b (h w) d -> b d h w', h=h, w=w)
        
        # Upsample to target size
        upsampled = self.upsample(spatial, target_h, target_w)  # [B, embedding_dim, H, W]
        
        # Project back to input dimension
        upsampled = upsampled.permute(0, 2, 3, 1)  # [B, H, W, embedding_dim]
        decoded = self.project_out(upsampled)  # [B, H, W, input_dim]
        
        return decoded
    
    def quantize(self, encoded: torch.Tensor, training: bool = True) -> tuple:
        """
        Quantize encoded features using NSVQ.
        
        Args:
            encoded: [B, code_seq_len, embedding_dim]
            training: Whether in training mode (uses noise substitution)
            
        Returns:
            quantized: [B, code_seq_len, embedding_dim]
            indices: [B, code_seq_len]
            perplexity: Average codebook usage
        """
        B, S, D = encoded.shape
        
        # Flatten for distance computation
        flat_encoded = encoded.reshape(-1, D)  # [B*S, D]
        
        # Compute distances to codebooks
        distances = (
            torch.sum(flat_encoded ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_encoded, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        
        # Find nearest codebook entries
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized = self.codebooks[min_indices]
        
        if training:
            # NSVQ: Use noise substitution for gradient flow
            random_vector = normal_dist.Normal(0, 1).sample(flat_encoded.shape).to(self.device)
            
            norm_quantization_residual = (flat_encoded - hard_quantized).square().sum(dim=1, keepdim=True).sqrt()
            norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
            
            vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector
            quantized = flat_encoded + vq_error
        else:
            quantized = hard_quantized
        
        # Calculate perplexity
        encodings = torch.zeros(flat_encoded.shape[0], self.num_embeddings, device=flat_encoded.device)
        encodings.scatter_(1, min_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))
        
        # Update usage counter
        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1
        
        # Reshape back
        quantized = quantized.reshape(B, S, D)
        indices = min_indices.reshape(B, S)
        
        return quantized, indices, perplexity
    
    def forward(
        self,
        features_t0: torch.Tensor,  # [B, H, W, D] or [B, N, D]
        features_t1: torch.Tensor,
        return_decoded: bool = True,
    ) -> tuple:
        """
        Encode feature difference and quantize to action codes.
        
        Args:
            features_t0: Features of first frame
            features_t1: Features of second frame
            return_decoded: Whether to return decoded features
            
        Returns:
            quantized: [B, code_seq_len, embedding_dim] or decoded [B, H, W, input_dim]
            perplexity: Codebook utilization metric
            codebook_usage: Usage count per codebook entry
            indices: [B, code_seq_len] action code indices
        """
        # Get spatial dimensions
        if features_t0.ndim == 4:
            H, W = features_t0.shape[1:3]
        else:
            N = features_t0.shape[1]
            H = W = int(math.sqrt(N))
        
        # Compute feature difference
        delta = features_t1 - features_t0
        
        # Encode to tokens
        encoded = self.encode(delta)  # [B, code_seq_len, embedding_dim]
        
        # Quantize
        quantized, indices, perplexity = self.quantize(encoded, training=self.training)
        
        if return_decoded:
            decoded = self.decode(quantized, H, W)  # [B, H, W, input_dim]
            return decoded, perplexity, self.codebooks_used.cpu().numpy(), indices
        
        return quantized, perplexity, self.codebooks_used.cpu().numpy(), indices
    
    def inference(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        user_action_token: torch.Tensor = None,
    ) -> tuple:
        """
        Inference mode: quantize feature difference or use provided action.
        
        Args:
            features_t0: Features of first frame
            features_t1: Features of second frame
            user_action_token: Optional user-specified action indices
            
        Returns:
            quantized: Quantized features
            indices: Action code indices
        """
        # Get spatial dimensions
        if features_t0.ndim == 4:
            H, W = features_t0.shape[1:3]
        else:
            N = features_t0.shape[1]
            H = W = int(math.sqrt(N))
        
        if user_action_token is not None:
            # Use user-specified action
            indices = user_action_token
            quantized = self.codebooks[indices]
            decoded = self.decode(quantized, H, W)
            return decoded, indices
        
        # Compute from features
        delta = features_t1 - features_t0
        encoded = self.encode(delta)
        
        # Hard quantization (no noise)
        B, S, D = encoded.shape
        flat_encoded = encoded.reshape(-1, D)
        
        distances = (
            torch.sum(flat_encoded ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_encoded, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        
        min_indices = torch.argmin(distances, dim=1)
        quantized = self.codebooks[min_indices].reshape(B, S, D)
        indices = min_indices.reshape(B, S)
        
        decoded = self.decode(quantized, H, W)
        return decoded, indices
    
    def replace_unused_codebooks(self, num_batches: int):
        """
        Replace unused codebook entries with used ones.
        
        Should be called periodically during training.
        """
        with torch.no_grad():
            unused_indices = torch.where(
                (self.codebooks_used.cpu() / num_batches) < self.discarding_threshold
            )[0]
            used_indices = torch.where(
                (self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold
            )[0]
            
            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]
            
            if used_count == 0:
                print('####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device)
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used
                
                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[:unused_count] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device
                )
            
            print(f'************* Replaced {unused_count} codebooks *************')
            self.codebooks_used[:] = 0
