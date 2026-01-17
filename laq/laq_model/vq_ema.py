"""
Vector Quantization with Exponential Moving Average (VQ-EMA).

A more robust VQ implementation that uses EMA updates for codebook learning,
avoiding the instability issues with NSVQ for low-dimensional latent spaces.

Based on the original VQ-VAE paper:
https://arxiv.org/abs/1711.00937
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates.
    
    This is more stable than gradient-based updates for VQ learning.
    
    Args:
        num_embeddings: Number of codebook entries (K)
        embedding_dim: Dimension of each codebook entry (D)
        commitment_cost: Weight for commitment loss
        decay: EMA decay rate (0.99 is common)
        epsilon: Small constant for numerical stability
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook embeddings
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        
        # EMA cluster size and embedding sum
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_sum", embedding.clone())
        
        # Track if first batch (for initialization)
        self.register_buffer("initialized", torch.tensor(False))
    
    def _init_from_data(self, flat_input: torch.Tensor):
        """Initialize codebook from first batch of data."""
        if self.initialized:
            return
        
        # Use random samples from input to initialize codebook
        N = flat_input.shape[0]
        if N >= self.num_embeddings:
            indices = torch.randperm(N)[:self.num_embeddings]
            self.embedding.data.copy_(flat_input[indices])
        else:
            # Repeat and add noise if not enough samples
            repeats = (self.num_embeddings // N) + 1
            data = flat_input.repeat(repeats, 1)[:self.num_embeddings]
            data = data + 0.01 * torch.randn_like(data)
            self.embedding.data.copy_(data)
        
        self.ema_embedding_sum.data.copy_(self.embedding.data)
        self.ema_cluster_size.data.fill_(1.0)
        self.initialized.fill_(True)
    
    def forward(self, inputs: torch.Tensor):
        """
        Forward pass with EMA codebook updates.
        
        Args:
            inputs: [B, N, D] or [B, H, W, D] input tensor
            
        Returns:
            quantized: Quantized output (same shape as input)
            loss: Commitment loss
            indices: [B, N] or [B, H, W] codebook indices
            perplexity: Codebook utilization metric
        """
        input_shape = inputs.shape
        
        # Flatten to [B*N, D]
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Initialize from first batch
        if self.training and not self.initialized:
            self._init_from_data(flat_input)
        
        # Compute distances: ||z - e||^2
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.t())
            + torch.sum(self.embedding ** 2, dim=1)
        )
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        
        # One-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize using straight-through estimator
        quantized = F.embedding(encoding_indices, self.embedding)
        
        # EMA codebook updates during training
        if self.training:
            # Update cluster sizes
            cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            
            # Update embedding sums
            embedding_sum = torch.matmul(encodings.t(), flat_input)
            self.ema_embedding_sum.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            
            # Normalize to get new embeddings
            n = self.ema_cluster_size.sum()
            cluster_size_normalized = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            
            self.embedding.data.copy_(
                self.ema_embedding_sum / cluster_size_normalized.unsqueeze(1)
            )
        
        # Commitment loss (encoder should commit to codebook entries)
        commitment_loss = F.mse_loss(quantized.detach(), flat_input)
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        # Compute perplexity (codebook utilization)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape back to original shape
        quantized = quantized.reshape(input_shape)
        indices = encoding_indices.reshape(input_shape[:-1])
        
        return quantized, commitment_loss, indices, perplexity


class LatentVQVAE(nn.Module):
    """
    VQ-VAE for Concerto latent space action quantization.
    
    Uses VQ-EMA for more stable codebook learning compared to NSVQ.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        code_seq_len: int = 4,
        feature_size: tuple = (14, 14),
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.code_seq_len = code_seq_len
        self.feature_size = feature_size
        
        # Encoder: project and downsample to code_seq_len tokens
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
        # Spatial aggregation
        self.aggregation = self._build_aggregation(code_seq_len, feature_size)
        
        # Vector quantizer
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        
        # Decoder: upsample and project back
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, input_dim),
        )
    
    def _build_aggregation(self, code_seq_len: int, feature_size: tuple) -> nn.Module:
        """Build spatial aggregation layer."""
        H, W = feature_size
        
        if code_seq_len == 1:
            return nn.AdaptiveAvgPool2d((1, 1))
        elif code_seq_len == 4:
            return nn.AdaptiveAvgPool2d((2, 2))
        elif code_seq_len == 16:
            return nn.AdaptiveAvgPool2d((4, 4))
        else:
            grid_size = int(math.sqrt(code_seq_len))
            return nn.AdaptiveAvgPool2d((grid_size, grid_size))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to code tokens.
        
        Args:
            x: [B, H, W, D] input features
            
        Returns:
            encoded: [B, code_seq_len, embedding_dim]
        """
        B, H, W, D = x.shape
        
        # Apply encoder MLP
        x = self.encoder(x)  # [B, H, W, embedding_dim]
        
        # Spatial aggregation
        x = x.permute(0, 3, 1, 2)  # [B, embedding_dim, H, W]
        x = self.aggregation(x)    # [B, embedding_dim, h, w]
        x = rearrange(x, 'b d h w -> b (h w) d')  # [B, code_seq_len, embedding_dim]
        
        return x
    
    def decode(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Decode quantized tokens back to spatial features.
        
        Args:
            x: [B, code_seq_len, embedding_dim]
            target_h, target_w: Target spatial dimensions
            
        Returns:
            decoded: [B, target_h, target_w, input_dim]
        """
        B = x.shape[0]
        
        # Apply decoder MLP
        x = self.decoder(x)  # [B, code_seq_len, input_dim]
        
        # Upsample to target size
        h = w = int(math.sqrt(self.code_seq_len))
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, input_dim]
        
        return x
    
    @property
    def codebooks(self):
        """Return codebook embeddings for spread loss computation."""
        return self.vq.embedding
    
    def forward(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        return_decoded: bool = True,
    ) -> tuple:
        """
        Encode feature difference and quantize to action codes.
        
        Args:
            features_t0: [B, H, W, D] features of first frame
            features_t1: [B, H, W, D] features of second frame
            
        Returns:
            decoded: [B, H, W, D] decoded action representation
            perplexity: Codebook utilization
            commitment_loss: VQ commitment loss
            indices: [B, code_seq_len] action code indices
        """
        B, H, W, D = features_t0.shape
        
        # Compute feature difference (action representation)
        delta = features_t1 - features_t0
        
        # Encode to tokens
        encoded = self.encode(delta)  # [B, code_seq_len, embedding_dim]
        
        # Quantize
        quantized, commitment_loss, indices, perplexity = self.vq(encoded)
        
        if return_decoded:
            # Decode back to spatial features
            decoded = self.decode(quantized, H, W)
            return decoded, perplexity, commitment_loss, indices
        
        return quantized, perplexity, commitment_loss, indices
    
    def inference(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        user_action_token: torch.Tensor = None,
    ) -> tuple:
        """
        Inference mode: quantize or use specified action.
        """
        B, H, W, D = features_t0.shape
        
        if user_action_token is not None:
            # Use user-specified action
            quantized = F.embedding(user_action_token, self.vq.embedding)
            decoded = self.decode(quantized, H, W)
            return decoded, user_action_token
        
        # Compute from features
        delta = features_t1 - features_t0
        encoded = self.encode(delta)
        quantized, _, indices, _ = self.vq(encoded)
        decoded = self.decode(quantized, H, W)
        
        return decoded, indices


# Keep backward compatibility with LatentSpaceNSVQ interface
class LatentVQVAECompat(LatentVQVAE):
    """Backward compatible wrapper with LatentSpaceNSVQ interface."""
    
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 32,
        num_embeddings: int = 256,
        code_seq_len: int = 4,
        feature_size: tuple = (14, 14),
        discarding_threshold: float = 0.1,  # Ignored, kept for compatibility
        initialization: str = 'normal',     # Ignored, kept for compatibility
        device: str = 'cuda',
    ):
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            code_seq_len=code_seq_len,
            feature_size=feature_size,
        )
    
    def forward(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        return_decoded: bool = True,
    ) -> tuple:
        decoded, perplexity, commitment_loss, indices = super().forward(
            features_t0, features_t1, return_decoded
        )
        
        # Return in LatentSpaceNSVQ format: (decoded, perplexity, codebook_usage, indices)
        # Create dummy codebook usage array
        codebook_usage = torch.zeros(self.num_embeddings).cpu().numpy()
        
        return decoded, perplexity, codebook_usage, indices
    
    def replace_unused_codebooks(self, num_batches: int):
        """No-op for compatibility. VQ-EMA doesn't need explicit replacement."""
        pass
