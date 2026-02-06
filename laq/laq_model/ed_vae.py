"""
Evidential Discrete VAE (edVAE) for Latent Action Quantization.

edVAE uses evidential deep learning instead of softmax to obtain probability
distributions over codebook embeddings. This prevents overconfident assignments
and promotes more liberal codebook usage, effectively mitigating codebook collapse.

Based on: "EdVAE: Mitigating Codebook Collapse with Evidential Discrete VAE"
https://openreview.net/forum?id=X38Z4L74Re
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EvidentialQuantizer(nn.Module):
    """
    Evidential Quantizer using Dirichlet-based probability distributions.
    
    Instead of softmax(logits / τ), we use:
    - α = exp(logits) + 1  (evidence parameters)
    - Dir(α) as the distribution over codebook entries
    
    This provides better calibrated uncertainty and prevents overconfident
    assignments that lead to codebook collapse.
    
    Args:
        num_embeddings: Number of codebook entries (K)
        embedding_dim: Dimension of each entry (D)
        commitment_cost: Weight for commitment loss
        prior_strength: Strength of uniform Dirichlet prior (higher = more uniform)
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
        prior_strength: float = 1.0,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.prior_strength = prior_strength
        
        # Learnable codebook
        self.embedding = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.1
        )
        
        # Logit projection (instead of distance-based)
        self.logit_proj = nn.Linear(embedding_dim, num_embeddings)
        
        # Temperature for sampling sharpness
        self.register_buffer("temperature", torch.tensor(1.0))
    
    def compute_evidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute evidence (Dirichlet concentration parameters) from input.
        
        Args:
            x: [B*N, D] input tensor
            
        Returns:
            alpha: [B*N, K] Dirichlet concentration parameters
        """
        # Project to logits
        logits = self.logit_proj(x)  # [B*N, K]
        
        # Evidence = exp(logits) + 1 (ensures α > 1)
        evidence = F.softplus(logits) + 1.0
        
        return evidence
    
    def sample_from_dirichlet(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Sample probabilities from Dirichlet distribution.
        
        During training, we use the reparameterization trick.
        During inference, we use the mode of the Dirichlet.
        
        Args:
            alpha: [B*N, K] concentration parameters
            
        Returns:
            probs: [B*N, K] probability distribution
        """
        if self.training:
            # Sample using Gamma reparameterization
            # Dir(α) = Gamma(α, 1) / sum(Gamma(α, 1))
            gamma_samples = torch.distributions.Gamma(alpha, 1.0).rsample()
            probs = gamma_samples / (gamma_samples.sum(dim=-1, keepdim=True) + 1e-10)
        else:
            # Use Dirichlet mode: (α - 1) / (sum(α) - K)
            alpha_sum = alpha.sum(dim=-1, keepdim=True)
            probs = (alpha - 1) / (alpha_sum - self.num_embeddings + 1e-10)
            probs = F.relu(probs)  # Ensure non-negative
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        return probs
    
    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Forward pass with evidential quantization.
        
        Args:
            inputs: [B, N, D] input tensor
            
        Returns:
            quantized: [B, N, D] quantized output
            commitment_loss: Scalar loss
            indices: [B, N] codebook indices
            perplexity: Codebook utilization metric
        """
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)  # [B*N, D]
        
        # Compute evidence (Dirichlet parameters)
        alpha = self.compute_evidence(flat_input)  # [B*N, K]
        
        # Sample/compute probabilities
        probs = self.sample_from_dirichlet(alpha)  # [B*N, K]
        
        # Hard indices for output
        indices = torch.argmax(probs, dim=-1)
        quantized_hard = F.embedding(indices, self.embedding)
        
        if self.training:
            # Soft quantization during training
            quantized_soft = torch.matmul(probs, self.embedding)  # [B*N, D]
            
            # Straight-through with probability-weighted combination
            quantized = flat_input + (quantized_soft - flat_input).detach()
        else:
            quantized = quantized_hard
        
        # Commitment loss
        commitment_loss = F.mse_loss(quantized_hard.detach(), flat_input)
        
        # KL divergence to uniform Dirichlet prior
        # D_KL(Dir(α) || Dir(1)) encourages uncertainty
        alpha_0 = alpha.sum(dim=-1, keepdim=True)  # [B*N, 1]
        prior_alpha = torch.ones_like(alpha) * self.prior_strength
        prior_alpha_0 = prior_alpha.sum(dim=-1, keepdim=True)
        
        # Simplified KL for Dirichlet
        kl_loss = (
            torch.lgamma(alpha_0) - torch.lgamma(prior_alpha_0)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True) 
            + torch.lgamma(prior_alpha).sum(dim=-1, keepdim=True)
            + ((alpha - prior_alpha) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=-1, keepdim=True)
        ).mean()
        
        # Add KL to commitment loss
        commitment_loss = commitment_loss + 0.01 * kl_loss
        
        # Perplexity from average probabilities
        avg_probs = torch.mean(probs, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape outputs
        quantized = quantized.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])
        
        return quantized, commitment_loss, indices, perplexity


class LatentEDVAE(nn.Module):
    """
    Latent edVAE for Concerto features.
    
    Uses Evidential Discrete VAE to prevent codebook collapse.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        code_seq_len: int = 4,
        feature_size: tuple = (14, 14),
        commitment_cost: float = 0.25,
        prior_strength: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.code_seq_len = code_seq_len
        self.feature_size = feature_size
        
        # Encoder with bottleneck
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
        
        # Evidential Quantizer
        self.vq = EvidentialQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            prior_strength=prior_strength,
        )
        
        # Decoder
        self.decoder = nn.Sequential(
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
        """Return codebook for compatibility."""
        return self.vq.embedding
    
    @property
    def temperature(self):
        return self.vq.temperature
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [B, H, W, D] to [B, code_seq_len, embedding_dim]."""
        B, H, W, D = x.shape
        
        # Add input noise for regularization
        if self.training:
            x = x + 0.01 * torch.randn_like(x)
        
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)  # [B, embedding_dim, H, W]
        x = self.aggregation(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        # Normalize for stable evidence computation
        x = F.normalize(x, dim=-1) * math.sqrt(self.embedding_dim)
        
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
        Compute action codes from feature difference.
        
        Returns: (decoded, perplexity, commitment_loss, indices)
        """
        B, H, W, D = features_t0.shape
        
        # Feature difference
        delta = features_t1 - features_t0
        
        # Encode
        encoded = self.encode(delta)
        
        # Quantize
        quantized, commitment_loss, indices, perplexity = self.vq(encoded)
        
        if return_decoded:
            decoded = self.decode(quantized, H, W)
            return decoded, perplexity, commitment_loss, indices
        
        return quantized, perplexity, commitment_loss, indices
    
    def inference(
        self,
        features_t0: torch.Tensor,
        features_t1: torch.Tensor,
        user_action_token: torch.Tensor = None,
    ) -> tuple:
        B, H, W, D = features_t0.shape
        
        if user_action_token is not None:
            quantized = F.embedding(user_action_token, self.vq.embedding)
            decoded = self.decode(quantized, H, W)
            return decoded, user_action_token
        
        delta = features_t1 - features_t0
        encoded = self.encode(delta)
        quantized, _, indices, _ = self.vq(encoded)
        decoded = self.decode(quantized, H, W)
        
        return decoded, indices
    
    def replace_unused_codebooks(self, num_batches: int):
        """No-op for API compatibility - edVAE doesn't need explicit replacement."""
        pass
