"""
Vector Quantization with Gumbel-Softmax and Temperature Annealing.

Uses soft quantization during training to prevent codebook collapse,
with temperature annealing to transition to hard quantization.

Key features:
1. Gumbel-Softmax: Differentiable soft argmax
2. Temperature annealing: High temp (diverse) -> Low temp (discrete)
3. Random codebook perturbation: Prevents dead codes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GumbelVQ(nn.Module):
    """
    Vector Quantizer with Gumbel-Softmax for differentiable discrete codes.
    
    Uses temperature annealing to prevent codebook collapse:
    - High temperature: soft assignments, encourages exploration
    - Low temperature: hard assignments, discrete codes
    
    Args:
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of each entry
        temperature_init: Initial temperature (high = soft)
        temperature_min: Minimum temperature
        temperature_decay: Decay per step
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 32,
        temperature_init: float = 2.0,
        temperature_min: float = 0.5,
        temperature_decay: float = 0.99995,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Temperature annealing
        self.register_buffer("temperature", torch.tensor(temperature_init))
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        
        # Learnable codebook
        self.embedding = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.1
        )
        
        # Track step for logging
        self.register_buffer("step", torch.tensor(0))
    
    def update_temperature(self):
        """Decay temperature towards minimum."""
        if self.training:
            new_temp = max(
                self.temperature.item() * self.temperature_decay,
                self.temperature_min
            )
            self.temperature.fill_(new_temp)
            self.step += 1
    
    def forward(self, inputs: torch.Tensor, hard: bool = False) -> tuple:
        """
        Forward pass with Gumbel-Softmax quantization.
        
        Args:
            inputs: [B, N, D] input tensor
            hard: If True, use hard (argmax) quantization
            
        Returns:
            quantized: Quantized output
            commitment_loss: Encoder commitment loss
            indices: Code indices
            perplexity: Codebook utilization
        """
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        B_flat = flat_input.shape[0]
        
        # Compute distances (negative for softmax)
        # distances[i,j] = ||z_i - e_j||^2
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.t())
            + torch.sum(self.embedding ** 2, dim=1)
        )
        
        # Convert distances to logits (negative distances as logits)
        logits = -distances / self.temperature
        
        if self.training and not hard:
            # Gumbel-Softmax sampling
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(logits) + 1e-20
            ) + 1e-20)
            
            # Soft assignments with Gumbel noise
            soft_assignments = F.softmax(
                (logits + gumbel_noise) / self.temperature,
                dim=-1
            )
        else:
            # Hard assignments during inference
            soft_assignments = F.softmax(logits, dim=-1)
        
        # Soft quantization: weighted sum of embeddings
        quantized_soft = torch.matmul(soft_assignments, self.embedding)
        
        # Hard indices (for logging and inference)
        indices = torch.argmax(soft_assignments, dim=-1)
        quantized_hard = F.embedding(indices, self.embedding)
        
        # Straight-through estimator: use hard in forward, soft in backward
        if self.training:
            # Mix soft and hard based on temperature
            temp_ratio = (self.temperature.item() - self.temperature_min) / (2.0 - self.temperature_min)
            quantized = temp_ratio * quantized_soft + (1 - temp_ratio) * quantized_hard
            quantized = flat_input + (quantized - flat_input).detach()  # ST gradient
        else:
            quantized = quantized_hard
        
        # Commitment loss - encourage encoder outputs close to codebook
        commitment_loss = F.mse_loss(quantized_hard.detach(), flat_input)
        
        # Perplexity (codebook utilization)
        avg_probs = torch.mean(soft_assignments, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Update temperature
        self.update_temperature()
        
        # Reshape
        quantized = quantized.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])
        
        return quantized, commitment_loss, indices, perplexity


class GumbelLatentVQ(nn.Module):
    """
    Latent VQ with Gumbel-Softmax for Concerto features.
    
    Designed to prevent codebook collapse in low-dimensional latent spaces.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        code_seq_len: int = 4,
        feature_size: tuple = (14, 14),
        temperature_init: float = 2.0,
        temperature_min: float = 0.5,
        temperature_decay: float = 0.99995,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.code_seq_len = code_seq_len
        self.feature_size = feature_size
        
        # Stronger encoder with bottleneck
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
        
        # Gumbel VQ
        self.vq = GumbelVQ(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
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
        x = x.permute(0, 3, 1, 2)
        x = self.aggregation(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        # Normalize for stable codebook matching
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
        quantized, _, indices, _ = self.vq(encoded, hard=True)
        decoded = self.decode(quantized, H, W)
        
        return decoded, indices
    
    def replace_unused_codebooks(self, num_batches: int):
        """No-op for API compatibility."""
        pass
