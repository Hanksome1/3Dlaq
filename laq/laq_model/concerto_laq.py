"""
Concerto-based Latent Action Quantization (ConcertoLAQ).

Main model that combines Concerto's 2D+3D aware features with LAQ's
latent action prediction framework. This enables learning discrete
action codes from video with spatial and geometric understanding.

Architecture:
1. Concerto Encoder (frozen) - Extract 2D+3D aware features from video frames
2. Feature Projection - Project Concerto features to model dimension
3. Temporal Encoder - Process temporal difference in feature space
4. Action Quantizer (NSVQ) - Discretize actions into codebook indices
5. Latent Predictor (training) - Predict next frame's latent for loss computation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from einops import rearrange, repeat

from laq_model.attention import Transformer, ContinuousPositionBias
from laq_model.concerto_wrapper import ConcertoEncoder
from laq_model.latent_nsvq import LatentSpaceNSVQ
from laq_model.vq_ema import LatentVQVAE


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


class LatentPredictor(nn.Module):
    """
    Predict next frame's latent features from current features and action codes.
    
    Uses transformer decoder architecture with cross-attention to action tokens.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Self-attention on spatial features
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                # Cross-attention to action tokens
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                # Feed-forward
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim),
                )
            ]))
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        features: torch.Tensor,  # [B, N, D] current frame features
        action_tokens: torch.Tensor,  # [B, S, D] quantized action tokens
    ) -> torch.Tensor:
        """
        Predict next frame's latent features.
        
        Args:
            features: Current frame features [B, N, D]
            action_tokens: Quantized action tokens [B, S, D]
            
        Returns:
            predicted: Predicted next frame features [B, N, D]
        """
        x = features
        
        for self_norm, self_attn, cross_norm, cross_attn, ff in self.layers:
            # Self-attention
            x_norm = self_norm(x)
            attn_out, _ = self_attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # Cross-attention to actions
            x_norm = cross_norm(x)
            attn_out, _ = cross_attn(x_norm, action_tokens, action_tokens)
            x = x + attn_out
            
            # Feed-forward
            x = x + ff(x)
        
        return self.final_norm(x)


class ConcertoLAQ(nn.Module):
    """
    Concerto-based Latent Action Quantization.
    
    This model learns discrete action codes from video by:
    1. Extracting 2D+3D aware features using frozen Concerto
    2. Computing temporal difference in latent space
    3. Quantizing the difference to discrete action codes
    4. (Training) Predicting next frame's latent given current + action
    
    Args:
        concerto_model_name: Concerto model size ("concerto_small", "concerto_base", "concerto_large")
        concerto_dim: Concerto output dimension (256 for base, 512 for large)
        dim: Internal model dimension
        codebook_size: Number of action codes in vocabulary
        code_seq_len: Number of action tokens per frame transition
        spatial_depth: Depth of spatial transformer encoder
        temporal_depth: Depth of temporal difference encoder
        predictor_depth: Depth of latent predictor (for training)
        dim_head: Attention head dimension
        heads: Number of attention heads
        feature_size: Spatial size of Concerto features (after downsampling)
        freeze_concerto: Whether to freeze Concerto weights
        use_depth: Whether to use depth estimation for 3D lifting
        depth_model_type: Type of depth estimator ("depth_anything", "zoedepth", "dummy")
        use_precomputed_features: If True, expects pre-computed Concerto features as input
    """
    
    def __init__(
        self,
        *,
        concerto_model_name: str = "concerto_base",
        concerto_dim: int = 256,
        dim: int = 1024,
        quant_dim: int = 32,
        codebook_size: int = 8,
        code_seq_len: int = 4,
        image_size: int = 256,
        feature_size: Tuple[int, int] = (8, 8),
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        predictor_depth: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        freeze_concerto: bool = True,
        use_depth: bool = True,
        depth_model_type: str = "dummy",
        use_precomputed_features: bool = False,
        use_ema_vq: bool = True,  # Use VQ-EMA instead of NSVQ
    ):
        super().__init__()
        
        self.image_size = pair(image_size)
        self.feature_size = pair(feature_size)
        self.code_seq_len = code_seq_len
        self.dim = dim
        self.concerto_dim = concerto_dim
        self.codebook_size = codebook_size
        self.use_precomputed_features = use_precomputed_features
        self.freeze_concerto = freeze_concerto
        
        # Concerto encoder (frozen by default)
        if not use_precomputed_features:
            self.concerto = ConcertoEncoder(
                concerto_model_name=concerto_model_name,
                freeze_concerto=freeze_concerto,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.depth_estimator = self.concerto.vggt  # VGGT handles depth internally
        else:
            self.concerto = None
            self.depth_estimator = None

        
        # Feature projection: Concerto dim -> model dim
        self.feature_proj = nn.Sequential(
            nn.Linear(concerto_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Spatial position bias for attention
        self.spatial_rel_pos_bias = ContinuousPositionBias(
            dim=dim,
            heads=heads,
            num_dims=2,
        )
        
        # Spatial encoder - process features within each frame
        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=False,
        )
        self.spatial_encoder = Transformer(depth=spatial_depth, **transformer_kwargs)
        
        # Temporal difference encoder - process the delta between frames
        self.temporal_encoder = Transformer(depth=temporal_depth, **transformer_kwargs)
        
        # Action quantizer - choose between NSVQ and VQ-EMA
        self.use_ema_vq = use_ema_vq
        if use_ema_vq:
            # VQ-EMA: More stable for low-dimensional latent spaces
            self.action_quantizer = LatentVQVAE(
                input_dim=dim,
                embedding_dim=quant_dim,
                num_embeddings=codebook_size,
                code_seq_len=code_seq_len,
                feature_size=feature_size,
            )
        else:
            # NSVQ: Original method from LAQ
            self.action_quantizer = LatentSpaceNSVQ(
                input_dim=dim,
                embedding_dim=quant_dim,
                num_embeddings=codebook_size,
                code_seq_len=code_seq_len,
                feature_size=feature_size,
            )
        
        # Latent predictor for training loss
        self.latent_predictor = LatentPredictor(
            dim=dim,
            depth=predictor_depth,
            dim_head=dim_head,
            heads=heads,
        )
        
        # Output projection (maps quantized action back to feature space for prediction)
        self.action_proj = nn.Sequential(
            nn.Linear(quant_dim, dim),
            nn.LayerNorm(dim),
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict=False)
    
    def extract_features(
        self,
        video: torch.Tensor,  # [B, C, 2, H, W]
        depth: Optional[torch.Tensor] = None,  # [B, 2, H, W] (ignored, VGGT handles internally)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Concerto features from video frames.
        
        Note: depth parameter is kept for API compatibility but is ignored.
        VGGT handles depth estimation internally.
        
        Args:
            video: Two consecutive RGB frames
            depth: Ignored (kept for API compatibility)
            
        Returns:
            features_t0: [B, H', W', concerto_dim]
            features_t1: [B, H', W', concerto_dim]
        """
        if self.concerto is None:
            raise ValueError("Concerto encoder not initialized. Use use_precomputed_features=False")
        
        # Extract features using Concerto (frozen)
        # VGGT handles depth estimation internally
        if self.freeze_concerto:
            with torch.no_grad():
                features = self.concerto(video)  # [B, 2, H', W', D]
        else:
            features = self.concerto(video)
        
        features_t0 = features[:, 0]
        features_t1 = features[:, 1]
        
        return features_t0, features_t1

    
    def encode_features(
        self,
        features: torch.Tensor,  # [B, H, W, D]
    ) -> torch.Tensor:
        """
        Encode spatial features using transformer.
        
        Args:
            features: [B, H, W, concerto_dim]
            
        Returns:
            encoded: [B, H, W, dim]
        """
        B, H, W, D = features.shape
        
        # Project to model dimension
        features = self.feature_proj(features)  # [B, H, W, dim]
        
        # Reshape for transformer: [B, H*W, dim]
        tokens = rearrange(features, 'b h w d -> b (h w) d')
        
        # Compute position bias
        attn_bias = self.spatial_rel_pos_bias(H, W, device=features.device)
        
        # Apply spatial encoder
        encoded = self.spatial_encoder(tokens, attn_bias=attn_bias)
        
        # Reshape back to spatial
        encoded = rearrange(encoded, 'b (h w) d -> b h w d', h=H, w=W)
        
        return encoded
    
    def compute_action(
        self,
        features_t0: torch.Tensor,  # [B, H, W, dim]
        features_t1: torch.Tensor,  # [B, H, W, dim]
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Compute and quantize action from feature difference.
        
        Args:
            features_t0: Encoded features of first frame
            features_t1: Encoded features of second frame
            step: Training step (for codebook management)
            
        Returns:
            action_tokens: [B, code_seq_len, quant_dim] quantized actions
            indices: [B, code_seq_len] codebook indices
            perplexity: Codebook utilization
            decoded_delta: [B, H, W, dim] decoded feature difference
            commitment_loss: VQ commitment loss (only for VQ-EMA)
        """
        B, H, W, D = features_t0.shape
        
        # Compute difference in encoded space
        delta = features_t1 - features_t0  # [B, H, W, dim]
        
        # Flatten for temporal encoder
        delta_tokens = rearrange(delta, 'b h w d -> b (h w) d')
        
        # Apply temporal encoder
        encoded_delta = self.temporal_encoder(delta_tokens)
        
        # Reshape for quantization
        encoded_delta = rearrange(encoded_delta, 'b (h w) d -> b h w d', h=H, w=W)
        
        # Quantize to action codes
        if self.use_ema_vq:
            # VQ-EMA returns: (decoded, perplexity, commitment_loss, indices)
            decoded_delta, perplexity, commitment_loss, indices = self.action_quantizer(
                features_t0, features_t1
            )
        else:
            # NSVQ returns: (decoded, perplexity, codebook_usage, indices)
            decoded_delta, perplexity, codebook_usage, indices = self.action_quantizer(
                features_t0, features_t1
            )
            commitment_loss = torch.tensor(0.0, device=decoded_delta.device)
            
            # Handle codebook replacement during training (NSVQ only)
            if self.training:
                if ((step % 10 == 0 and step < 100) or 
                    (step % 100 == 0 and step < 1000) or 
                    (step % 500 == 0 and step < 5000)) and step != 0:
                    print(f"update codebook {step}")
                    self.action_quantizer.replace_unused_codebooks(decoded_delta.shape[0])
        
        # Get quantized action tokens
        action_tokens = self.action_quantizer.codebooks[indices]  # [B, code_seq_len, quant_dim]
        
        return action_tokens, indices, perplexity, decoded_delta, commitment_loss
    
    def predict_next_frame(
        self,
        features_t0: torch.Tensor,  # [B, H, W, dim]
        action_tokens: torch.Tensor,  # [B, code_seq_len, quant_dim]
    ) -> torch.Tensor:
        """
        Predict next frame's features given current frame and action.
        
        Args:
            features_t0: Current frame features
            action_tokens: Quantized action tokens
            
        Returns:
            predicted: [B, H, W, dim] predicted next frame features
        """
        B, H, W, D = features_t0.shape
        
        # Project action tokens to model dimension
        action_projected = self.action_proj(action_tokens)  # [B, code_seq_len, dim]
        
        # Flatten spatial features
        features_flat = rearrange(features_t0, 'b h w d -> b (h w) d')
        
        # Predict using cross-attention
        predicted = self.latent_predictor(features_flat, action_projected)
        
        # Reshape to spatial
        predicted = rearrange(predicted, 'b (h w) d -> b h w d', h=H, w=W)
        
        return predicted
    
    def forward(
        self,
        video: Optional[torch.Tensor] = None,  # [B, C, 2, H, W]
        features: Optional[torch.Tensor] = None,  # [B, 2, H', W', concerto_dim] pre-computed
        depth: Optional[torch.Tensor] = None,  # [B, 2, H, W]
        step: int = 0,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Forward pass for training or inference.
        
        Args:
            video: Raw video frames (if not using pre-computed features)
            features: Pre-computed Concerto features
            depth: Optional depth maps
            step: Training step
            return_recons_only: Return only predicted features
            return_only_codebook_ids: Return only action indices
            
        Returns:
            If training: (loss, num_unique_indices)
            If return_only_codebook_ids: indices
            If return_recons_only: predicted features
        """
        # Get Concerto features
        if features is not None:
            # Pre-computed features provided
            features_t0 = features[:, 0]  # [B, H', W', concerto_dim]
            features_t1 = features[:, 1]
        elif video is not None:
            # Extract features from video
            features_t0, features_t1 = self.extract_features(video, depth)
        else:
            raise ValueError("Either video or features must be provided")
        
        # Encode features through spatial transformer
        encoded_t0 = self.encode_features(features_t0)  # [B, H, W, dim]
        encoded_t1 = self.encode_features(features_t1)
        
        # Compute and quantize action
        action_tokens, indices, perplexity, decoded_delta, commitment_loss = self.compute_action(
            encoded_t0, encoded_t1, step
        )
        
        num_unique_indices = indices.unique().size(0)
        
        if return_only_codebook_ids:
            return indices
        
        # Predict next frame's features
        predicted_t1 = self.predict_next_frame(encoded_t0, action_tokens)
        
        if return_recons_only:
            return predicted_t1
        
        # Compute losses
        reconstruction_loss = F.mse_loss(predicted_t1, encoded_t1.detach())
        
        # ===== Diversity regularization to prevent codebook collapse =====
        flat_indices = indices.view(-1)
        batch_unique = flat_indices.unique()
        num_unique_indices = batch_unique.size(0)
        
        # 1. Entropy-based diversity loss (batch level)
        counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        probs = counts / (counts.sum() + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(self.codebook_size), device=entropy.device))
        normalized_entropy = entropy / max_entropy
        entropy_loss = 1.0 - normalized_entropy
        
        # 2. Codebook spread loss: encourage codebook vectors to be spread out
        if hasattr(self, 'action_quantizer'):
            codebook = self.action_quantizer.codebooks  # [K, D]
            K, D = codebook.shape
            
            # Compute pairwise distances between codebook entries
            codebook_norm = codebook / (codebook.norm(dim=1, keepdim=True) + 1e-8)
            similarity_matrix = torch.mm(codebook_norm, codebook_norm.t())  # [K, K]
            
            # Mask out diagonal (self-similarity)
            mask = torch.eye(K, device=similarity_matrix.device).bool()
            similarity_matrix = similarity_matrix.masked_fill(mask, -1)
            
            # Penalize high similarity between different codebook entries
            max_similarity = similarity_matrix.max(dim=1)[0].mean()
            spread_loss = max_similarity + 1.0  # Range [0, 2]
        else:
            spread_loss = torch.tensor(0.0, device=reconstruction_loss.device)
        
        # 3. Usage balance loss
        if num_unique_indices > 1:
            usage_variance = probs[probs > 0].var()
            balance_loss = usage_variance * 10.0
        else:
            balance_loss = torch.tensor(1.0, device=reconstruction_loss.device)
        
        # Total loss - include commitment_loss for VQ-EMA
        entropy_weight = 2.0
        spread_weight = 0.5
        balance_weight = 0.1
        commitment_weight = 0.25  # Standard VQ-VAE commitment weight
        
        total_loss = (
            reconstruction_loss 
            + entropy_weight * entropy_loss 
            + spread_weight * spread_loss
            + balance_weight * balance_loss
            + commitment_weight * commitment_loss
        )
        
        # Compute codebook utilization rate
        codebook_utilization = num_unique_indices / self.codebook_size
        
        # Build metrics dictionary
        metrics = {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'commitment_loss': commitment_loss,
            'entropy_loss': entropy_loss,
            'spread_loss': spread_loss,
            'balance_loss': balance_loss,
            'entropy': entropy,
            'num_unique_codes': num_unique_indices,
            'codebook_utilization': codebook_utilization,
            'perplexity': perplexity.mean() if perplexity.numel() > 0 else torch.tensor(0.0),
        }
        
        return total_loss, num_unique_indices, metrics
    
    def inference(
        self,
        video: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        user_action_token_num: Optional[Union[int, list]] = None,
        return_only_codebook_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inference mode: extract action codes from video or apply user-specified action.
        
        Args:
            video: Raw video frames
            features: Pre-computed Concerto features
            depth: Optional depth maps
            user_action_token_num: Optional user-specified action indices
            return_only_codebook_ids: Return only action indices
            
        Returns:
            If return_only_codebook_ids: indices
            Otherwise: predicted features
        """
        self.eval()
        
        with torch.no_grad():
            # Get Concerto features
            if features is not None:
                features_t0 = features[:, 0]
                features_t1 = features[:, 1]
            elif video is not None:
                features_t0, features_t1 = self.extract_features(video, depth)
            else:
                raise ValueError("Either video or features must be provided")
            
            # Encode features
            encoded_t0 = self.encode_features(features_t0)
            encoded_t1 = self.encode_features(features_t1)
            
            if user_action_token_num is not None:
                # Use user-specified action
                if isinstance(user_action_token_num, list):
                    indices = torch.tensor(user_action_token_num, device=self.device)
                else:
                    B = encoded_t0.shape[0]
                    indices = torch.full(
                        (B, self.code_seq_len), 
                        user_action_token_num, 
                        device=self.device
                    )
                action_tokens = self.action_quantizer.codebooks[indices]
            else:
                # Compute action from features
                action_tokens, indices, _, _ = self.compute_action(encoded_t0, encoded_t1)
            
            if return_only_codebook_ids:
                return indices
            
            # Predict next frame
            predicted_t1 = self.predict_next_frame(encoded_t0, action_tokens)
            
            return predicted_t1


# Backward compatibility alias
LatentActionQuantizationConcerto = ConcertoLAQ
