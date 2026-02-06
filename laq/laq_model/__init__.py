# Original LAQ (ViT-based pixel prediction)
from laq_model.latent_action_quantization import LatentActionQuantization
from laq_model.laq_trainer import LAQTrainer

# Concerto-LAQ (2D+3D latent space prediction)
from laq_model.concerto_laq import ConcertoLAQ, LatentActionQuantizationConcerto
from laq_model.concerto_trainer import ConcertoLAQTrainer
from laq_model.concerto_wrapper import ConcertoEncoder, DepthEstimator, PointCloudLifter
from laq_model.latent_nsvq import LatentSpaceNSVQ
from laq_model.concerto_data import ConcertoVideoDataset, PrecomputedFeatureDataset

# Alternative VQ implementations
from laq_model.pq_vae import LatentPQVAE, ProductQuantizer
from laq_model.ed_vae import LatentEDVAE, EvidentialQuantizer
from laq_model.gumbel_vq import GumbelLatentVQ
from laq_model.vq_ema import LatentVQVAE
