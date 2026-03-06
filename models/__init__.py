"""
Models Package
模型模块初始化
"""

from models.ema_codebook import EMAVectorQuantizer, SoftQuantizer
from models.rq_vae import (
    RQVAE, RQVAEEncoder, RQVAEDecoder, ResidualQuantizer,
    convert_ids_to_string, compute_semantic_similarity
)
from models.mmoe_encoder import (
    MMOEEncoder, MultiModalFeatureEncoder, ExpertNetwork, GateNetwork,
    ModalityAlignmentLoss, ModalityDisentangleLoss
)
from models.recommender import (
    SemanticRecommender, SemanticIDEmbedding, TransformerSequenceEncoder,
    ColdStartHandler, RecommendationLoss
)
from models.behavior_aware_finetuning import (
    BehaviorAwareRQVAE, BehaviorAwareFineTuner, JointLoss,
    SoftIndexQuantizer, StraightThroughEstimator,
    create_behavior_aware_model
)

__all__ = [
    # EMA Codebook
    'EMAVectorQuantizer',
    'SoftQuantizer',
    
    # RQ-VAE
    'RQVAE',
    'RQVAEEncoder',
    'RQVAEDecoder',
    'ResidualQuantizer',
    'convert_ids_to_string',
    'compute_semantic_similarity',
    
    # MMOE Encoder
    'MMOEEncoder',
    'MultiModalFeatureEncoder',
    'ExpertNetwork',
    'GateNetwork',
    'ModalityAlignmentLoss',
    'ModalityDisentangleLoss',
    
    # Recommender
    'SemanticRecommender',
    'SemanticIDEmbedding',
    'TransformerSequenceEncoder',
    'ColdStartHandler',
    'RecommendationLoss',
    
    # Behavior-Aware Fine-tuning
    'BehaviorAwareRQVAE',
    'BehaviorAwareFineTuner',
    'JointLoss',
    'SoftIndexQuantizer',
    'StraightThroughEstimator',
    'create_behavior_aware_model'
]
