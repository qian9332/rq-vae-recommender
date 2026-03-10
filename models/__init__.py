"""
RQ-VAE Recommender Models
模型模块
"""

from .rq_vae import RQVAE
from .mmoe_encoder import MultiModalFeatureEncoder
from .ema_codebook import EMAVectorQuantizer, SoftQuantizer
from .recommender import (
    SemanticIDEmbedding,
    TransformerSequenceEncoder,
    SemanticRecommender,
    ColdStartHandler,
    RecommendationLoss
)
from .user_sequence import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    UserSequenceEncoder,
    UserBehaviorModel,
    SequentialRecommender
)
from .behavior_aware_finetuning import (
    StraightThroughEstimator,
    SoftIndexQuantizer,
    BehaviorAwareFineTuner
)

__all__ = [
    # RQ-VAE
    'RQVAE',
    
    # MMOE Encoder
    'MultiModalFeatureEncoder',
    
    # Codebook
    'EMAVectorQuantizer',
    'SoftQuantizer',
    
    # Recommender
    'SemanticIDEmbedding',
    'TransformerSequenceEncoder',
    'SemanticRecommender',
    'ColdStartHandler',
    'RecommendationLoss',
    
    # User Sequence
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock',
    'UserSequenceEncoder',
    'UserBehaviorModel',
    'SequentialRecommender',
    
    # Fine-tuning
    'StraightThroughEstimator',
    'SoftIndexQuantizer',
    'BehaviorAwareFineTuner'
]
