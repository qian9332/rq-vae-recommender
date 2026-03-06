"""
RQ-VAE Recommender
基于残差量化的多模态推荐系统

核心特性:
1. 残差量化VAE (RQ-VAE): 将Item多模态特征编码为离散语义ID序列
2. EMA码本更新 + 死码重置: 码本利用率从10%-20%提升至85%+
3. MMOE多模态编码器: 实现文本和视觉模态的协同与解耦
4. 行为感知微调: 软索引+STE允许梯度回传至RQ-VAE
"""

__version__ = "1.0.0"
__author__ = "RQ-VAE Team"

from models import (
    RQVAE,
    EMAVectorQuantizer,
    MultiModalFeatureEncoder,
    SemanticRecommender,
    BehaviorAwareFineTuner,
    create_behavior_aware_model
)

from configs import Config, default_config
from data import generate_synthetic_data, create_dataloaders
from utils import MetricsCalculator, evaluate_model

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Models
    'RQVAE',
    'EMAVectorQuantizer',
    'MultiModalFeatureEncoder',
    'SemanticRecommender',
    'BehaviorAwareFineTuner',
    'create_behavior_aware_model',
    
    # Config
    'Config',
    'default_config',
    
    # Data
    'generate_synthetic_data',
    'create_dataloaders',
    
    # Utils
    'MetricsCalculator',
    'evaluate_model'
]
