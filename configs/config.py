"""
RQ-VAE Recommender Configuration
配置文件：包含模型、训练、数据处理的所有超参数配置
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class CodebookConfig:
    """码本配置"""
    num_layers: int = 3  # 残差量化层数 (2-3层)
    codebook_size: int = 256  # 每层码本大小
    embedding_dim: int = 128  # 码本向量维度
    ema_decay: float = 0.99  # EMA衰减系数
    commitment_cost: float = 0.25  # commitment loss权重
    dead_code_threshold: int = 100  # 死码判定阈值（多少步未使用）
    dead_code_reset_threshold: float = 0.01  # 死码重置阈值（利用率低于此值）
    

@dataclass
class MMOEConfig:
    """MMOE多模态编码器配置"""
    text_input_dim: int = 768  # 文本特征维度 (如BERT输出)
    visual_input_dim: int = 2048  # 视觉特征维度 (如ResNet输出)
    hidden_dim: int = 256  # 隐藏层维度
    output_dim: int = 128  # 输出维度（与码本维度一致）
    num_experts: int = 4  # 专家网络数量
    num_tasks: int = 2  # 任务数量（文本任务、视觉任务）
    expert_hidden_dim: int = 128  # 专家网络隐藏层维度
    dropout: float = 0.1  # Dropout比例
    activation: str = "relu"  # 激活函数


@dataclass
class RecommenderConfig:
    """推荐模型配置"""
    num_users: int = 10537  # 用户数量 (Amazon Video_Games)
    num_items: int = 16297  # 物品数量 (Amazon Video_Games)
    user_embedding_dim: int = 128  # 用户嵌入维度
    max_seq_length: int = 50  # 最大序列长度
    num_heads: int = 4  # Transformer注意力头数
    num_layers: int = 2  # Transformer层数
    dropout: float = 0.2  # Dropout比例
    temperature: float = 0.1  # Softmax温度系数


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 100
    
    # 预训练配置
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    
    # 微调配置
    finetune_epochs: int = 30
    finetune_lr: float = 5e-5
    reconstruction_weight: float = 1.0  # 重建损失权重
    recommendation_weight: float = 1.0  # 推荐损失权重


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"
    train_file: str = "train.csv"
    valid_file: str = "valid.csv"
    test_file: str = "test.csv"
    item_features_file: str = "item_features.pkl"
    user_features_file: str = "user_features.pkl"
    text_features_file: str = "text_features.npy"
    visual_features_file: str = "visual_features.npy"
    num_workers: int = 4


@dataclass
class Config:
    """总配置类"""
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    mmoe: MMOEConfig = field(default_factory=MMOEConfig)
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 实验相关
    experiment_name: str = "rq_vae_recommender"
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'codebook' in config_dict:
            config.codebook = CodebookConfig(**config_dict['codebook'])
        if 'mmoe' in config_dict:
            config.mmoe = MMOEConfig(**config_dict['mmoe'])
        if 'recommender' in config_dict:
            config.recommender = RecommenderConfig(**config_dict['recommender'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        for key in ['experiment_name', 'seed', 'device']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            'codebook': self.codebook.__dict__,
            'mmoe': self.mmoe.__dict__,
            'recommender': self.recommender.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


# 默认配置实例
default_config = Config()
