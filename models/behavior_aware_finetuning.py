"""
Behavior-Aware Fine-tuning Module
行为感知微调模块
通过软索引+STE允许推荐任务梯度回传至RQ-VAE
联合优化重建损失与推荐损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math


class StraightThroughEstimator(torch.autograd.Function):
    """
    直通估计器（Straight-Through Estimator）
    前向传播使用离散值，反向传播使用连续梯度
    
    解决量化操作不可微的问题，允许梯度回传
    """
    
    @staticmethod
    def forward(ctx, z, z_q):
        """
        前向传播：返回量化后的值
        
        Args:
            z: 原始连续值
            z_q: 量化后的值
            
        Returns:
            z_q: 量化后的值（直通）
        """
        return z_q
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：梯度直通
        
        Args:
            grad_output: 上游梯度
            
        Returns:
            grad_output: 梯度直接传递给原始连续值
            None: 量化值不需要梯度
        """
        return grad_output, None


class SoftIndexQuantizer(nn.Module):
    """
    软索引量化器
    使用Gumbel-Softmax实现可微分的软索引
    结合STE实现梯度回传
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.99
    ):
        """
        初始化软索引量化器
        
        Args:
            num_embeddings: 码本大小
            embedding_dim: 嵌入维度
            temperature: 初始温度
            min_temperature: 最小温度
            temperature_decay: 温度衰减系数
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        
        # 码本嵌入（可学习）
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.normal_(self.embedding, std=0.02)
        
        # 当前温度
        self.current_temperature = temperature
        
    def forward(
        self,
        z: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z: 输入特征 [B, D]
            hard: 是否使用硬采样
            
        Returns:
            z_q: 量化后的特征 [B, D]
            soft_indices: 软索引概率分布 [B, K]
            hard_indices: 硬索引 [B]
        """
        # 计算距离
        distances = self._compute_distances(z)  # [B, K]
        
        # 转换为logits
        logits = -distances
        
        # Gumbel-Softmax
        soft_indices = F.gumbel_softmax(
            logits,
            tau=self.current_temperature,
            hard=hard,
            dim=-1
        )  # [B, K]
        
        # 软量化：加权求和
        z_q_soft = torch.matmul(soft_indices, self.embedding)  # [B, D]
        
        # 硬索引
        hard_indices = torch.argmax(soft_indices, dim=-1)  # [B]
        
        # 硬量化
        z_q_hard = self.embedding[hard_indices]  # [B, D]
        
        # 使用STE：前向用硬量化，反向用软量化
        z_q = StraightThroughEstimator.apply(z_q_soft, z_q_hard)
        
        return z_q, soft_indices, hard_indices
    
    def _compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """计算距离矩阵"""
        z_sq = torch.sum(z ** 2, dim=-1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=-1)
        z_e = torch.matmul(z, self.embedding.t())
        return z_sq + e_sq - 2 * z_e
    
    def update_temperature(self):
        """更新温度（退火）"""
        self.current_temperature = max(
            self.min_temperature,
            self.current_temperature * self.temperature_decay
        )
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引获取嵌入"""
        return self.embedding[indices]


class BehaviorAwareRQVAE(nn.Module):
    """
    行为感知的RQ-VAE
    支持梯度回传的残差量化VAE
    
    核心特性：
    1. 软索引：使用Gumbel-Softmax实现可微分量化
    2. STE：直通估计器允许梯度回传
    3. 联合优化：同时优化重建损失和推荐损失
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_quantization_layers: int = 3,
        codebook_size: int = 256,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.99
    ):
        """
        初始化行为感知RQ-VAE
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            embedding_dim: 嵌入维度
            num_quantization_layers: 量化层数
            codebook_size: 码本大小
            temperature: 初始温度
            min_temperature: 最小温度
            temperature_decay: 温度衰减系数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_quantization_layers = num_quantization_layers
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 软索引量化器（每层一个）
        self.quantizers = nn.ModuleList([
            SoftIndexQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=embedding_dim,
                temperature=temperature,
                min_temperature=min_temperature,
                temperature_decay=temperature_decay
            ) for _ in range(num_quantization_layers)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            hard: 是否使用硬采样
            
        Returns:
            x_recon: 重建特征 [B, input_dim]
            semantic_ids: 语义ID序列 [B, L]
            z_q: 量化后的特征 [B, embedding_dim]
            info: 额外信息
        """
        batch_size = x.size(0)
        
        # 编码
        z = self.encoder(x)  # [B, embedding_dim]
        
        # 残差量化
        residual = z
        z_q = torch.zeros_like(z)
        semantic_ids_list = []
        soft_indices_list = []
        
        for quantizer in self.quantizers:
            z_q_layer, soft_indices, hard_indices = quantizer(residual, hard=hard)
            z_q = z_q + z_q_layer
            semantic_ids_list.append(hard_indices)
            soft_indices_list.append(soft_indices)
            residual = residual - z_q_layer
        
        # 解码
        x_recon = self.decoder(z_q)
        
        # 语义ID序列
        semantic_ids = torch.stack(semantic_ids_list, dim=-1)  # [B, L]
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(x_recon, x)
        
        info = {
            'reconstruction_loss': reconstruction_loss.item(),
            'soft_indices': soft_indices_list,
            'z_norm': torch.norm(z, dim=-1).mean().item(),
            'z_q_norm': torch.norm(z_q, dim=-1).mean().item()
        }
        
        return x_recon, semantic_ids, z_q, info
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码为语义ID"""
        with torch.no_grad():
            z = self.encoder(x)
            residual = z
            semantic_ids_list = []
            
            for quantizer in self.quantizers:
                _, _, hard_indices = quantizer(residual, hard=True)
                semantic_ids_list.append(hard_indices)
                z_q_layer = quantizer.get_embedding(hard_indices)
                residual = residual - z_q_layer
            
            semantic_ids = torch.stack(semantic_ids_list, dim=-1)
        
        return semantic_ids
    
    def decode(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """从语义ID解码"""
        with torch.no_grad():
            z_q = torch.zeros(semantic_ids.size(0), self.embedding_dim, 
                             device=semantic_ids.device)
            
            for i, quantizer in enumerate(self.quantizers):
                z_q = z_q + quantizer.get_embedding(semantic_ids[:, i])
            
            x_recon = self.decoder(z_q)
        
        return x_recon
    
    def update_temperature(self):
        """更新所有量化器的温度"""
        for quantizer in self.quantizers:
            quantizer.update_temperature()


class JointLoss(nn.Module):
    """
    联合损失函数
    整合重建损失和推荐损失
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        recommendation_weight: float = 1.0,
        commitment_weight: float = 0.25,
        diversity_weight: float = 0.1
    ):
        """
        初始化联合损失
        
        Args:
            reconstruction_weight: 重建损失权重
            recommendation_weight: 推荐损失权重
            commitment_weight: commitment损失权重
            diversity_weight: 多样性损失权重
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.recommendation_weight = recommendation_weight
        self.commitment_weight = commitment_weight
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        z_q: torch.Tensor,
        recommendation_loss: Optional[torch.Tensor] = None,
        soft_indices: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算联合损失
        
        Args:
            x: 原始输入
            x_recon: 重建输出
            z: 编码器输出
            z_q: 量化输出
            recommendation_loss: 推荐损失
            soft_indices: 软索引列表
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        loss_dict = {}
        
        # 重建损失
        reconstruction_loss = F.mse_loss(x_recon, x)
        loss_dict['reconstruction_loss'] = reconstruction_loss.item()
        
        # Commitment损失（鼓励编码器输出接近量化值）
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss_dict['commitment_loss'] = commitment_loss.item()
        
        # 总损失
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.commitment_weight * commitment_loss
        )
        
        # 推荐损失
        if recommendation_loss is not None:
            total_loss = total_loss + self.recommendation_weight * recommendation_loss
            loss_dict['recommendation_loss'] = recommendation_loss.item()
        
        # 多样性损失（鼓励码本使用均匀）
        if soft_indices is not None and self.diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss(soft_indices)
            total_loss = total_loss + self.diversity_weight * diversity_loss
            loss_dict['diversity_loss'] = diversity_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_diversity_loss(self, soft_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        计算多样性损失
        鼓励码本使用更加均匀
        
        Args:
            soft_indices: 软索引列表
            
        Returns:
            diversity_loss: 多样性损失
        """
        total_entropy = 0.0
        
        for indices in soft_indices:
            # 计算每个码本向量的平均使用概率
            avg_probs = indices.mean(dim=0)  # [K]
            
            # 计算熵（鼓励高熵，即均匀分布）
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            
            # 归一化到[0, 1]
            max_entropy = math.log(avg_probs.size(0))
            normalized_entropy = entropy / max_entropy
            
            total_entropy = total_entropy + normalized_entropy
        
        # 返回负熵（最小化负熵 = 最大化熵）
        return -total_entropy / len(soft_indices)


class BehaviorAwareFineTuner(nn.Module):
    """
    行为感知微调器
    整合RQ-VAE和推荐模型的联合训练
    """
    
    def __init__(
        self,
        rq_vae: BehaviorAwareRQVAE,
        recommender: nn.Module,
        reconstruction_weight: float = 1.0,
        recommendation_weight: float = 1.0,
        freeze_encoder_steps: int = 0
    ):
        """
        初始化行为感知微调器
        
        Args:
            rq_vae: 行为感知RQ-VAE
            recommender: 推荐模型
            reconstruction_weight: 重建损失权重
            recommendation_weight: 推荐损失权重
            freeze_encoder_steps: 冻结编码器的步数
        """
        super().__init__()
        
        self.rq_vae = rq_vae
        self.recommender = recommender
        self.reconstruction_weight = reconstruction_weight
        self.recommendation_weight = recommendation_weight
        self.freeze_encoder_steps = freeze_encoder_steps
        
        self.joint_loss = JointLoss(
            reconstruction_weight=reconstruction_weight,
            recommendation_weight=recommendation_weight
        )
        
        self.global_step = 0
        
    def forward(
        self,
        item_features: torch.Tensor,
        user_ids: torch.Tensor,
        history_semantic_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_semantic_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            item_features: 物品多模态特征 [B, input_dim]
            user_ids: 用户ID [B]
            history_semantic_ids: 历史语义ID [B, S, L]
            history_mask: 历史掩码 [B, S]
            candidate_semantic_ids: 候选语义ID [B, K, L]
            labels: 标签 [B, K]
            
        Returns:
            total_loss: 总损失
            info: 额外信息
        """
        # 判断是否冻结编码器
        if self.freeze_encoder_steps > 0 and self.global_step < self.freeze_encoder_steps:
            # 冻结RQ-VAE
            with torch.no_grad():
                x_recon, semantic_ids, z_q, rq_info = self.rq_vae(item_features, hard=True)
                z = self.rq_vae.encoder(item_features)
        else:
            # 正常训练RQ-VAE
            x_recon, semantic_ids, z_q, rq_info = self.rq_vae(item_features, hard=False)
            z = self.rq_vae.encoder(item_features)
        
        # 推荐模型前向
        scores, rec_info = self.recommender(
            user_ids,
            history_semantic_ids,
            history_mask,
            candidate_semantic_ids
        )
        
        # 推荐损失
        recommendation_loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
        # 联合损失
        total_loss, loss_dict = self.joint_loss(
            x=item_features,
            x_recon=x_recon,
            z=z,
            z_q=z_q,
            recommendation_loss=recommendation_loss,
            soft_indices=rq_info.get('soft_indices')
        )
        
        # 更新温度
        if self.training:
            self.rq_vae.update_temperature()
            self.global_step += 1
        
        # 汇总信息
        info = {
            **loss_dict,
            **rq_info,
            **rec_info,
            'semantic_ids': semantic_ids,
            'scores': scores
        }
        
        return total_loss, info
    
    def get_item_semantic_ids(self, item_features: torch.Tensor) -> torch.Tensor:
        """获取物品的语义ID"""
        return self.rq_vae.encode(item_features)
    
    def get_item_embedding(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入"""
        return self.recommender.get_item_embedding_by_semantic_id(semantic_ids)


class GradientFlowMonitor:
    """
    梯度流监控器
    监控梯度从推荐任务回传到RQ-VAE的情况
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化监控器
        
        Args:
            model: 要监控的模型
        """
        self.model = model
        self.gradient_stats = {}
        
    def register_hooks(self):
        """注册梯度钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._gradient_hook(grad, n)
                )
                
    def _gradient_hook(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """梯度钩子"""
        self.gradient_stats[name] = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.max().item(),
            'min': grad.min().item(),
            'norm': grad.norm().item()
        }
        return grad
    
    def get_stats(self) -> Dict:
        """获取梯度统计"""
        return self.gradient_stats
    
    def clear_stats(self):
        """清空统计"""
        self.gradient_stats = {}


def create_behavior_aware_model(
    input_dim: int,
    num_users: int,
    num_items: int,
    embedding_dim: int = 128,
    num_quantization_layers: int = 3,
    codebook_size: int = 256,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    max_seq_length: int = 50
) -> BehaviorAwareFineTuner:
    """
    创建行为感知模型的便捷函数
    
    Args:
        input_dim: 输入特征维度
        num_users: 用户数量
        num_items: 物品数量
        embedding_dim: 嵌入维度
        num_quantization_layers: 量化层数
        codebook_size: 码本大小
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
        num_transformer_layers: Transformer层数
        max_seq_length: 最大序列长度
        
    Returns:
        fine_tuner: 行为感知微调器
    """
    from models.recommender import SemanticRecommender
    
    # 创建RQ-VAE
    rq_vae = BehaviorAwareRQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_quantization_layers=num_quantization_layers,
        codebook_size=codebook_size
    )
    
    # 创建推荐模型
    recommender = SemanticRecommender(
        num_users=num_users,
        num_items=num_items,
        num_quantization_layers=num_quantization_layers,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        max_seq_length=max_seq_length
    )
    
    # 创建微调器
    fine_tuner = BehaviorAwareFineTuner(
        rq_vae=rq_vae,
        recommender=recommender
    )
    
    return fine_tuner
