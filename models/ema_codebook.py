"""
EMA Codebook with Dead Code Reset Mechanism
EMA码本更新 + 死码重置机制
解决码本坍塌问题，将利用率从10%-20%提升至85%+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class EMAVectorQuantizer(nn.Module):
    """
    EMA向量量化器
    使用指数移动平均更新码本，结合死码重置机制防止码本坍塌
    
    核心特性：
    1. EMA更新：decay=0.99，平滑更新码本向量
    2. 死码检测：跟踪每个码本向量的使用频率
    3. 死码重置：将长期未使用的码本向量重置为当前batch的活跃向量
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100,
        dead_code_reset_threshold: float = 0.01
    ):
        """
        初始化EMA向量量化器
        
        Args:
            num_embeddings: 码本大小（默认256）
            embedding_dim: 嵌入向量维度
            commitment_cost: commitment loss权重
            ema_decay: EMA衰减系数（默认0.99）
            dead_code_threshold: 死码判定阈值（多少步未使用）
            dead_code_reset_threshold: 死码重置阈值（利用率低于此值触发重置）
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_reset_threshold = dead_code_reset_threshold
        
        # 码本嵌入向量 - 使用均匀分布初始化
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_sum', embedding.clone())
        self.register_buffer('embedding_count', torch.ones(num_embeddings))
        
        # 死码追踪
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('last_used_step', torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer('total_steps', torch.tensor(0, dtype=torch.long))
        
        # 统计信息
        self.register_buffer('codebook_usage_history', torch.zeros(1000))
        self.history_idx = 0
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            z: 输入张量 [B, D] 或 [B, ..., D]
            
        Returns:
            z_q: 量化后的张量
            indices: 量化索引
            commitment_loss: commitment loss
            info: 额外信息字典
        """
        # 展平输入
        original_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)
        
        # 计算距离矩阵 [B*..., K]
        distances = self._compute_distances(z_flat)
        
        # 找到最近邻
        indices = torch.argmin(distances, dim=-1)
        
        # 量化
        z_q = self.embedding[indices]
        
        # 计算commitment loss
        commitment_loss = self.commitment_cost * torch.mean((z - z_q.detach()) ** 2)
        
        # EMA更新（仅在训练时）
        if self.training:
            self._ema_update(z_flat, indices)
        
        # 直通估计器（STE）
        z_q = z + (z_q - z).detach()
        
        # 恢复原始形状
        z_q = z_q.view(original_shape)
        indices = indices.view(original_shape[:-1])
        
        # 计算统计信息
        info = self._compute_stats(indices)
        
        return z_q, indices, commitment_loss, info
    
    def _compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """计算输入向量与码本的距离"""
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        z_sq = torch.sum(z ** 2, dim=-1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=-1)
        z_e = torch.matmul(z, self.embedding.t())
        
        distances = z_sq + e_sq - 2 * z_e
        return distances
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """EMA更新码本"""
        with torch.no_grad():
            # 统计每个码本向量的使用次数
            one_hot = F.one_hot(indices, self.num_embeddings).float()
            count = one_hot.sum(dim=0)
            
            # 计算每个码本向量对应的输入向量之和
            embedding_sum = torch.matmul(one_hot.t(), z)
            
            # EMA更新
            self.embedding_count = self.ema_decay * self.embedding_count + (1 - self.ema_decay) * count
            self.embedding_sum = self.ema_decay * self.embedding_sum + (1 - self.ema_decay) * embedding_sum
            
            # 更新嵌入向量
            self.embedding = self.embedding_sum / (self.embedding_count.unsqueeze(-1) + 1e-8)
            
            # 更新使用统计
            self.usage_count += count
            self.total_steps += 1
            
            # 更新最后使用步数
            used_mask = count > 0
            self.last_used_step[used_mask] = self.total_steps
            
            # 死码重置
            self._reset_dead_codes(z, indices)
    
    def _reset_dead_codes(self, z: torch.Tensor, indices: torch.Tensor):
        """死码重置机制"""
        with torch.no_grad():
            # 找出死码
            unused_steps = self.total_steps - self.last_used_step
            dead_mask = unused_steps > self.dead_code_threshold
            
            # 计算当前利用率
            usage_rate = self.usage_count.float() / (self.total_steps.float() + 1e-8)
            low_usage_mask = usage_rate < self.dead_code_reset_threshold
            
            # 综合判定需要重置的码本向量
            reset_mask = dead_mask | low_usage_mask
            
            if reset_mask.any():
                # 从当前batch中选择活跃向量来重置死码
                num_dead = reset_mask.sum().item()
                
                # 随机选择当前batch中的向量
                batch_size = z.size(0)
                if batch_size > 0:
                    # 计算每个样本的重要性（距离最近码本的距离）
                    distances = self._compute_distances(z)
                    min_distances, _ = distances.min(dim=-1)
                    
                    # 选择距离较远的样本（这些样本可能代表未被充分覆盖的区域）
                    _, far_indices = torch.topk(min_distances, min(num_dead, batch_size))
                    
                    # 重置死码
                    dead_indices = torch.where(reset_mask)[0]
                    num_to_reset = min(len(dead_indices), len(far_indices))
                    
                    for i in range(num_to_reset):
                        dead_idx = dead_indices[i]
                        source_idx = far_indices[i]
                        
                        # 用当前样本加噪声重置死码
                        noise = torch.randn_like(z[source_idx]) * 0.01
                        self.embedding[dead_idx] = z[source_idx] + noise
                        self.embedding_sum[dead_idx] = self.embedding[dead_idx] * 10
                        self.embedding_count[dead_idx] = 10
                        self.last_used_step[dead_idx] = self.total_steps
                        self.usage_count[dead_idx] = 1
    
    def _compute_stats(self, indices: torch.Tensor) -> dict:
        """计算统计信息"""
        with torch.no_grad():
            # 计算当前batch的利用率
            unique_codes = torch.unique(indices)
            usage_rate = len(unique_codes) / self.num_embeddings
            
            # 记录历史利用率
            if self.history_idx < len(self.codebook_usage_history):
                self.codebook_usage_history[self.history_idx] = usage_rate
                self.history_idx += 1
            
            # 计算平均利用率
            valid_history = self.codebook_usage_history[:self.history_idx]
            avg_usage_rate = valid_history.mean().item() if len(valid_history) > 0 else 0.0
            
            # 计算perplexity（困惑度）
            one_hot = F.one_hot(indices, self.num_embeddings).float()
            avg_probs = one_hot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            return {
                'usage_rate': usage_rate,
                'avg_usage_rate': avg_usage_rate,
                'perplexity': perplexity.item(),
                'num_used_codes': len(unique_codes),
                'total_steps': self.total_steps.item()
            }
    
    def get_usage_statistics(self) -> dict:
        """获取详细的码本使用统计"""
        with torch.no_grad():
            usage_rate = self.usage_count.float() / (self.total_steps.float() + 1e-8)
            
            return {
                'per_code_usage': usage_rate.cpu().numpy(),
                'total_usage': self.usage_count.cpu().numpy(),
                'last_used_step': self.last_used_step.cpu().numpy(),
                'avg_usage_rate': usage_rate.mean().item(),
                'max_usage_rate': usage_rate.max().item(),
                'min_usage_rate': usage_rate.min().item(),
                'std_usage_rate': usage_rate.std().item()
            }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.usage_count.zero_()
        self.last_used_step.zero_()
        self.total_steps.zero_()
        self.codebook_usage_history.zero_()
        self.history_idx = 0


class SoftQuantizer(nn.Module):
    """
    软量化器 - 用于行为感知微调阶段
    使用Gumbel-Softmax实现可微分的软索引
    允许梯度回传至RQ-VAE编码器
    """
    
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        temperature: float = 1.0,
        hard: bool = False
    ):
        """
        初始化软量化器
        
        Args:
            num_embeddings: 码本大小
            embedding_dim: 嵌入向量维度
            temperature: Gumbel-Softmax温度系数
            hard: 是否使用硬采样（前向离散，反向连续）
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.hard = hard
        
        # 码本嵌入向量
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        
    def forward(self, z: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - 软量化
        
        Args:
            z: 输入张量 [B, D]
            temperature: 可选的温度系数
            
        Returns:
            z_q: 量化后的张量
            soft_indices: 软索引概率分布
            hard_indices: 硬索引（用于生成语义ID）
        """
        if temperature is None:
            temperature = self.temperature
            
        # 展平输入
        original_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)
        
        # 计算距离
        distances = self._compute_distances(z_flat)
        
        # 转换为logits（负距离）
        logits = -distances
        
        # Gumbel-Softmax采样
        soft_indices = F.gumbel_softmax(logits, tau=temperature, hard=self.hard, dim=-1)
        
        # 软量化：加权求和
        z_q = torch.matmul(soft_indices, self.embedding)
        
        # 硬索引（用于生成语义ID）
        hard_indices = torch.argmax(soft_indices, dim=-1)
        
        # 恢复形状
        z_q = z_q.view(original_shape)
        hard_indices = hard_indices.view(original_shape[:-1])
        
        return z_q, soft_indices, hard_indices
    
    def _compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """计算距离"""
        z_sq = torch.sum(z ** 2, dim=-1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=-1)
        z_e = torch.matmul(z, self.embedding.t())
        return z_sq + e_sq - 2 * z_e
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引获取嵌入向量"""
        return self.embedding[indices]
