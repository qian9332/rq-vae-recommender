"""
Residual Quantized VAE (RQ-VAE)
残差量化变分自编码器
将Item多模态特征编码为离散语义ID序列，替代传统Hash ID
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np

from models.ema_codebook import EMAVectorQuantizer, SoftQuantizer


class ResidualQuantizer(nn.Module):
    """
    残差量化器
    通过多层残差量化将连续特征编码为离散语义ID序列
    
    核心思想：
    1. 第一层量化原始特征的主要信息
    2. 后续层量化残差（未被前层捕获的信息）
    3. 最终得到多级语义ID序列 [id_1, id_2, id_3]
    
    优势：
    - 解决冷启动语义孤岛问题：新物品通过语义相似性获得相近ID
    - 层次化表示：不同层捕获不同粒度的语义信息
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100,
        dead_code_reset_threshold: float = 0.01,
        use_soft_quantization: bool = False,
        soft_temperature: float = 1.0
    ):
        """
        初始化残差量化器
        
        Args:
            num_layers: 残差量化层数（2-3层）
            codebook_size: 每层码本大小（256）
            embedding_dim: 嵌入向量维度
            commitment_cost: commitment loss权重
            ema_decay: EMA衰减系数
            dead_code_threshold: 死码判定阈值
            dead_code_reset_threshold: 死码重置阈值
            use_soft_quantization: 是否使用软量化（用于微调阶段）
            soft_temperature: 软量化温度系数
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.use_soft_quantization = use_soft_quantization
        
        # 创建多层码本
        if use_soft_quantization:
            self.codebooks = nn.ModuleList([
                SoftQuantizer(
                    num_embeddings=codebook_size,
                    embedding_dim=embedding_dim,
                    temperature=soft_temperature,
                    hard=False
                ) for _ in range(num_layers)
            ])
        else:
            self.codebooks = nn.ModuleList([
                EMAVectorQuantizer(
                    num_embeddings=codebook_size,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    ema_decay=ema_decay,
                    dead_code_threshold=dead_code_threshold,
                    dead_code_reset_threshold=dead_code_reset_threshold
                ) for _ in range(num_layers)
            ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播 - 残差量化
        
        Args:
            z: 输入特征 [B, D]
            
        Returns:
            z_q: 重建特征 [B, D]
            indices: 语义ID序列 [B, L] (L=num_layers)
            total_loss: 总损失（commitment loss之和）
            info: 统计信息字典
        """
        batch_size = z.size(0)
        device = z.device
        
        # 初始化
        residual = z
        z_q = torch.zeros_like(z)
        indices_list = []
        total_commitment_loss = 0.0
        all_info = {'layer_stats': [], 'usage_rates': []}
        
        # 逐层量化
        for layer_idx in range(self.num_layers):
            codebook = self.codebooks[layer_idx]
            
            if self.use_soft_quantization:
                # 软量化（用于微调）
                z_q_layer, soft_indices, hard_indices = codebook(residual)
                indices_list.append(hard_indices)
                z_q = z_q + z_q_layer
                # 软量化没有commitment loss
            else:
                # EMA量化
                z_q_layer, idx, commitment_loss, info = codebook(residual)
                indices_list.append(idx)
                total_commitment_loss = total_commitment_loss + commitment_loss
                z_q = z_q + z_q_layer
                
                # 记录统计信息
                all_info['layer_stats'].append(info)
                all_info['usage_rates'].append(info['usage_rate'])
            
            # 计算残差
            residual = residual - z_q_layer
        
        # 堆叠索引得到语义ID序列
        indices = torch.stack(indices_list, dim=-1)  # [B, L]
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(z_q, z.detach())
        
        all_info['reconstruction_loss'] = reconstruction_loss.item()
        all_info['total_commitment_loss'] = total_commitment_loss.item() if isinstance(total_commitment_loss, torch.Tensor) else total_commitment_loss
        
        return z_q, indices, total_commitment_loss, all_info
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅编码，返回语义ID序列
        
        Args:
            z: 输入特征 [B, D]
            
        Returns:
            indices: 语义ID序列 [B, L]
        """
        with torch.no_grad():
            _, indices, _, _ = self.forward(z)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从语义ID序列解码重建特征
        
        Args:
            indices: 语义ID序列 [B, L]
            
        Returns:
            z_q: 重建特征 [B, D]
        """
        batch_size = indices.size(0)
        z_q = torch.zeros(batch_size, self.embedding_dim, device=indices.device)
        
        for layer_idx in range(self.num_layers):
            layer_indices = indices[:, layer_idx]
            codebook = self.codebooks[layer_idx]
            
            if self.use_soft_quantization:
                z_q = z_q + codebook.get_embedding(layer_indices)
            else:
                z_q = z_q + codebook.embedding[layer_indices]
        
        return z_q
    
    def get_codebook_usage(self) -> Dict:
        """获取所有码本的使用统计"""
        usage_stats = {}
        for i, codebook in enumerate(self.codebooks):
            if not self.use_soft_quantization:
                usage_stats[f'layer_{i}'] = codebook.get_usage_statistics()
        return usage_stats


class RQVAEEncoder(nn.Module):
    """
    RQ-VAE编码器
    将多模态特征编码为适合残差量化的表示
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（与码本维度一致）
            num_layers: MLP层数
            dropout: Dropout比例
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(x)


class RQVAEDecoder(nn.Module):
    """
    RQ-VAE解码器
    从量化表示重建原始特征
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化解码器
        
        Args:
            input_dim: 输入维度（码本维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（原始特征维度）
            num_layers: MLP层数
            dropout: Dropout比例
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(z)


class RQVAE(nn.Module):
    """
    完整的残差量化变分自编码器
    整合编码器、残差量化器和解码器
    
    用于将Item多模态特征编码为离散语义ID序列
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_quantization_layers: int = 3,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100,
        dead_code_reset_threshold: float = 0.01,
        use_soft_quantization: bool = False,
        soft_temperature: float = 1.0
    ):
        """
        初始化RQ-VAE
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            embedding_dim: 嵌入/码本维度
            num_quantization_layers: 残差量化层数
            codebook_size: 码本大小
            commitment_cost: commitment loss权重
            ema_decay: EMA衰减系数
            dead_code_threshold: 死码判定阈值
            dead_code_reset_threshold: 死码重置阈值
            use_soft_quantization: 是否使用软量化
            soft_temperature: 软量化温度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_quantization_layers = num_quantization_layers
        
        # 编码器
        self.encoder = RQVAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=2
        )
        
        # 残差量化器
        self.quantizer = ResidualQuantizer(
            num_layers=num_quantization_layers,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            dead_code_reset_threshold=dead_code_reset_threshold,
            use_soft_quantization=use_soft_quantization,
            soft_temperature=soft_temperature
        )
        
        # 解码器
        self.decoder = RQVAEDecoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=2
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            
        Returns:
            x_recon: 重建特征 [B, input_dim]
            semantic_ids: 语义ID序列 [B, L]
            loss: 总损失
            info: 统计信息
        """
        # 编码
        z = self.encoder(x)
        
        # 残差量化
        z_q, semantic_ids, commitment_loss, quant_info = self.quantizer(z)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(x_recon, x)
        
        # 总损失
        total_loss = reconstruction_loss + commitment_loss
        
        # 汇总信息
        info = {
            'reconstruction_loss': reconstruction_loss.item(),
            'commitment_loss': commitment_loss.item() if isinstance(commitment_loss, torch.Tensor) else commitment_loss,
            'total_loss': total_loss.item(),
            **quant_info
        }
        
        return x_recon, semantic_ids, total_loss, info
    
    def encode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入特征编码为语义ID序列
        
        Args:
            x: 输入特征 [B, input_dim]
            
        Returns:
            semantic_ids: 语义ID序列 [B, L]
        """
        with torch.no_grad():
            z = self.encoder(x)
            semantic_ids = self.quantizer.encode(z)
        return semantic_ids
    
    def decode_from_ids(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        从语义ID序列解码重建特征
        
        Args:
            semantic_ids: 语义ID序列 [B, L]
            
        Returns:
            x_recon: 重建特征 [B, input_dim]
        """
        with torch.no_grad():
            z_q = self.quantizer.decode(semantic_ids)
            x_recon = self.decoder(z_q)
        return x_recon
    
    def get_semantic_embedding(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        获取语义ID对应的嵌入向量
        
        Args:
            semantic_ids: 语义ID序列 [B, L]
            
        Returns:
            embedding: 嵌入向量 [B, embedding_dim]
        """
        with torch.no_grad():
            z_q = self.quantizer.decode(semantic_ids)
        return z_q
    
    def get_codebook_usage(self) -> Dict:
        """获取码本使用统计"""
        return self.quantizer.get_codebook_usage()


def convert_ids_to_string(semantic_ids: torch.Tensor, codebook_size: int = 256) -> List[str]:
    """
    将语义ID序列转换为字符串表示
    
    Args:
        semantic_ids: 语义ID序列 [B, L]
        codebook_size: 码本大小
        
    Returns:
        id_strings: 字符串ID列表
    """
    batch_size, num_layers = semantic_ids.shape
    id_strings = []
    
    for i in range(batch_size):
        ids = semantic_ids[i].cpu().numpy()
        # 使用类似IP地址的格式表示层次化语义ID
        id_str = '.'.join([str(int(idx)) for idx in ids])
        id_strings.append(id_str)
    
    return id_strings


def compute_semantic_similarity(
    ids1: torch.Tensor,
    ids2: torch.Tensor,
    codebook_size: int = 256
) -> torch.Tensor:
    """
    计算两个语义ID序列之间的语义相似度
    
    基于层次化结构，不同层赋予不同权重
    
    Args:
        ids1: 语义ID序列 [B, L]
        ids2: 语义ID序列 [B, L]
        codebook_size: 码本大小
        
    Returns:
        similarity: 相似度分数 [B]
    """
    batch_size, num_layers = ids1.shape
    
    # 层权重（越高层权重越小）
    layer_weights = torch.tensor([2 ** (num_layers - i - 1) for i in range(num_layers)], 
                                  device=ids1.device, dtype=torch.float)
    layer_weights = layer_weights / layer_weights.sum()
    
    # 计算每层的匹配分数
    matches = (ids1 == ids2).float()  # [B, L]
    
    # 加权求和
    similarity = (matches * layer_weights.unsqueeze(0)).sum(dim=-1)
    
    return similarity
