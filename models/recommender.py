"""
Semantic ID-based Recommender Model
基于语义ID的推荐模型
使用RQ-VAE生成的语义ID序列进行推荐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math

from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder


class SemanticIDEmbedding(nn.Module):
    """
    语义ID嵌入层
    将多级语义ID序列转换为嵌入向量
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 128
    ):
        """
        初始化语义ID嵌入层
        
        Args:
            num_layers: 残差量化层数
            codebook_size: 码本大小
            embedding_dim: 嵌入维度
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # 每层的嵌入表
        self.layer_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim) for _ in range(num_layers)
        ])
        
        # 层权重（可学习）
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            semantic_ids: 语义ID序列 [B, L]
            
        Returns:
            embedding: 嵌入向量 [B, D]
        """
        batch_size, num_layers = semantic_ids.shape
        
        # 获取每层的嵌入
        layer_embeds = []
        for i in range(num_layers):
            layer_embed = self.layer_embeddings[i](semantic_ids[:, i])  # [B, D]
            layer_embeds.append(layer_embed)
        
        # 加权求和
        weights = F.softmax(self.layer_weights, dim=0)
        embedding = sum(w * e for w, e in zip(weights, layer_embeds))
        
        return embedding


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer序列编码器
    编码用户历史行为序列
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_length: int = 50
    ):
        """
        初始化Transformer序列编码器
        
        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
            max_seq_length: 最大序列长度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        item_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            item_embeddings: 物品嵌入序列 [B, S, D]
            mask: 注意力掩码 [B, S]
            
        Returns:
            sequence_output: 序列表示 [B, D]
        """
        batch_size, seq_length, _ = item_embeddings.shape
        
        # 位置编码
        positions = torch.arange(seq_length, device=item_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # 添加位置编码
        embeddings = item_embeddings + position_embeds
        embeddings = self.dropout(embeddings)
        
        # 创建注意力掩码
        if mask is not None:
            # 将padding位置设为True（会被忽略）
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        
        # Transformer编码
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 取最后一个有效位置的输出
        if mask is not None:
            # 找到每个序列最后一个有效位置
            last_valid_idx = mask.sum(dim=1) - 1
            sequence_output = encoded[torch.arange(batch_size), last_valid_idx]
        else:
            sequence_output = encoded[:, -1, :]
        
        return sequence_output


class SemanticRecommender(nn.Module):
    """
    基于语义ID的推荐模型
    
    核心思想：
    1. 使用RQ-VAE将物品多模态特征编码为语义ID
    2. 语义ID嵌入替代传统ID嵌入，解决冷启动问题
    3. 用户历史行为序列通过Transformer编码
    4. 预测用户对候选物品的兴趣分数
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_quantization_layers: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2,
        temperature: float = 0.1
    ):
        """
        初始化推荐模型
        
        Args:
            num_users: 用户数量
            num_items: 物品数量
            num_quantization_layers: 残差量化层数
            codebook_size: 码本大小
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            num_transformer_layers: Transformer层数
            max_seq_length: 最大序列长度
            dropout: Dropout比例
            temperature: Softmax温度系数
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_quantization_layers = num_quantization_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # 语义ID嵌入（替代传统物品ID嵌入）
        self.semantic_id_embedding = SemanticIDEmbedding(
            num_layers=num_quantization_layers,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
        
        # 传统物品ID嵌入（用于对比实验）
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 序列编码器
        self.sequence_encoder = TransformerSequenceEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        # 温度参数（可学习）
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        
    def forward(
        self,
        user_ids: torch.Tensor,
        history_semantic_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_semantic_ids: torch.Tensor,
        use_semantic: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            user_ids: 用户ID [B]
            history_semantic_ids: 历史行为语义ID序列 [B, S, L]
            history_mask: 历史行为掩码 [B, S]
            candidate_semantic_ids: 候选物品语义ID [B, K, L]
            use_semantic: 是否使用语义ID嵌入
            
        Returns:
            scores: 预测分数 [B, K]
            info: 额外信息
        """
        batch_size = user_ids.size(0)
        num_candidates = candidate_semantic_ids.size(1)
        
        # 用户嵌入
        user_embed = self.user_embedding(user_ids)  # [B, D]
        
        # 历史序列嵌入
        history_embeds = self.semantic_id_embedding(
            history_semantic_ids.view(-1, self.num_quantization_layers)
        ).view(batch_size, -1, self.embedding_dim)  # [B, S, D]
        
        # 序列编码
        sequence_output = self.sequence_encoder(history_embeds, history_mask)  # [B, D]
        
        # 用户表示 = 用户嵌入 + 序列表示
        user_representation = user_embed + sequence_output  # [B, D]
        
        # 候选物品嵌入
        candidate_embeds = self.semantic_id_embedding(
            candidate_semantic_ids.view(-1, self.num_quantization_layers)
        ).view(batch_size, num_candidates, self.embedding_dim)  # [B, K, D]
        
        # 计算分数
        # 方法1：内积
        inner_product = torch.bmm(
            user_representation.unsqueeze(1),  # [B, 1, D]
            candidate_embeds.transpose(1, 2)    # [B, D, K]
        ).squeeze(1)  # [B, K]
        
        # 方法2：拼接后MLP
        user_expanded = user_representation.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, K, D]
        mlp_input = torch.cat([user_expanded, candidate_embeds], dim=-1)  # [B, K, 2D]
        mlp_scores = self.predictor(mlp_input).squeeze(-1)  # [B, K]
        
        # 综合分数
        scores = inner_product + mlp_scores
        
        # 应用温度缩放
        scores = scores / self.temperature_param
        
        info = {
            'user_embed_norm': torch.norm(user_embed, dim=-1).mean().item(),
            'sequence_output_norm': torch.norm(sequence_output, dim=-1).mean().item(),
            'candidate_embed_norm': torch.norm(candidate_embeds, dim=-1).mean().item()
        }
        
        return scores, info
    
    def get_item_embedding_by_semantic_id(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        通过语义ID获取物品嵌入
        
        Args:
            semantic_ids: 语义ID序列 [B, L]
            
        Returns:
            embedding: 物品嵌入 [B, D]
        """
        return self.semantic_id_embedding(semantic_ids)
    
    def compute_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        num_negatives: int = 4
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            scores: 预测分数 [B, K]
            labels: 标签 [B, K]（正样本为1，负样本为0）
            num_negatives: 负样本数量
            
        Returns:
            loss: 损失值
        """
        # 使用Binary Cross Entropy Loss
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
        return loss


class ColdStartHandler(nn.Module):
    """
    冷启动处理器
    处理新物品的推荐
    """
    
    def __init__(
        self,
        rq_vae: RQVAE,
        mmoe_encoder: MultiModalFeatureEncoder,
        recommender: SemanticRecommender
    ):
        """
        初始化冷启动处理器
        
        Args:
            rq_vae: RQ-VAE模型
            mmoe_encoder: MMOE编码器
            recommender: 推荐模型
        """
        super().__init__()
        
        self.rq_vae = rq_vae
        self.mmoe_encoder = mmoe_encoder
        self.recommender = recommender
        
    @torch.no_grad()
    def encode_new_item(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码新物品
        
        Args:
            text_features: 文本特征 [B, text_dim]
            visual_features: 视觉特征 [B, visual_dim]
            
        Returns:
            semantic_ids: 语义ID序列 [B, L]
            item_embedding: 物品嵌入 [B, D]
        """
        # 多模态编码
        fused_features, _, _, _ = self.mmoe_encoder(text_features, visual_features)
        
        # RQ-VAE编码
        semantic_ids = self.rq_vae.encode_to_ids(fused_features)
        
        # 获取物品嵌入
        item_embedding = self.recommender.get_item_embedding_by_semantic_id(semantic_ids)
        
        return semantic_ids, item_embedding
    
    @torch.no_grad()
    def find_similar_items(
        self,
        query_semantic_ids: torch.Tensor,
        item_semantic_ids: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查找相似物品
        
        Args:
            query_semantic_ids: 查询语义ID [B, L]
            item_semantic_ids: 物品语义ID库 [N, L]
            top_k: 返回top-k相似物品
            
        Returns:
            indices: 相似物品索引 [B, K]
            similarities: 相似度分数 [B, K]
        """
        # 获取嵌入
        query_embed = self.recommender.get_item_embedding_by_semantic_id(query_semantic_ids)
        item_embeds = self.recommender.get_item_embedding_by_semantic_id(item_semantic_ids)
        
        # 计算相似度
        query_embed = F.normalize(query_embed, dim=-1)
        item_embeds = F.normalize(item_embeds, dim=-1)
        
        similarities = torch.matmul(query_embed, item_embeds.t())  # [B, N]
        
        # 获取top-k
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=-1)
        
        return top_indices, top_similarities


class RecommendationLoss(nn.Module):
    """
    推荐损失函数
    整合多种损失
    """
    
    def __init__(
        self,
        recommendation_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.1
    ):
        """
        初始化推荐损失
        
        Args:
            recommendation_weight: 推荐损失权重
            reconstruction_weight: 重建损失权重
            contrastive_weight: 对比损失权重
        """
        super().__init__()
        
        self.recommendation_weight = recommendation_weight
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        reconstruction_loss: Optional[torch.Tensor] = None,
        contrastive_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失
        
        Args:
            scores: 预测分数
            labels: 标签
            reconstruction_loss: 重建损失
            contrastive_loss: 对比损失
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        # 推荐损失
        rec_loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        
        total_loss = self.recommendation_weight * rec_loss
        loss_dict = {'recommendation_loss': rec_loss.item()}
        
        # 重建损失
        if reconstruction_loss is not None:
            total_loss = total_loss + self.reconstruction_weight * reconstruction_loss
            loss_dict['reconstruction_loss'] = reconstruction_loss.item()
        
        # 对比损失
        if contrastive_loss is not None:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
