"""
User Sequence Modeling Module
用户序列建模模块
完整的用户行为序列编码和建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math


class PositionalEncoding(nn.Module):
    """
    位置编码
    使用正弦函数生成位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, S_q, D]
            key: [B, S_k, D]
            value: [B, S_k, D]
            mask: [B, S_k] or [B, S_q, S_k]
        Returns:
            output: [B, S_q, D]
            attention_weights: [B, num_heads, S_q, S_k]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer块
    包含自注意力和前馈网络
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
            mask: [B, S]
        Returns:
            output: [B, S, D]
            attention_weights: [B, num_heads, S, S]
        """
        # Self-attention with residual
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attention_weights


class UserSequenceEncoder(nn.Module):
    """
    用户序列编码器
    编码用户历史行为序列
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        item_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        编码用户序列
        
        Args:
            item_embeddings: 物品嵌入序列 [B, S, D]
            mask: 有效位置掩码 [B, S], 1表示有效，0表示padding
            return_all_outputs: 是否返回所有层的输出
            
        Returns:
            sequence_output: 序列表示 [B, D]
            info: 额外信息
        """
        # 添加位置编码
        x = self.positional_encoding(item_embeddings)
        
        # 通过Transformer块
        all_outputs = []
        all_attention = []
        
        for transformer_block in self.transformer_blocks:
            x, attention = transformer_block(x, mask)
            all_outputs.append(x)
            all_attention.append(attention)
        
        # 输出归一化
        x = self.output_norm(x)
        
        # 获取序列表示（最后一个有效位置）
        if mask is not None:
            # 找到每个序列最后一个有效位置
            seq_lengths = mask.sum(dim=1).long() - 1
            batch_size = x.size(0)
            sequence_output = x[torch.arange(batch_size), seq_lengths]
        else:
            sequence_output = x[:, -1, :]
        
        info = {
            'all_outputs': all_outputs if return_all_outputs else None,
            'attention_weights': all_attention
        }
        
        return sequence_output, info


class UserBehaviorModel(nn.Module):
    """
    用户行为模型
    完整的用户行为建模，包括：
    - 用户ID嵌入
    - 历史序列编码
    - 用户兴趣提取
    """
    
    def __init__(
        self,
        num_users: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(num_users, d_model)
        
        # 序列编码器
        self.sequence_encoder = UserSequenceEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # 用户兴趣融合层
        self.interest_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        用户行为建模
        
        Args:
            user_ids: 用户ID [B]
            history_embeddings: 历史物品嵌入 [B, S, D]
            history_mask: 历史序列掩码 [B, S]
            
        Returns:
            user_representation: 用户表示 [B, D]
            info: 额外信息
        """
        # 用户ID嵌入
        user_embed = self.user_embedding(user_ids)  # [B, D]
        
        # 序列编码
        sequence_output, seq_info = self.sequence_encoder(history_embeddings, history_mask)
        
        # 融合用户ID和历史序列
        user_representation = self.interest_fusion(
            torch.cat([user_embed, sequence_output], dim=-1)
        )
        
        info = {
            'user_embed': user_embed,
            'sequence_output': sequence_output,
            'attention_weights': seq_info.get('attention_weights')
        }
        
        return user_representation, info


class SequentialRecommender(nn.Module):
    """
    序列推荐模型
    完整的序列推荐实现
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_quantization_layers: int = 3,
        codebook_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_quantization_layers = num_quantization_layers
        self.codebook_size = codebook_size
        self.d_model = d_model
        
        # 语义ID嵌入层
        self.semantic_embedding_layers = nn.ModuleList([
            nn.Embedding(codebook_size, d_model)
            for _ in range(num_quantization_layers)
        ])
        
        # 语义ID融合
        self.semantic_fusion = nn.Sequential(
            nn.Linear(d_model * num_quantization_layers, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 用户行为模型
        self.user_model = UserBehaviorModel(
            num_users=num_users,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def encode_semantic_ids(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        编码语义ID为嵌入向量
        
        Args:
            semantic_ids: [B, L] 或 [B, S, L]
            
        Returns:
            embeddings: [B, D] 或 [B, S, D]
        """
        original_shape = semantic_ids.shape
        if len(original_shape) == 2:
            semantic_ids = semantic_ids.unsqueeze(1)  # [B, 1, L]
        
        batch_size, seq_len, num_layers = semantic_ids.shape
        
        # 获取每层的嵌入
        layer_embeds = []
        for i in range(num_layers):
            layer_embed = self.semantic_embedding_layers[i](semantic_ids[:, :, i])
            layer_embeds.append(layer_embed)
        
        # 拼接并融合
        concat_embeds = torch.cat(layer_embeds, dim=-1)  # [B, S, D*L]
        embeddings = self.semantic_fusion(concat_embeds)  # [B, S, D]
        
        if len(original_shape) == 2:
            embeddings = embeddings.squeeze(1)  # [B, D]
        
        return embeddings
    
    def forward(
        self,
        user_ids: torch.Tensor,
        history_semantic_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_semantic_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            user_ids: 用户ID [B]
            history_semantic_ids: 历史语义ID [B, S, L]
            history_mask: 历史掩码 [B, S]
            candidate_semantic_ids: 候选语义ID [B, K, L]
            
        Returns:
            scores: 预测分数 [B, K]
            info: 额外信息
        """
        batch_size = user_ids.size(0)
        num_candidates = candidate_semantic_ids.size(1)
        
        # 编码历史序列
        history_embeddings = self.encode_semantic_ids(history_semantic_ids)  # [B, S, D]
        
        # 用户建模
        user_representation, user_info = self.user_model(
            user_ids, history_embeddings, history_mask
        )  # [B, D]
        
        # 编码候选物品
        candidate_embeddings = self.encode_semantic_ids(candidate_semantic_ids)  # [B, K, D]
        
        # 计算分数
        # 方法1: 内积
        inner_product = torch.bmm(
            user_representation.unsqueeze(1),
            candidate_embeddings.transpose(1, 2)
        ).squeeze(1)  # [B, K]
        
        # 方法2: MLP
        user_expanded = user_representation.unsqueeze(1).expand(-1, num_candidates, -1)
        mlp_input = torch.cat([user_expanded, candidate_embeddings], dim=-1)
        mlp_scores = self.predictor(mlp_input).squeeze(-1)  # [B, K]
        
        # 综合分数
        scores = (inner_product + mlp_scores) / self.temperature
        
        info = {
            'user_representation': user_representation,
            'candidate_embeddings': candidate_embeddings,
            'attention_weights': user_info.get('attention_weights')
        }
        
        return scores, info
    
    def compute_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算BPR损失
        
        Args:
            pos_scores: 正样本分数 [B]
            neg_scores: 负样本分数 [B, num_neg]
            
        Returns:
            loss: 损失值
            info: 损失信息
        """
        # BPR损失
        pos_scores = pos_scores.unsqueeze(1)  # [B, 1]
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        info = {
            'pos_score_mean': pos_scores.mean().item(),
            'neg_score_mean': neg_scores.mean().item(),
            'bpr_loss': bpr_loss.item()
        }
        
        return bpr_loss, info


# 导出
__all__ = [
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock',
    'UserSequenceEncoder',
    'UserBehaviorModel',
    'SequentialRecommender'
]
