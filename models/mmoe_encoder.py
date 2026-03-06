"""
MMOE Multi-Modal Encoder
多任务多专家（Multi-gate Mixture-of-Experts）多模态编码器
实现文本和视觉模态的协同与解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class ExpertNetwork(nn.Module):
    """
    专家网络
    每个专家是一个前馈网络，学习特定的特征变换
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        初始化专家网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout比例
            activation: 激活函数类型
        """
        super().__init__()
        
        # 选择激活函数
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "swish":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


class GateNetwork(nn.Module):
    """
    门控网络
    学习如何组合不同专家的输出
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        dropout: float = 0.1
    ):
        """
        初始化门控网络
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            dropout: Dropout比例
        """
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            gate_weights: 专家权重 [B, num_experts]
        """
        return self.gate(x)


class MMOEEncoder(nn.Module):
    """
    MMOE多模态编码器
    Multi-gate Mixture-of-Experts for Multi-Modal Encoding
    
    核心思想：
    1. 共享专家层：多个专家网络学习通用的特征变换
    2. 任务特定门控：每个模态有独立的门控网络，学习如何组合专家
    3. 协同与解耦：通过门控机制实现模态间的信息共享和独立建模
    
    优势：
    - 文本和视觉模态可以共享底层特征变换
    - 门控机制允许每个模态选择性地利用专家
    - 缓解多模态之间的冲突和干扰
    """
    
    def __init__(
        self,
        text_input_dim: int = 768,
        visual_input_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_experts: int = 4,
        num_tasks: int = 2,
        expert_hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        初始化MMOE编码器
        
        Args:
            text_input_dim: 文本特征维度（如BERT输出768）
            visual_input_dim: 视觉特征维度（如ResNet输出2048）
            hidden_dim: 共享隐藏层维度
            output_dim: 输出维度（与码本维度一致）
            num_experts: 专家网络数量
            num_tasks: 任务数量（文本任务、视觉任务）
            expert_hidden_dim: 专家网络隐藏层维度
            dropout: Dropout比例
            activation: 激活函数
        """
        super().__init__()
        
        self.text_input_dim = text_input_dim
        self.visual_input_dim = visual_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # 模态特定的输入投影层
        self.text_projection = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 共享专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=hidden_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_experts)
        ])
        
        # 任务特定的门控网络
        # 门控1：文本模态门控
        # 门控2：视觉模态门控
        self.gates = nn.ModuleList([
            GateNetwork(
                input_dim=hidden_dim,
                num_experts=num_experts,
                dropout=dropout
            ) for _ in range(num_tasks)
        ])
        
        # 任务塔网络（输出层）
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_tasks)
        ])
        
        # 跨模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 模态权重学习（自适应融合）
        self.modality_gate = nn.Sequential(
            nn.Linear(output_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            text_features: 文本特征 [B, text_input_dim]
            visual_features: 视觉特征 [B, visual_input_dim]
            
        Returns:
            fused_output: 融合后的多模态表示 [B, output_dim]
            text_output: 文本模态表示 [B, output_dim]
            visual_output: 视觉模态表示 [B, output_dim]
            info: 额外信息字典
        """
        batch_size = text_features.size(0)
        
        # 投影到共享空间
        text_proj = self.text_projection(text_features)  # [B, hidden_dim]
        visual_proj = self.visual_projection(visual_features)  # [B, hidden_dim]
        
        # 专家网络处理
        expert_outputs = []
        for expert in self.experts:
            # 每个专家处理两种模态的拼接或单独处理
            # 这里采用分别处理后加权的方式
            expert_out = expert(text_proj) + expert(visual_proj)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, hidden_dim]
        
        # 门控网络计算权重
        gate_weights_text = self.gates[0](text_proj)  # [B, num_experts]
        gate_weights_visual = self.gates[1](visual_proj)  # [B, num_experts]
        
        # 加权组合专家输出
        # [B, num_experts, 1] * [B, num_experts, hidden_dim] -> [B, hidden_dim]
        text_expert_out = (gate_weights_text.unsqueeze(-1) * expert_outputs).sum(dim=1)
        visual_expert_out = (gate_weights_visual.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        # 任务塔输出
        text_output = self.towers[0](text_expert_out)  # [B, output_dim]
        visual_output = self.towers[1](visual_expert_out)  # [B, output_dim]
        
        # 跨模态融合
        concat_features = torch.cat([text_output, visual_output], dim=-1)  # [B, output_dim*2]
        
        # 自适应模态权重
        modality_weights = self.modality_gate(concat_features)  # [B, 2]
        
        # 加权融合
        fused_output = self.fusion_layer(concat_features)  # [B, output_dim]
        
        # 额外信息
        info = {
            'gate_weights_text': gate_weights_text,
            'gate_weights_visual': gate_weights_visual,
            'modality_weights': modality_weights,
            'text_proj_norm': torch.norm(text_proj, dim=-1).mean().item(),
            'visual_proj_norm': torch.norm(visual_proj, dim=-1).mean().item()
        }
        
        return fused_output, text_output, visual_output, info
    
    def get_expert_importance(self) -> Dict:
        """获取各专家的重要性分数"""
        # 通过门控权重的平均值来衡量专家重要性
        importance = {}
        
        with torch.no_grad():
            for i, gate in enumerate(self.gates):
                # 使用随机输入估计平均门控权重
                dummy_input = torch.randn(100, self.hidden_dim, device=next(self.parameters()).device)
                weights = gate(dummy_input).mean(dim=0)
                importance[f'gate_{i}'] = weights.cpu().numpy()
        
        return importance


class MultiModalFeatureEncoder(nn.Module):
    """
    完整的多模态特征编码器
    整合MMOE和特征处理
    """
    
    def __init__(
        self,
        text_input_dim: int = 768,
        visual_input_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        初始化多模态特征编码器
        
        Args:
            text_input_dim: 文本特征维度
            visual_input_dim: 视觉特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_experts: 专家数量
            expert_hidden_dim: 专家隐藏层维度
            dropout: Dropout比例
            activation: 激活函数
        """
        super().__init__()
        
        self.mmoe = MMOEEncoder(
            text_input_dim=text_input_dim,
            visual_input_dim=visual_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            num_tasks=2,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
            activation=activation
        )
        
        # 特征缺失处理
        self.text_missing_embedding = nn.Parameter(torch.randn(1, text_input_dim))
        self.visual_missing_embedding = nn.Parameter(torch.randn(1, visual_input_dim))
        
    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            text_features: 文本特征 [B, text_input_dim]，可为None
            visual_features: 视觉特征 [B, visual_input_dim]，可为None
            batch_size: 批次大小（当特征缺失时需要）
            
        Returns:
            fused_output: 融合后的多模态表示
            text_output: 文本模态表示
            visual_output: 视觉模态表示
            info: 额外信息
        """
        device = next(self.parameters()).device
        
        # 处理缺失特征
        if text_features is None:
            if batch_size is None:
                raise ValueError("batch_size must be provided when text_features is None")
            text_features = self.text_missing_embedding.expand(batch_size, -1).to(device)
        
        if visual_features is None:
            if batch_size is None:
                batch_size = text_features.size(0)
            visual_features = self.visual_missing_embedding.expand(batch_size, -1).to(device)
        
        return self.mmoe(text_features, visual_features)
    
    def encode_text_only(self, text_features: torch.Tensor) -> torch.Tensor:
        """仅使用文本特征编码"""
        batch_size = text_features.size(0)
        device = text_features.device
        dummy_visual = self.visual_missing_embedding.expand(batch_size, -1).to(device)
        fused, text_out, _, _ = self.mmoe(text_features, dummy_visual)
        return text_out
    
    def encode_visual_only(self, visual_features: torch.Tensor) -> torch.Tensor:
        """仅使用视觉特征编码"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        dummy_text = self.text_missing_embedding.expand(batch_size, -1).to(device)
        fused, _, visual_out, _ = self.mmoe(dummy_text, visual_features)
        return visual_out


class ModalityAlignmentLoss(nn.Module):
    """
    模态对齐损失
    促进文本和视觉模态在共享空间中的对齐
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        初始化模态对齐损失
        
        Args:
            temperature: 温度系数
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比学习风格的模态对齐损失
        
        Args:
            text_features: 文本特征 [B, D]
            visual_features: 视觉特征 [B, D]
            
        Returns:
            loss: 对齐损失
        """
        # L2归一化
        text_features = F.normalize(text_features, dim=-1)
        visual_features = F.normalize(visual_features, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(text_features, visual_features.t()) / self.temperature
        
        # 对称的对比损失
        batch_size = text_features.size(0)
        labels = torch.arange(batch_size, device=text_features.device)
        
        loss_text = F.cross_entropy(similarity, labels)
        loss_visual = F.cross_entropy(similarity.t(), labels)
        
        return (loss_text + loss_visual) / 2


class ModalityDisentangleLoss(nn.Module):
    """
    模态解耦损失
    促进不同模态学习互补信息
    """
    
    def __init__(self, margin: float = 0.5):
        """
        初始化解耦损失
        
        Args:
            margin: 正交性边界
        """
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算模态解耦损失（正交性约束）
        
        Args:
            text_features: 文本特征 [B, D]
            visual_features: 视觉特征 [B, D]
            
        Returns:
            loss: 解耦损失
        """
        # 归一化
        text_features = F.normalize(text_features, dim=-1)
        visual_features = F.normalize(visual_features, dim=-1)
        
        # 计算余弦相似度
        similarity = (text_features * visual_features).sum(dim=-1)
        
        # 鼓励正交（相似度接近0）
        loss = torch.mean(torch.abs(similarity))
        
        return loss
