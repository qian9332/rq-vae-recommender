"""
Evaluation Metrics
评估指标：包含推荐系统常用评估指标
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """评估指标基类"""
    
    def __init__(self, k: Optional[int] = None):
        """
        初始化
        
        Args:
            k: Top-K参数
        """
        self.k = k
        
    @abstractmethod
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        计算指标
        
        Args:
            scores: 预测分数 [B, N] 或 [B]
            labels: 真实标签 [B, N] 或 [B]
            
        Returns:
            metric_value: 指标值
        """
        pass


class Recall(BaseMetric):
    """
    Recall@K
    召回率：推荐列表中相关物品占所有相关物品的比例
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        计算Recall@K
        
        Args:
            scores: 预测分数 [B, N]
            labels: 真实标签 [B, N]（二值）
            
        Returns:
            recall: 召回率
        """
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        # 获取Top-K索引
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        # 计算命中数
        batch_size = scores.size(0)
        recalls = []
        
        for i in range(batch_size):
            relevant = (labels[i] == 1).sum().item()
            if relevant == 0:
                continue
            
            top_k_labels = labels[i][top_k_indices[i]]
            hits = top_k_labels.sum().item()
            recalls.append(hits / relevant)
        
        return np.mean(recalls) if recalls else 0.0


class Precision(BaseMetric):
    """
    Precision@K
    准确率：推荐列表中相关物品占推荐列表的比例
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算Precision@K"""
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        batch_size = scores.size(0)
        precisions = []
        
        for i in range(batch_size):
            top_k_labels = labels[i][top_k_indices[i]]
            precision = top_k_labels.sum().item() / k
            precisions.append(precision)
        
        return np.mean(precisions)


class NDCG(BaseMetric):
    """
    NDCG@K (Normalized Discounted Cumulative Gain)
    归一化折损累积增益：考虑排序位置的评估指标
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算NDCG@K"""
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        batch_size = scores.size(0)
        ndcgs = []
        
        for i in range(batch_size):
            # 计算DCG
            top_k_labels = labels[i][top_k_indices[i]].float()
            gains = torch.pow(2.0, top_k_labels) - 1
            discounts = torch.log2(torch.arange(1, k + 1, dtype=torch.float) + 1)
            dcg = (gains / discounts).sum().item()
            
            # 计算IDCG
            ideal_labels, _ = torch.sort(labels[i], descending=True)
            ideal_labels = ideal_labels[:k].float()
            ideal_gains = torch.pow(2.0, ideal_labels) - 1
            idcg = (ideal_gains / discounts).sum().item()
            
            # 计算NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0


class HitRate(BaseMetric):
    """
    Hit Rate@K
    命中率：推荐列表中是否包含至少一个相关物品
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算Hit Rate@K"""
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        batch_size = scores.size(0)
        hits = 0
        
        for i in range(batch_size):
            top_k_labels = labels[i][top_k_indices[i]]
            if top_k_labels.sum() > 0:
                hits += 1
        
        return hits / batch_size


class MRR(BaseMetric):
    """
    MRR (Mean Reciprocal Rank)
    平均倒数排名：第一个相关物品排名的倒数
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算MRR"""
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        
        batch_size = scores.size(0)
        reciprocal_ranks = []
        
        for i in range(batch_size):
            sorted_labels = labels[i][sorted_indices[i]]
            # 找到第一个相关物品的位置
            for rank, label in enumerate(sorted_labels):
                if label == 1:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


class AUC(BaseMetric):
    """
    AUC (Area Under ROC Curve)
    ROC曲线下面积
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算AUC"""
        scores = scores.flatten()
        labels = labels.flatten()
        
        # 排序
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # 计算TPR和FPR
        total_pos = sorted_labels.sum().item()
        total_neg = len(sorted_labels) - total_pos
        
        if total_pos == 0 or total_neg == 0:
            return 0.5
        
        tp = 0
        fp = 0
        auc = 0.0
        prev_fp = 0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                # 梯形法则计算AUC
                auc += tp
        
        auc = auc / (total_pos * total_neg)
        return auc


class Coverage(BaseMetric):
    """
    Coverage@K
    覆盖率：推荐列表覆盖的物品占总物品的比例
    """
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        num_total_items: int
    ) -> float:
        """计算Coverage@K"""
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        # 统计推荐的物品
        recommended_items = set()
        for indices in top_k_indices:
            recommended_items.update(indices.tolist())
        
        return len(recommended_items) / num_total_items


class Diversity(BaseMetric):
    """
    Diversity@K
    多样性：推荐列表中物品的多样性
    """
    
    def __init__(self, k: Optional[int] = None, item_features: Optional[torch.Tensor] = None):
        super().__init__(k)
        self.item_features = item_features
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算Diversity@K"""
        if self.item_features is None:
            return 0.0
        
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        batch_size = scores.size(0)
        diversities = []
        
        for i in range(batch_size):
            indices = top_k_indices[i]
            features = self.item_features[indices]
            
            # 计算物品特征的平均相似度
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            similarity_matrix = torch.mm(features, features.t())
            
            # 排除对角线
            mask = 1 - torch.eye(k, device=similarity_matrix.device)
            avg_similarity = (similarity_matrix * mask).sum() / (k * (k - 1))
            
            # 多样性 = 1 - 平均相似度
            diversities.append(1 - avg_similarity.item())
        
        return np.mean(diversities)


class Novelty(BaseMetric):
    """
    Novelty@K
    新颖性：推荐物品的平均流行度的倒数
    """
    
    def __init__(self, k: Optional[int] = None, item_popularity: Optional[torch.Tensor] = None):
        super().__init__(k)
        self.item_popularity = item_popularity
    
    def __call__(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算Novelty@K"""
        if self.item_popularity is None:
            return 0.0
        
        if self.k is None:
            k = scores.size(1)
        else:
            k = min(self.k, scores.size(1))
        
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        
        batch_size = scores.size(0)
        novelties = []
        
        for i in range(batch_size):
            indices = top_k_indices[i]
            popularities = self.item_popularity[indices]
            
            # 新颖性 = -log(流行度)
            novelty = -torch.log(popularities + 1e-10).mean().item()
            novelties.append(novelty)
        
        return np.mean(novelties)


class MetricsCalculator:
    """
    指标计算器
    统一计算多个指标
    """
    
    def __init__(
        self,
        metrics: List[str] = ['recall@10', 'ndcg@10', 'hit_rate@10', 'mrr'],
        k_values: List[int] = [5, 10, 20]
    ):
        """
        初始化指标计算器
        
        Args:
            metrics: 要计算的指标列表
            k_values: Top-K值列表
        """
        self.metric_functions = {}
        
        for metric_name in metrics:
            metric_lower = metric_name.lower()
            
            if 'recall' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metric_functions[metric_name] = Recall(k=k)
            elif 'precision' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metric_functions[metric_name] = Precision(k=k)
            elif 'ndcg' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metric_functions[metric_name] = NDCG(k=k)
            elif 'hit_rate' in metric_lower or 'hr' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metric_functions[metric_name] = HitRate(k=k)
            elif 'mrr' in metric_lower:
                self.metric_functions[metric_name] = MRR()
            elif 'auc' in metric_lower:
                self.metric_functions[metric_name] = AUC()
    
    def compute(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            scores: 预测分数
            labels: 真实标签
            
        Returns:
            metrics: 指标字典
        """
        results = {}
        for name, metric_fn in self.metric_functions.items():
            results[name] = metric_fn(scores, labels)
        return results


def evaluate_model(
    model,
    test_loader,
    device: str = 'cuda',
    metrics: List[str] = ['recall@10', 'ndcg@10', 'hit_rate@10', 'mrr']
) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 推荐模型
        test_loader: 测试数据加载器
        device: 设备
        metrics: 要计算的指标
        
    Returns:
        results: 评估结果
    """
    model.eval()
    calculator = MetricsCalculator(metrics)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 根据模型类型处理
            # 这里需要根据具体模型调整
            pass
    
    # 计算指标
    if all_scores and all_labels:
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        results = calculator.compute(all_scores, all_labels)
    else:
        results = {m: 0.0 for m in metrics}
    
    return results
