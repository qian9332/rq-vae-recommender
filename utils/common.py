"""
Utility Functions
通用工具函数
"""

import os
import random
import numpy as np
import torch
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


def set_seed(seed: int = 42):
    """
    设置随机种子，确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        num_params: 参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    获取可用设备
    
    Returns:
        device: torch设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_json(data: Dict, filepath: str):
    """
    保存JSON文件
    
    Args:
        data: 数据字典
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        data: 数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Dict, filepath: str):
    """
    保存YAML文件
    
    Args:
        data: 数据字典
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(filepath: str) -> Dict:
    """
    加载YAML文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        data: 数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class EarlyStopping:
    """
    早停机制
    当验证指标在指定轮数内没有提升时停止训练
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改进量
            mode: 'max'或'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            score: 当前分数
            
        Returns:
            should_stop: 是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """
    平均值计算器
    用于跟踪和计算平均值
    """
    
    def __init__(self):
        """初始化"""
        self.reset()
        
    def reset(self):
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        """
        更新值
        
        Args:
            val: 新值
            n: 数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """
    计时器
    用于测量代码执行时间
    """
    
    def __init__(self):
        """初始化"""
        self.start_time = None
        self.elapsed = 0
        
    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            elapsed: 经过时间（秒）
        """
        if self.start_time is not None:
            self.elapsed = (datetime.now() - self.start_time).total_seconds()
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """上下文管理器出口"""
        self.stop()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        lr: 当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr: float
):
    """
    调整学习率
    
    Args:
        optimizer: 优化器
        lr: 新学习率
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def freeze_model(model: torch.nn.Module):
    """
    冻结模型参数
    
    Args:
        model: 模型
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module):
    """
    解冻模型参数
    
    Args:
        model: 模型
    """
    for param in model.parameters():
        param.requires_grad = True


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    获取模型大小信息
    
    Args:
        model: 模型
        
    Returns:
        size_info: 大小信息字典
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb,
        'num_params': count_parameters(model)
    }


def print_model_summary(model: torch.nn.Module):
    """
    打印模型摘要
    
    Args:
        model: 模型
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"{name}: {param.shape}, {param_count:,} params, trainable={param.requires_grad}")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    size_info = get_model_size(model)
    print(f"Model size: {size_info['total_size_mb']:.2f} MB")
    print("=" * 60)


def ensure_dir(path: str):
    """
    确保目录存在
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def move_to_device(
    data: Any,
    device: torch.device
) -> Any:
    """
    将数据移动到指定设备
    
    Args:
        data: 数据（可以是tensor、dict、list等）
        device: 目标设备
        
    Returns:
        moved_data: 移动后的数据
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data


def normalize_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    归一化张量
    
    Args:
        tensor: 输入张量
        dim: 归一化维度
        eps: 防止除零的小值
        
    Returns:
        normalized: 归一化后的张量
    """
    return tensor / (tensor.norm(dim=dim, keepdim=True) + eps)


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    计算余弦相似度
    
    Args:
        a: 第一个张量
        b: 第二个张量
        dim: 计算维度
        eps: 防止除零的小值
        
    Returns:
        similarity: 相似度
    """
    a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=dim)


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1
) -> float:
    """
    计算Top-K准确率
    
    Args:
        predictions: 预测分数 [B, N]
        targets: 目标索引 [B]
        k: Top-K
        
    Returns:
        accuracy: 准确率
    """
    with torch.no_grad():
        _, top_k_indices = torch.topk(predictions, k, dim=-1)
        correct = top_k_indices.eq(targets.unsqueeze(-1).expand_as(top_k_indices))
        return correct.float().sum().item() / targets.size(0)
