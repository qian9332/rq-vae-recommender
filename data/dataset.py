"""
RQ-VAE Recommender Dataset Module
数据集处理模块
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch


class RQVAEDataset(Dataset):
    """
    RQ-VAE预训练数据集
    用于学习商品特征的语义ID编码
    """
    
    def __init__(
        self,
        text_features: np.ndarray,
        visual_features: np.ndarray,
        transform=None
    ):
        """
        初始化RQ-VAE数据集
        
        Args:
            text_features: 文本特征 [N, 768]
            visual_features: 视觉特征 [N, 2048]
            transform: 数据增强
        """
        self.text_features = text_features.astype(np.float32)
        self.visual_features = visual_features.astype(np.float32)
        self.transform = transform
        
        assert len(text_features) == len(visual_features), \
            "文本特征和视觉特征数量不匹配"
        
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        text_feat = self.text_features[idx]
        visual_feat = self.visual_features[idx]
        
        if self.transform:
            text_feat, visual_feat = self.transform(text_feat, visual_feat)
        
        return {
            'text_features': torch.from_numpy(text_feat),
            'visual_features': torch.from_numpy(visual_feat),
            'item_idx': idx
        }


class SequentialRecommendationDataset(Dataset):
    """
    序列推荐数据集
    用于推荐模型微调
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        text_features: np.ndarray,
        visual_features: np.ndarray,
        user_id_map: Dict,
        item_id_map: Dict,
        max_seq_length: int = 50,
        num_negatives: int = 4,
        mode: str = 'train'
    ):
        """
        初始化序列推荐数据集
        
        Args:
            interactions: 交互数据DataFrame
            text_features: 文本特征
            visual_features: 视觉特征
            user_id_map: 用户ID映射
            item_id_map: 商品ID映射
            max_seq_length: 最大序列长度
            num_negatives: 负采样数量
            mode: train/val/test
        """
        self.interactions = interactions
        self.text_features = text_features.astype(np.float32)
        self.visual_features = visual_features.astype(np.float32)
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.mode = mode
        
        # 按用户分组交互
        self.user_sequences = self._group_by_user()
        self.all_items = list(item_id_map.values())
        
    def _group_by_user(self) -> Dict:
        """按用户分组交互"""
        user_seqs = {}
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            if user_id not in user_seqs:
                user_seqs[user_id] = []
            user_seqs[user_id].append({
                'item_id': row['item_id'],
                'timestamp': row.get('timestamp', 0)
            })
        
        # 按时间排序
        for user_id in user_seqs:
            user_seqs[user_id].sort(key=lambda x: x['timestamp'])
        
        return user_seqs
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        target_item = row['item_id']
        
        # 获取用户历史序列
        user_seq = self.user_sequences.get(user_id, [])
        
        # 排除目标商品
        history = [i['item_id'] for i in user_seq if i['item_id'] != target_item]
        
        # 截断或填充
        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length:]
        
        history_mask = [1] * len(history)
        
        # 填充
        while len(history) < self.max_seq_length:
            history.append(0)
            history_mask.append(0)
        
        # 负采样
        negatives = []
        for _ in range(self.num_negatives):
            neg = np.random.choice(self.all_items)
            while neg == target_item or neg in negatives:
                neg = np.random.choice(self.all_items)
            negatives.append(neg)
        
        return {
            'user_id': user_id,
            'history_items': np.array(history),
            'history_mask': np.array(history_mask),
            'target_item': target_item,
            'target_text': self.text_features[target_item],
            'target_visual': self.visual_features[target_item],
            'negatives': np.array(negatives)
        }


def load_amazon_data(data_dir: str) -> Tuple:
    """
    加载Amazon数据集
    
    Args:
        data_dir: 数据目录
        
    Returns:
        text_features, visual_features, interactions, id_mappings
    """
    # 加载特征
    text_features = np.load(os.path.join(data_dir, 'text_features.npy'))
    visual_features = np.load(os.path.join(data_dir, 'visual_features.npy'))
    
    # 加载交互数据
    interactions = pd.read_csv(os.path.join(data_dir, 'interactions_processed.csv'))
    
    # 加载ID映射
    with open(os.path.join(data_dir, 'id_mappings.pkl'), 'rb') as f:
        id_mappings = pickle.load(f)
    
    return text_features, visual_features, interactions, id_mappings


def create_pretrain_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    创建预训练数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        
    Returns:
        train_loader, val_loader
    """
    text_features, visual_features, _, _ = load_amazon_data(data_dir)
    
    # 划分训练/验证集
    num_items = len(text_features)
    indices = np.random.permutation(num_items)
    train_size = int(num_items * train_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建数据集
    train_dataset = RQVAEDataset(
        text_features[train_indices],
        visual_features[train_indices]
    )
    val_dataset = RQVAEDataset(
        text_features[val_indices],
        visual_features[val_indices]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def create_finetune_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    max_seq_length: int = 50,
    num_negatives: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建微调数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        max_seq_length: 最大序列长度
        num_negatives: 负采样数量
        
    Returns:
        train_loader, val_loader, test_loader
    """
    text_features, visual_features, interactions, id_mappings = load_amazon_data(data_dir)
    
    # 划分数据集
    num_interactions = len(interactions)
    indices = np.random.permutation(num_interactions)
    
    train_size = int(num_interactions * 0.7)
    val_size = int(num_interactions * 0.15)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建ID映射
    user_id_map = id_mappings.get('user_id_to_idx', {})
    item_id_map = id_mappings.get('item_id_to_idx', {})
    
    # 创建数据集
    train_dataset = SequentialRecommendationDataset(
        interactions.iloc[train_indices],
        text_features, visual_features,
        user_id_map, item_id_map,
        max_seq_length, num_negatives, 'train'
    )
    val_dataset = SequentialRecommendationDataset(
        interactions.iloc[val_indices],
        text_features, visual_features,
        user_id_map, item_id_map,
        max_seq_length, num_negatives, 'val'
    )
    test_dataset = SequentialRecommendationDataset(
        interactions.iloc[test_indices],
        text_features, visual_features,
        user_id_map, item_id_map,
        max_seq_length, num_negatives, 'test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(data_dir: str) -> Dict:
    """
    获取数据集信息
    
    Args:
        data_dir: 数据目录
        
    Returns:
        数据集信息字典
    """
    text_features, visual_features, interactions, id_mappings = load_amazon_data(data_dir)
    
    return {
        'num_items': len(text_features),
        'num_users': interactions['user_id'].nunique(),
        'num_interactions': len(interactions),
        'text_feature_dim': text_features.shape[1],
        'visual_feature_dim': visual_features.shape[1],
        'avg_interactions_per_user': len(interactions) / interactions['user_id'].nunique(),
        'avg_interactions_per_item': len(interactions) / len(text_features)
    }


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "data/amazon_videogames"
    
    print("加载数据集...")
    info = get_dataset_info(data_dir)
    
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n创建预训练数据加载器...")
    train_loader, val_loader = create_pretrain_dataloaders(data_dir, batch_size=32)
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    print("\n测试数据加载...")
    batch = next(iter(train_loader))
    print(f"  文本特征形状: {batch['text_features'].shape}")
    print(f"  视觉特征形状: {batch['visual_features'].shape}")
