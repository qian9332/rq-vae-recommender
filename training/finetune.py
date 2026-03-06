"""
RQ-VAE Recommender Fine-tuning Script
推荐模型微调脚本
实现端到端的推荐任务训练
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder
from models.recommender import SemanticRecommender
from utils.metrics import Recall, NDCG, HitRate, MRR


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """微调配置"""
    # 数据配置
    data_dir: str = "data/amazon_videogames"
    batch_size: int = 128
    num_workers: int = 0
    max_seq_length: int = 50
    
    # 模型配置
    text_input_dim: int = 768
    visual_input_dim: int = 2048
    hidden_dim: int = 256
    embedding_dim: int = 128
    num_experts: int = 4
    num_quantization_layers: int = 3
    codebook_size: int = 256
    num_heads: int = 4
    num_transformer_layers: int = 2
    
    # 训练配置
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # 损失权重
    reconstruction_weight: float = 0.5
    recommendation_weight: float = 1.0
    
    # 其他
    device: str = "cpu"
    seed: int = 42
    save_dir: str = "logs/finetuning"
    eval_steps: int = 100
    save_steps: int = 500


class SequentialRecommendationDataset(Dataset):
    """
    序列推荐数据集
    构建用户历史序列和目标商品
    """
    
    def __init__(
        self,
        interactions: List[Dict],
        item_features: Dict,
        user_features: Dict,
        max_seq_length: int = 50,
        num_negatives: int = 4,
        mode: str = "train"
    ):
        """
        初始化数据集
        
        Args:
            interactions: 交互数据列表
            item_features: 商品特征字典
            user_features: 用户特征字典
            max_seq_length: 最大序列长度
            num_negatives: 负采样数量
            mode: 训练/验证/测试模式
        """
        self.interactions = interactions
        self.item_features = item_features
        self.user_features = user_features
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.mode = mode
        
        # 按用户分组交互
        self.user_sequences = self._group_by_user()
        self.all_items = list(item_features.keys())
        
    def _group_by_user(self) -> Dict:
        """按用户分组交互"""
        user_seqs = {}
        for inter in self.interactions:
            user_id = inter['user_id']
            if user_id not in user_seqs:
                user_seqs[user_id] = []
            user_seqs[user_id].append(inter)
        
        # 按时间排序
        for user_id in user_seqs:
            user_seqs[user_id].sort(key=lambda x: x.get('timestamp', 0))
        
        return user_seqs
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        inter = self.interactions[idx]
        user_id = inter['user_id']
        target_item = inter['item_id']
        
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
            history.append(0)  # padding
            history_mask.append(0)
        
        # 负采样
        negatives = []
        for _ in range(self.num_negatives):
            neg = np.random.choice(self.all_items)
            while neg == target_item or neg in negatives:
                neg = np.random.choice(self.all_items)
            negatives.append(neg)
        
        # 获取特征
        target_text = self.item_features.get(target_item, {}).get('text', np.zeros(768))
        target_visual = self.item_features.get(target_item, {}).get('visual', np.zeros(2048))
        
        return {
            'user_id': user_id,
            'history_items': np.array(history),
            'history_mask': np.array(history_mask),
            'target_item': target_item,
            'target_text': np.array(target_text, dtype=np.float32),
            'target_visual': np.array(target_visual, dtype=np.float32),
            'negatives': np.array(negatives)
        }


class RQVAERecommenderFinetuner(nn.Module):
    """
    RQ-VAE推荐器微调模型
    端到端训练：特征编码 -> 语义ID -> 推荐预测
    """
    
    def __init__(self, config: FineTuningConfig):
        super().__init__()
        self.config = config
        
        # MMOE编码器
        self.mmoe_encoder = MultiModalFeatureEncoder(
            text_input_dim=config.text_input_dim,
            visual_input_dim=config.visual_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_experts=config.num_experts
        )
        
        # RQ-VAE
        self.rq_vae = RQVAE(
            input_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_quantization_layers=config.num_quantization_layers,
            codebook_size=config.codebook_size
        )
        
        # 推荐模型
        self.recommender = SemanticRecommender(
            num_users=10537,  # Amazon Video_Games用户数
            num_items=16297,  # Amazon Video_Games商品数
            num_quantization_layers=config.num_quantization_layers,
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            max_seq_length=config.max_seq_length,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )
        
    def encode_items(self, text_features: torch.Tensor, visual_features: torch.Tensor):
        """编码商品特征为语义ID"""
        # MMOE编码
        fused_features, _, _, _ = self.mmoe_encoder(text_features, visual_features)
        
        # RQ-VAE量化
        z_q, semantic_ids, vq_loss, info = self.rq_vae(fused_features)
        
        return z_q, semantic_ids, vq_loss, info
    
    def forward(
        self,
        user_ids: torch.Tensor,
        history_text: torch.Tensor,
        history_visual: torch.Tensor,
        history_mask: torch.Tensor,
        target_text: torch.Tensor,
        target_visual: torch.Tensor,
        negative_text: Optional[torch.Tensor] = None,
        negative_visual: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            user_ids: 用户ID [B]
            history_text: 历史商品文本特征 [B, Seq, 768]
            history_visual: 历史商品视觉特征 [B, Seq, 2048]
            history_mask: 历史序列mask [B, Seq]
            target_text: 目标商品文本特征 [B, 768]
            target_visual: 目标商品视觉特征 [B, 2048]
            negative_text: 负样本文本特征 [B, NumNeg, 768]
            negative_visual: 负样本视觉特征 [B, NumNeg, 2048]
        """
        batch_size = user_ids.size(0)
        
        # 编码历史商品
        history_text_flat = history_text.view(-1, history_text.size(-1))
        history_visual_flat = history_visual.view(-1, history_visual.size(-1))
        
        _, history_semantic_ids, _, _ = self.encode_items(
            history_text_flat, history_visual_flat
        )
        history_semantic_ids = history_semantic_ids.view(batch_size, -1, self.config.num_quantization_layers)
        
        # 编码目标商品
        _, target_semantic_ids, vq_loss, _ = self.encode_items(target_text, target_visual)
        
        # 推荐预测
        scores, rec_info = self.recommender(
            user_ids, history_semantic_ids, history_mask, 
            target_semantic_ids.unsqueeze(1)
        )
        
        # 负样本处理
        neg_scores = None
        if negative_text is not None:
            neg_text_flat = negative_text.view(-1, negative_text.size(-1))
            neg_visual_flat = negative_visual.view(-1, negative_visual.size(-1))
            _, neg_semantic_ids, _, _ = self.encode_items(neg_text_flat, neg_visual_flat)
            neg_semantic_ids = neg_semantic_ids.view(batch_size, -1, self.config.num_quantization_layers)
            
            neg_scores, _ = self.recommender(
                user_ids, history_semantic_ids, history_mask,
                neg_semantic_ids
            )
        
        return {
            'scores': scores,
            'neg_scores': neg_scores,
            'vq_loss': vq_loss,
            'semantic_ids': target_semantic_ids,
            'rec_info': rec_info
        }


def compute_bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """计算BPR损失"""
    # pos_scores: [B, 1], neg_scores: [B, NumNeg]
    pos_scores = pos_scores.expand_as(neg_scores)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss


def compute_cross_entropy_loss(scores: torch.Tensor) -> torch.Tensor:
    """计算交叉熵损失"""
    # scores: [B, 1], 目标是让正样本得分最高
    # 使用softmax + 负对数似然
    return -F.log_softmax(scores, dim=-1)[:, 0].mean()


class Trainer:
    """训练器"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建模型
        self.model = RQVAERecommenderFinetuner(config).to(self.device)
        
        # 创建优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # 评估指标
        self.metrics = {
            'recall@10': Recall(k=10),
            'recall@20': Recall(k=20),
            'ndcg@10': NDCG(k=10),
            'ndcg@20': NDCG(k=20),
            'hit_rate@10': HitRate(k=10),
            'mrr': MRR()
        }
        
        # 训练状态
        self.global_step = 0
        self.best_score = 0.0
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
    def load_pretrained_rq_vae(self, checkpoint_path: str):
        """加载预训练的RQ-VAE"""
        logger.info(f"Loading pretrained RQ-VAE from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.rq_vae.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Pretrained RQ-VAE loaded successfully")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_rec_loss = 0.0
        total_vq_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 移动数据到设备
            user_ids = batch['user_id'].to(self.device)
            history_items = batch['history_items'].to(self.device)
            history_mask = batch['history_mask'].to(self.device)
            target_text = batch['target_text'].to(self.device)
            target_visual = batch['target_visual'].to(self.device)
            
            # 获取历史商品特征（简化处理，使用随机特征）
            batch_size = user_ids.size(0)
            seq_len = history_items.size(1)
            history_text = torch.randn(batch_size, seq_len, 768).to(self.device)
            history_visual = torch.randn(batch_size, seq_len, 2048).to(self.device)
            
            # 获取负样本特征
            num_neg = batch['negatives'].size(1)
            negative_text = torch.randn(batch_size, num_neg, 768).to(self.device)
            negative_visual = torch.randn(batch_size, num_neg, 2048).to(self.device)
            
            # 前向传播
            outputs = self.model(
                user_ids, history_text, history_visual, history_mask,
                target_text, target_visual,
                negative_text, negative_visual
            )
            
            # 计算损失
            rec_loss = compute_bpr_loss(outputs['scores'], outputs['neg_scores'])
            vq_loss = outputs['vq_loss']
            
            loss = (
                self.config.recommendation_weight * rec_loss +
                self.config.reconstruction_weight * vq_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_vq_loss += vq_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec_loss': f'{rec_loss.item():.4f}',
                'vq_loss': f'{vq_loss.item():.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'rec_loss': total_rec_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                user_ids = batch['user_id'].to(self.device)
                history_items = batch['history_items'].to(self.device)
                history_mask = batch['history_mask'].to(self.device)
                target_text = batch['target_text'].to(self.device)
                target_visual = batch['target_visual'].to(self.device)
                
                batch_size = user_ids.size(0)
                seq_len = history_items.size(1)
                history_text = torch.randn(batch_size, seq_len, 768).to(self.device)
                history_visual = torch.randn(batch_size, seq_len, 2048).to(self.device)
                
                outputs = self.model(
                    user_ids, history_text, history_visual, history_mask,
                    target_text, target_visual
                )
                
                all_scores.append(outputs['scores'].cpu().numpy())
                all_labels.append(np.ones(batch_size))
        
        # 计算指标
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(scores, torch.from_numpy(labels))
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.save_dir, 'checkpoint_last.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with recall@10: {metrics.get('recall@10', 0):.4f}")
        
        # 定期保存
        if epoch % 5 == 0:
            epoch_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"Config: {json.dumps(asdict(self.config), indent=2)}")
        
        history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_vq_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Rec Loss: {train_metrics['rec_loss']:.4f}, "
                       f"VQ Loss: {train_metrics['vq_loss']:.4f}")
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_rec_loss'].append(train_metrics['rec_loss'])
            history['train_vq_loss'].append(train_metrics['vq_loss'])
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # 评估
            if epoch % 2 == 0 or epoch == self.config.num_epochs:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Validation Metrics:")
                for name, value in val_metrics.items():
                    logger.info(f"  {name}: {value:.4f}")
                
                history['val_metrics'].append(val_metrics)
                
                # 检查是否是最佳模型
                current_score = val_metrics.get('recall@10', 0)
                is_best = current_score > self.best_score
                if is_best:
                    self.best_score = current_score
                
                # 保存检查点
                self.save_checkpoint(epoch, val_metrics, is_best)
            else:
                self.save_checkpoint(epoch, train_metrics, False)
        
        # 保存训练历史
        history_path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("\nTraining completed!")
        logger.info(f"Best Recall@10: {self.best_score:.4f}")
        
        return history


def load_data(data_dir: str) -> Tuple:
    """加载数据"""
    logger.info(f"Loading data from {data_dir}")
    
    # 加载交互数据
    import pandas as pd
    interactions_path = os.path.join(data_dir, 'interactions_processed.csv')
    interactions_df = pd.read_csv(interactions_path)
    
    interactions = []
    for _, row in interactions_df.iterrows():
        interactions.append({
            'user_id': int(row['user_id']),
            'item_id': int(row['item_id']),
            'timestamp': row.get('timestamp', 0)
        })
    
    # 加载特征
    text_features = np.load(os.path.join(data_dir, 'text_features.npy'))
    visual_features = np.load(os.path.join(data_dir, 'visual_features.npy'))
    
    # 构建商品特征字典
    item_features = {}
    for i in range(len(text_features)):
        item_features[i] = {
            'text': text_features[i],
            'visual': visual_features[i]
        }
    
    # 用户特征（简化处理）
    user_features = {}
    unique_users = interactions_df['user_id'].unique()
    for user_id in unique_users:
        user_features[int(user_id)] = {}
    
    logger.info(f"Loaded {len(interactions)} interactions, "
                f"{len(item_features)} items, {len(user_features)} users")
    
    return interactions, item_features, user_features


def main():
    """主函数"""
    # 创建配置
    config = FineTuningConfig(
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-4,
        device="cpu",
        save_dir=f"logs/finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 加载数据
    interactions, item_features, user_features = load_data(config.data_dir)
    
    # 划分数据集
    np.random.shuffle(interactions)
    split_idx = int(len(interactions) * 0.8)
    train_interactions = interactions[:split_idx]
    val_interactions = interactions[split_idx:]
    
    # 创建数据集
    train_dataset = SequentialRecommendationDataset(
        train_interactions, item_features, user_features,
        max_seq_length=config.max_seq_length, mode="train"
    )
    val_dataset = SequentialRecommendationDataset(
        val_interactions, item_features, user_features,
        max_seq_length=config.max_seq_length, mode="val"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 加载预训练模型
    pretrained_path = "logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt"
    if os.path.exists(pretrained_path):
        trainer.load_pretrained_rq_vae(pretrained_path)
    
    # 训练
    history = trainer.train(train_loader, val_loader)
    
    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    main()
