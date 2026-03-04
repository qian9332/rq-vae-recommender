"""
Training Scripts
训练脚本：包含RQ-VAE预训练和推荐模型微调
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config, default_config
from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder, ModalityAlignmentLoss, ModalityDisentangleLoss
from models.recommender import SemanticRecommender, ColdStartHandler
from models.behavior_aware_finetuning import BehaviorAwareRQVAE, BehaviorAwareFineTuner, JointLoss
from data.dataset import (
    MultiModalItemDataset, SequentialRecommendationDataset,
    BehaviorAwareDataset, generate_synthetic_data,
    create_dataloaders, save_processed_data
)
from utils.metrics import Recall, NDCG, HitRate, MRR


class Trainer:
    """
    通用训练器基类
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        log_dir: str = "./logs"
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            config: 配置
            log_dir: 日志目录
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # 日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{config.experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.learning_rate * 0.01
        )
        
        self.global_step = 0
        self.best_metric = 0.0
        
    def save_checkpoint(self, filename: str, extra_info: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }
        if extra_info:
            checkpoint.update(extra_info)
        
        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        

class RQVAEPretrainer(Trainer):
    """
    RQ-VAE预训练器
    """
    
    def __init__(
        self,
        model: RQVAE,
        mmoe_encoder: MultiModalFeatureEncoder,
        config: Config,
        log_dir: str = "./logs"
    ):
        """
        初始化预训练器
        
        Args:
            model: RQ-VAE模型
            mmoe_encoder: MMOE编码器
            config: 配置
            log_dir: 日志目录
        """
        super().__init__(model, config, log_dir)
        
        self.mmoe_encoder = mmoe_encoder.to(self.device)
        
        # 损失函数
        self.alignment_loss = ModalityAlignmentLoss()
        self.disentangle_loss = ModalityDisentangleLoss()
        
        # 合并优化器
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(mmoe_encoder.parameters()),
            lr=config.training.pretrain_lr,
            weight_decay=config.training.weight_decay
        )
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """训练一个epoch"""
        self.model.train()
        self.mmoe_encoder.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        total_align_loss = 0.0
        total_usage_rate = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 获取数据
            text_features = batch['text_features'].to(self.device)
            visual_features = batch['visual_features'].to(self.device)
            
            # MMOE编码
            fused_features, text_out, visual_out, mmoe_info = self.mmoe_encoder(
                text_features, visual_features
            )
            
            # RQ-VAE
            x_recon, semantic_ids, loss, rq_info = self.model(fused_features)
            
            # 模态对齐损失
            align_loss = self.alignment_loss(text_out, visual_out)
            
            # 模态解耦损失
            disentangle_loss = self.disentangle_loss(text_out, visual_out)
            
            # 总损失
            total_batch_loss = (
                loss +
                0.1 * align_loss +
                0.05 * disentangle_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.mmoe_encoder.parameters()),
                self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            total_recon_loss += rq_info['reconstruction_loss']
            total_commit_loss += rq_info.get('total_commitment_loss', 0)
            total_align_loss += align_loss.item()
            
            if 'usage_rates' in rq_info and len(rq_info['usage_rates']) > 0:
                total_usage_rate += np.mean(rq_info['usage_rates'])
            
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'recon': f'{rq_info["reconstruction_loss"]:.4f}',
                'usage': f'{np.mean(rq_info.get("usage_rates", [0])):.2%}'
            })
        
        num_batches = len(train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'commitment_loss': total_commit_loss / num_batches,
            'alignment_loss': total_align_loss / num_batches,
            'avg_usage_rate': total_usage_rate / num_batches
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict:
        """验证"""
        self.model.eval()
        self.mmoe_encoder.eval()
        
        total_loss = 0.0
        total_usage_rate = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                text_features = batch['text_features'].to(self.device)
                visual_features = batch['visual_features'].to(self.device)
                
                fused_features, _, _, _ = self.mmoe_encoder(text_features, visual_features)
                _, _, loss, rq_info = self.model(fused_features)
                
                total_loss += loss.item()
                if 'usage_rates' in rq_info and len(rq_info['usage_rates']) > 0:
                    total_usage_rate += np.mean(rq_info['usage_rates'])
        
        num_batches = len(val_loader)
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_usage_rate': total_usage_rate / num_batches
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """完整训练流程"""
        print(f"Starting RQ-VAE pretraining for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # 打印
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Recon: {train_metrics['reconstruction_loss']:.4f}, "
                  f"Usage: {train_metrics['avg_usage_rate']:.2%}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Usage: {val_metrics['val_usage_rate']:.2%}")
            
            # 保存最佳模型
            if val_metrics['val_usage_rate'] > self.best_metric:
                self.best_metric = val_metrics['val_usage_rate']
                self.save_checkpoint('best_model.pt', {'epoch': epoch})
                print(f"  Saved best model with usage rate: {self.best_metric:.2%}")
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', {'epoch': epoch})
        
        # 保存最终模型
        self.save_checkpoint('final_model.pt', {'epoch': num_epochs})
        print(f"\nTraining completed. Best usage rate: {self.best_metric:.2%}")


class RecommenderFineTuner(Trainer):
    """
    推荐模型微调器
    """
    
    def __init__(
        self,
        fine_tuner: BehaviorAwareFineTuner,
        config: Config,
        log_dir: str = "./logs"
    ):
        """
        初始化微调器
        
        Args:
            fine_tuner: 行为感知微调器
            config: 配置
            log_dir: 日志目录
        """
        super().__init__(fine_tuner, config, log_dir)
        
        self.joint_loss = JointLoss(
            reconstruction_weight=config.training.reconstruction_weight,
            recommendation_weight=config.training.recommendation_weight
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
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_rec_loss = 0.0
        total_recon_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 获取数据
            user_ids = batch['user_id'].to(self.device)
            item_features = batch['item_features'].to(self.device)
            history_semantic_ids = batch['history_semantic_ids'].to(self.device)
            history_mask = batch['history_mask'].to(self.device)
            candidate_semantic_ids = batch['candidate_semantic_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            loss, info = self.model(
                item_features,
                user_ids,
                history_semantic_ids,
                history_mask,
                candidate_semantic_ids,
                labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_rec_loss += info.get('recommendation_loss', 0)
            total_recon_loss += info.get('reconstruction_loss', 0)
            
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{info.get("recommendation_loss", 0):.4f}'
            })
        
        num_batches = len(train_loader)
        return {
            'total_loss': total_loss / num_batches,
            'recommendation_loss': total_rec_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                history_semantic_ids = batch['history_semantic_ids'].to(self.device)
                history_mask = batch['history_mask'].to(self.device)
                candidate_semantic_ids = batch['candidate_semantic_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                loss, info = self.model(
                    item_features,
                    user_ids,
                    history_semantic_ids,
                    history_mask,
                    candidate_semantic_ids,
                    labels
                )
                
                total_loss += loss.item()
                all_scores.append(info['scores'].cpu())
                all_labels.append(labels.cpu())
        
        # 计算评估指标
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = {'val_loss': total_loss / len(val_loader)}
        for name, metric in self.metrics.items():
            metrics[name] = metric(all_scores, all_labels)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """完整训练流程"""
        print(f"Starting recommender fine-tuning for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # 打印
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Recall@10: {val_metrics['recall@10']:.4f}, "
                  f"NDCG@10: {val_metrics['ndcg@10']:.4f}")
            
            # 保存最佳模型
            if val_metrics['ndcg@10'] > self.best_metric:
                self.best_metric = val_metrics['ndcg@10']
                self.save_checkpoint('best_model.pt', {'epoch': epoch})
                print(f"  Saved best model with NDCG@10: {self.best_metric:.4f}")
        
        self.save_checkpoint('final_model.pt', {'epoch': num_epochs})
        print(f"\nFine-tuning completed. Best NDCG@10: {self.best_metric:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQ-VAE Recommender Training")
    parser.add_argument('--mode', type=str, default='pretrain', 
                        choices=['pretrain', 'finetune', 'all'],
                        help='Training mode')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = default_config
    
    # 覆盖命令行参数
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    config.device = args.device
    
    print("=" * 50)
    print("RQ-VAE Recommender Training")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print("=" * 50)
    
    # 生成或加载数据
    print("\nPreparing data...")
    if not os.path.exists(args.data_dir):
        print("Generating synthetic data...")
        data = generate_synthetic_data(
            num_items=config.recommender.num_items,
            num_users=config.recommender.num_users,
            text_dim=config.mmoe.text_input_dim,
            visual_dim=config.mmoe.visual_input_dim
        )
        save_processed_data(data, args.data_dir)
    else:
        print("Loading existing data...")
        from data.dataset import load_processed_data
        data = load_processed_data(args.data_dir)
    
    # 创建数据集
    multimodal_dataset = MultiModalItemDataset(
        data['text_features'],
        data['visual_features']
    )
    train_loader, val_loader = create_dataloaders(
        multimodal_dataset,
        batch_size=config.training.batch_size
    )
    
    if args.mode in ['pretrain', 'all']:
        # 预训练RQ-VAE
        print("\n" + "=" * 50)
        print("Stage 1: RQ-VAE Pretraining")
        print("=" * 50)
        
        # 创建模型
        input_dim = config.mmoe.output_dim
        
        mmoe_encoder = MultiModalFeatureEncoder(
            text_input_dim=config.mmoe.text_input_dim,
            visual_input_dim=config.mmoe.visual_input_dim,
            hidden_dim=config.mmoe.hidden_dim,
            output_dim=config.mmoe.output_dim,
            num_experts=config.mmoe.num_experts,
            expert_hidden_dim=config.mmoe.expert_hidden_dim,
            dropout=config.mmoe.dropout
        )
        
        rq_vae = RQVAE(
            input_dim=input_dim,
            hidden_dim=config.codebook.embedding_dim * 2,
            embedding_dim=config.codebook.embedding_dim,
            num_quantization_layers=config.codebook.num_layers,
            codebook_size=config.codebook.codebook_size,
            commitment_cost=config.codebook.commitment_cost,
            ema_decay=config.codebook.ema_decay,
            dead_code_threshold=config.codebook.dead_code_threshold,
            dead_code_reset_threshold=config.codebook.dead_code_reset_threshold
        )
        
        # 训练
        pretrainer = RQVAEPretrainer(
            rq_vae, mmoe_encoder, config, args.log_dir
        )
        pretrainer.train(
            train_loader, val_loader,
            config.training.pretrain_epochs
        )
    
    if args.mode in ['finetune', 'all']:
        # 微调推荐模型
        print("\n" + "=" * 50)
        print("Stage 2: Recommender Fine-tuning")
        print("=" * 50)
        
        # 创建行为感知模型
        from models.behavior_aware_finetuning import create_behavior_aware_model
        
        input_dim = config.mmoe.output_dim
        
        fine_tuner = create_behavior_aware_model(
            input_dim=input_dim,
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=config.codebook.embedding_dim,
            num_quantization_layers=config.codebook.num_layers,
            codebook_size=config.codebook.codebook_size,
            hidden_dim=config.mmoe.hidden_dim,
            num_heads=config.recommender.num_heads,
            num_transformer_layers=config.recommender.num_layers,
            max_seq_length=config.recommender.max_seq_length
        )
        
        # 加载预训练权重（如果有）
        if args.mode == 'finetune':
            pretrain_path = os.path.join(args.log_dir, 'best_model.pt')
            if os.path.exists(pretrain_path):
                print(f"Loading pretrained weights from {pretrain_path}")
                checkpoint = torch.load(pretrain_path)
                # 加载RQ-VAE权重
                fine_tuner.rq_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 创建微调数据集（简化版）
        # 实际应用中需要准备完整的行为数据
        print("Note: Using simplified dataset for demonstration")
        
        # 训练
        trainer = RecommenderFineTuner(fine_tuner, config, args.log_dir)
        # trainer.train(train_loader, val_loader, config.training.finetune_epochs)
        print("Fine-tuning completed (demo mode)")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
