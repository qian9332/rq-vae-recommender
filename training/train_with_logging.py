"""
Enhanced Training Script with Comprehensive Logging
增强版训练脚本，包含详细的日志记录
"""

import os
import sys
import argparse
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config, default_config
from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder, ModalityAlignmentLoss, ModalityDisentangleLoss
from models.recommender import SemanticRecommender
from models.behavior_aware_finetuning import BehaviorAwareRQVAE, BehaviorAwareFineTuner, JointLoss
from data.dataset import MultiModalItemDataset
from utils.metrics import Recall, NDCG, HitRate, MRR
from utils.common import AverageMeter, EarlyStopping, Timer


class TrainingLogger:
    """
    训练日志记录器
    记录训练过程中的所有重要信息
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置文件日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        
        # 配置logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'codebook_usage': [],
            'epoch_time': [],
            'metrics': []
        }
        
        self.log_file = log_file
        
    def log_config(self, config: Config):
        """记录配置"""
        self.logger.info("=" * 60)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 60)
        
        config_dict = {
            'codebook': config.codebook.__dict__,
            'mmoe': config.mmoe.__dict__,
            'recommender': config.recommender.__dict__,
            'training': config.training.__dict__,
            'experiment_name': config.experiment_name,
            'seed': config.seed,
            'device': config.device
        }
        
        for section, params in config_dict.items():
            if isinstance(params, dict):
                self.logger.info(f"\n{section}:")
                for k, v in params.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{section}: {params}")
        
        # 保存配置到JSON
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"\nConfig saved to {config_path}")
        
    def log_model_summary(self, model: nn.Module, model_name: str):
        """记录模型摘要"""
        self.logger.info(f"\n{model_name} Summary:")
        self.logger.info("-" * 40)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            self.logger.debug(f"  {name}: {param.shape}, {param_count:,} params")
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # 模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        self.logger.info(f"Model size: {model_size:.2f} MB")
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info(f"{'='*60}")
        
    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        lr: float,
        extra_info: Optional[Dict] = None
    ):
        """记录batch信息"""
        msg = f"Batch {batch}/{total_batches} - Loss: {loss:.4f} - LR: {lr:.6f}"
        if extra_info:
            for k, v in extra_info.items():
                if isinstance(v, (int, float)):
                    msg += f" - {k}: {v:.4f}"
        self.logger.debug(msg)
        
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch_time: float
    ):
        """记录epoch结束"""
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Time: {epoch_time:.1f}s")
        
        self.logger.info("  Train Metrics:")
        for k, v in train_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}")
            
        self.logger.info("  Val Metrics:")
        for k, v in val_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}")
        
        # 记录到历史
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['epoch_time'].append(epoch_time)
        
    def log_codebook_usage(self, usage_stats: Dict, epoch: int):
        """记录码本使用情况"""
        self.logger.info(f"\nCodebook Usage Statistics (Epoch {epoch}):")
        
        for layer_name, stats in usage_stats.items():
            avg_usage = stats.get('avg_usage_rate', 0)
            max_usage = stats.get('max_usage_rate', 0)
            min_usage = stats.get('min_usage_rate', 0)
            
            self.logger.info(f"  {layer_name}:")
            self.logger.info(f"    Average: {avg_usage:.2%}")
            self.logger.info(f"    Max: {max_usage:.2%}")
            self.logger.info(f"    Min: {min_usage:.2%}")
            
            # 记录到TensorBoard
            self.writer.add_scalar(f'codebook/{layer_name}_avg_usage', avg_usage, epoch)
            
        self.history['codebook_usage'].append(usage_stats)
        
    def log_checkpoint(self, path: str, epoch: int, metric: float):
        """记录检查点保存"""
        self.logger.info(f"\nCheckpoint saved: {path}")
        self.logger.info(f"  Epoch: {epoch}")
        self.logger.info(f"  Metric: {metric:.4f}")
        
    def log_training_complete(self, total_time: float, best_metric: float):
        """记录训练完成"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training Complete!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Metric: {best_metric:.4f}")
        
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        self.logger.info(f"Training history saved to {history_path}")
        
    def close(self):
        """关闭日志"""
        self.save_history()
        self.writer.close()
        self.logger.info(f"\nLogs saved to {self.log_dir}")


class CheckpointManager:
    """
    检查点管理器
    管理模型检查点的保存和加载
    """
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            max_checkpoints: 最大检查点数量
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        self.checkpoints = []
        self.best_metric = 0.0
        self.best_checkpoint = None
        
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict,
        filename: str = None
    ):
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            metrics: 指标字典
            filename: 文件名
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
            
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # 管理检查点数量
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                
        return path
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict,
        metric_name: str = 'loss',
        mode: str = 'min'
    ):
        """
        保存最佳模型
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            metrics: 指标字典
            metric_name: 指标名称
            mode: 'min' or 'max'
        """
        current_metric = metrics.get(metric_name, 0)
        
        is_best = False
        if mode == 'min':
            is_best = current_metric < self.best_metric or self.best_metric == 0
        else:
            is_best = current_metric > self.best_metric
            
        if is_best:
            self.best_metric = current_metric
            path = os.path.join(self.save_dir, 'best_model.pt')
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'best_metric': self.best_metric,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, path)
            self.best_checkpoint = path
            
        return is_best
    
    def load(self, path: str, model: nn.Module, optimizer=None, scheduler=None):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint


class RQVAETrainer:
    """
    RQ-VAE训练器
    包含完整的训练流程和日志记录
    """
    
    def __init__(
        self,
        config: Config,
        data_dir: str,
        log_dir: str
    ):
        """
        初始化训练器
        
        Args:
            config: 配置
            data_dir: 数据目录
            log_dir: 日志目录
        """
        self.config = config
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 日志记录器
        self.logger = TrainingLogger(log_dir, config.experiment_name)
        self.logger.log_config(config)
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            os.path.join(log_dir, 'checkpoints')
        )
        
        # 计时器
        self.timer = Timer()
        
        # 全局步数
        self.global_step = 0
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """准备数据"""
        self.logger.logger.info("\nPreparing Data...")
        
        # 加载特征
        text_features = np.load(os.path.join(self.data_dir, 'text_features.npy'))
        visual_features = np.load(os.path.join(self.data_dir, 'visual_features.npy'))
        
        self.logger.logger.info(f"Text features: {text_features.shape}")
        self.logger.logger.info(f"Visual features: {visual_features.shape}")
        
        # 转换为float32（如果需要）
        if text_features.dtype == np.float16:
            text_features = text_features.astype(np.float32)
        if visual_features.dtype == np.float16:
            visual_features = visual_features.astype(np.float32)
        
        # 创建数据集
        dataset = MultiModalItemDataset(text_features, visual_features)
        
        # 划分训练集和验证集
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.logger.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.logger.info(f"Batch size: {self.config.training.batch_size}")
        
        return train_loader, val_loader
    
    def build_models(self) -> Tuple[MultiModalFeatureEncoder, RQVAE]:
        """构建模型"""
        self.logger.logger.info("\nBuilding Models...")
        
        # MMOE编码器
        mmoe_encoder = MultiModalFeatureEncoder(
            text_input_dim=self.config.mmoe.text_input_dim,
            visual_input_dim=self.config.mmoe.visual_input_dim,
            hidden_dim=self.config.mmoe.hidden_dim,
            output_dim=self.config.mmoe.output_dim,
            num_experts=self.config.mmoe.num_experts,
            expert_hidden_dim=self.config.mmoe.expert_hidden_dim,
            dropout=self.config.mmoe.dropout
        )
        
        # RQ-VAE
        rq_vae = RQVAE(
            input_dim=self.config.mmoe.output_dim,
            hidden_dim=self.config.codebook.embedding_dim * 2,
            embedding_dim=self.config.codebook.embedding_dim,
            num_quantization_layers=self.config.codebook.num_layers,
            codebook_size=self.config.codebook.codebook_size,
            commitment_cost=self.config.codebook.commitment_cost,
            ema_decay=self.config.codebook.ema_decay,
            dead_code_threshold=self.config.codebook.dead_code_threshold,
            dead_code_reset_threshold=self.config.codebook.dead_code_reset_threshold
        )
        
        # 记录模型摘要
        self.logger.log_model_summary(mmoe_encoder, "MMOE Encoder")
        self.logger.log_model_summary(rq_vae, "RQ-VAE")
        
        # 移动到设备
        mmoe_encoder.to(self.device)
        rq_vae.to(self.device)
        
        return mmoe_encoder, rq_vae
    
    def train_epoch(
        self,
        mmoe_encoder: nn.Module,
        rq_vae: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int
    ) -> Dict:
        """训练一个epoch"""
        mmoe_encoder.train()
        rq_vae.train()
        
        meters = {
            'loss': AverageMeter(),
            'reconstruction_loss': AverageMeter(),
            'commitment_loss': AverageMeter(),
            'alignment_loss': AverageMeter(),
            'usage_rate': AverageMeter()
        }
        
        alignment_loss_fn = ModalityAlignmentLoss()
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            text_features = batch['text_features'].to(self.device)
            visual_features = batch['visual_features'].to(self.device)
            
            # MMOE编码
            fused_features, text_out, visual_out, mmoe_info = mmoe_encoder(
                text_features, visual_features
            )
            
            # RQ-VAE
            x_recon, semantic_ids, loss, rq_info = rq_vae(fused_features)
            
            # 模态对齐损失
            align_loss = alignment_loss_fn(text_out, visual_out)
            
            # 总损失
            total_loss = loss + 0.1 * align_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(mmoe_encoder.parameters()) + list(rq_vae.parameters()),
                self.config.training.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # 更新统计
            batch_size = text_features.size(0)
            meters['loss'].update(total_loss.item(), batch_size)
            meters['reconstruction_loss'].update(
                rq_info.get('reconstruction_loss', 0), batch_size
            )
            meters['commitment_loss'].update(
                rq_info.get('total_commitment_loss', 0), batch_size
            )
            meters['alignment_loss'].update(align_loss.item(), batch_size)
            
            if 'usage_rates' in rq_info and len(rq_info['usage_rates']) > 0:
                meters['usage_rate'].update(
                    np.mean(rq_info['usage_rates']), batch_size
                )
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{meters["loss"].avg:.4f}',
                'recon': f'{meters["reconstruction_loss"].avg:.4f}',
                'usage': f'{meters["usage_rate"].avg:.2%}'
            })
            
            # 记录到TensorBoard
            self.logger.writer.add_scalar(
                'train/loss', total_loss.item(), self.global_step
            )
            self.logger.writer.add_scalar(
                'train/lr', optimizer.param_groups[0]['lr'], self.global_step
            )
            
            self.global_step += 1
            
            # 定期记录batch信息
            if batch_idx % 50 == 0:
                self.logger.log_batch(
                    epoch, batch_idx, len(train_loader),
                    total_loss.item(), optimizer.param_groups[0]['lr'],
                    {'usage': meters['usage_rate'].avg}
                )
        
        return {k: v.avg for k, v in meters.items()}
    
    @torch.no_grad()
    def validate(
        self,
        mmoe_encoder: nn.Module,
        rq_vae: nn.Module,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """验证"""
        mmoe_encoder.eval()
        rq_vae.eval()
        
        meters = {
            'loss': AverageMeter(),
            'reconstruction_loss': AverageMeter(),
            'usage_rate': AverageMeter()
        }
        
        for batch in tqdm(val_loader, desc="Validating"):
            text_features = batch['text_features'].to(self.device)
            visual_features = batch['visual_features'].to(self.device)
            
            fused_features, _, _, _ = mmoe_encoder(text_features, visual_features)
            _, _, loss, rq_info = rq_vae(fused_features)
            
            batch_size = text_features.size(0)
            meters['loss'].update(loss.item(), batch_size)
            meters['reconstruction_loss'].update(
                rq_info.get('reconstruction_loss', 0), batch_size
            )
            
            if 'usage_rates' in rq_info and len(rq_info['usage_rates']) > 0:
                meters['usage_rate'].update(
                    np.mean(rq_info['usage_rates']), batch_size
                )
        
        metrics = {f'val_{k}': v.avg for k, v in meters.items()}
        
        # 记录到TensorBoard
        for k, v in metrics.items():
            self.logger.writer.add_scalar(f'val/{k}', v, epoch)
        
        return metrics
    
    def train(self, num_epochs: int):
        """完整训练流程"""
        self.logger.logger.info("\n" + "=" * 60)
        self.logger.logger.info("Starting RQ-VAE Pretraining")
        self.logger.logger.info("=" * 60)
        self.logger.logger.info(f"Device: {self.device}")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 构建模型
        mmoe_encoder, rq_vae = self.build_models()
        
        # 优化器
        optimizer = optim.AdamW(
            list(mmoe_encoder.parameters()) + list(rq_vae.parameters()),
            lr=self.config.training.pretrain_lr,
            weight_decay=self.config.training.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.pretrain_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # 记录优化器配置
        self.logger.logger.info(f"\nOptimizer: AdamW")
        self.logger.logger.info(f"Learning rate: {self.config.training.pretrain_lr}")
        self.logger.logger.info(f"Weight decay: {self.config.training.weight_decay}")
        
        # 开始训练
        self.timer.start()
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            self.logger.log_epoch_start(epoch, num_epochs)
            
            # 训练
            train_metrics = self.train_epoch(
                mmoe_encoder, rq_vae, train_loader, 
                optimizer, scheduler, epoch
            )
            
            # 验证
            val_metrics = self.validate(mmoe_encoder, rq_vae, val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # 记录epoch结果
            self.logger.log_epoch_end(epoch, train_metrics, val_metrics, epoch_time)
            
            # 记录码本使用情况
            usage_stats = rq_vae.get_codebook_usage()
            self.logger.log_codebook_usage(usage_stats, epoch)
            
            # 保存检查点
            checkpoint_path = self.checkpoint_manager.save(
                rq_vae, optimizer, scheduler, epoch, 
                {**train_metrics, **val_metrics}
            )
            self.logger.log_checkpoint(checkpoint_path, epoch, val_metrics.get('val_loss', 0))
            
            # 保存最佳模型
            is_best = self.checkpoint_manager.save_best(
                rq_vae, optimizer, scheduler, epoch,
                val_metrics, metric_name='val_loss', mode='min'
            )
            if is_best:
                self.logger.logger.info(f"  New best model! Val loss: {val_metrics['val_loss']:.4f}")
            
            # 记录学习率
            self.logger.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 训练完成
        total_time = self.timer.stop()
        self.logger.log_training_complete(total_time, self.checkpoint_manager.best_metric)
        self.logger.close()
        
        return mmoe_encoder, rq_vae


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQ-VAE Training with Comprehensive Logging")
    parser.add_argument('--data_dir', type=str, default='./data/amazon_videogames',
                        help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建配置
    config = default_config
    config.training.batch_size = args.batch_size
    config.training.pretrain_lr = args.lr
    config.device = args.device
    config.experiment_name = "rq_vae_pretraining"
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{config.experiment_name}_{timestamp}")
    
    # 创建训练器并训练
    trainer = RQVAETrainer(config, args.data_dir, log_dir)
    
    try:
        mmoe_encoder, rq_vae = trainer.train(args.num_epochs)
        print(f"\nTraining completed successfully!")
        print(f"Logs saved to: {log_dir}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
        trainer.logger.close()


if __name__ == "__main__":
    main()
