"""
Quick Fine-tuning Script for Testing
快速微调脚本用于测试
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRecommender(nn.Module):
    """简单的推荐模型"""
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_ids, item_features):
        user_emb = self.user_embedding(user_ids)
        # item_features是语义ID的嵌入
        score = self.fc(user_emb * item_features)
        return score


class QuickDataset(Dataset):
    """快速数据集"""
    def __init__(self, interactions, text_features, visual_features, num_items):
        self.interactions = interactions
        self.text_features = text_features
        self.visual_features = visual_features
        self.num_items = num_items
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        inter = self.interactions[idx]
        user_id = inter['user_id']
        item_id = inter['item_id']
        
        # 负采样
        neg_item = np.random.randint(0, self.num_items)
        while neg_item == item_id:
            neg_item = np.random.randint(0, self.num_items)
        
        return {
            'user_id': user_id,
            'pos_item': item_id,
            'neg_item': neg_item,
            'pos_text': self.text_features[item_id].astype(np.float32),
            'pos_visual': self.visual_features[item_id].astype(np.float32),
            'neg_text': self.text_features[neg_item].astype(np.float32),
            'neg_visual': self.visual_features[neg_item].astype(np.float32)
        }


def main():
    logger.info("="*60)
    logger.info("RQ-VAE Recommender Quick Fine-tuning")
    logger.info("="*60)
    
    device = torch.device("cpu")
    
    # 加载数据
    data_dir = "data/amazon_videogames"
    logger.info(f"Loading data from {data_dir}")
    
    import pandas as pd
    interactions_df = pd.read_csv(os.path.join(data_dir, "interactions_processed.csv"))
    text_features = np.load(os.path.join(data_dir, "text_features.npy"))
    visual_features = np.load(os.path.join(data_dir, "visual_features.npy"))
    
    # 转换float16到float32
    text_features = text_features.astype(np.float32)
    visual_features = visual_features.astype(np.float32)
    
    num_users = interactions_df['user_id'].nunique()
    num_items = len(text_features)
    
    interactions = []
    for _, row in interactions_df.iterrows():
        interactions.append({
            'user_id': int(row['user_id']),
            'item_id': int(row['item_id'])
        })
    
    logger.info(f"Data loaded: {len(interactions)} interactions, {num_users} users, {num_items} items")
    
    # 划分数据
    np.random.shuffle(interactions)
    split = int(len(interactions) * 0.8)
    train_data = interactions[:split]
    val_data = interactions[split:]
    
    # 创建数据集
    train_dataset = QuickDataset(train_data, text_features, visual_features, num_items)
    val_dataset = QuickDataset(val_data, text_features, visual_features, num_items)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # 创建模型
    logger.info("Creating models...")
    
    # MMOE编码器
    mmoe = MultiModalFeatureEncoder(
        text_input_dim=768,
        visual_input_dim=2048,
        hidden_dim=256,
        output_dim=128,
        num_experts=4
    ).to(device)
    
    # RQ-VAE
    rq_vae = RQVAE(
        input_dim=128,
        hidden_dim=256,
        embedding_dim=128,
        num_quantization_layers=3,
        codebook_size=256
    ).to(device)
    
    # 加载预训练权重
    pretrained_path = "logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            rq_vae.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pretrained weights loaded")
    
    # 简单推荐模型
    recommender = SimpleRecommender(num_users, num_items, embedding_dim=128).to(device)
    
    # 优化器
    optimizer = AdamW(
        list(mmoe.parameters()) + list(rq_vae.parameters()) + list(recommender.parameters()),
        lr=1e-4
    )
    
    # 创建保存目录
    save_dir = f"logs/finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练
    num_epochs = 5
    history = {'train_loss': [], 'val_loss': [], 'val_recall': []}
    
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        mmoe.train()
        rq_vae.train()
        recommender.train()
        
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            user_ids = torch.tensor(batch['user_id']).to(device)
            pos_text = torch.tensor(batch['pos_text']).to(device)
            pos_visual = torch.tensor(batch['pos_visual']).to(device)
            neg_text = torch.tensor(batch['neg_text']).to(device)
            neg_visual = torch.tensor(batch['neg_visual']).to(device)
            
            # 编码正样本
            pos_fused, _, _, _ = mmoe(pos_text, pos_visual)
            pos_z_q, pos_semantic_ids, vq_loss, _ = rq_vae(pos_fused)
            
            # 编码负样本
            neg_fused, _, _, _ = mmoe(neg_text, neg_visual)
            neg_z_q, _, _, _ = rq_vae(neg_fused)
            
            # 推荐分数
            pos_scores = recommender(user_ids, pos_z_q)
            neg_scores = recommender(user_ids, neg_z_q)
            
            # BPR损失
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss = bpr_loss + 0.1 * vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        mmoe.eval()
        rq_vae.eval()
        recommender.eval()
        
        val_loss = 0
        val_batches = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = torch.tensor(batch['user_id']).to(device)
                pos_text = torch.tensor(batch['pos_text']).to(device)
                pos_visual = torch.tensor(batch['pos_visual']).to(device)
                neg_text = torch.tensor(batch['neg_text']).to(device)
                neg_visual = torch.tensor(batch['neg_visual']).to(device)
                
                pos_fused, _, _, _ = mmoe(pos_text, pos_visual)
                pos_z_q, _, _, _ = rq_vae(pos_fused)
                
                neg_fused, _, _, _ = mmoe(neg_text, neg_visual)
                neg_z_q, _, _, _ = rq_vae(neg_fused)
                
                pos_scores = recommender(user_ids, pos_z_q)
                neg_scores = recommender(user_ids, neg_z_q)
                
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                val_loss += loss.item()
                val_batches += 1
                
                # 计算准确率
                correct += (pos_scores > neg_scores).sum().item()
                total += pos_scores.size(0)
        
        avg_val_loss = val_loss / val_batches
        accuracy = correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_recall'].append(accuracy)
        
        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.4f}")
    
    # 保存结果
    torch.save({
        'mmoe': mmoe.state_dict(),
        'rq_vae': rq_vae.state_dict(),
        'recommender': recommender.state_dict()
    }, os.path.join(save_dir, 'model.pt'))
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Final Accuracy: {accuracy:.4f}")
    logger.info(f"Results saved to {save_dir}")
    
    return history


if __name__ == "__main__":
    main()
