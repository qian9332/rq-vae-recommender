"""
RQ-VAE Recommender Evaluation Script
评估脚本
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rq_vae import RQVAE
from models.mmoe_encoder import MultiModalFeatureEncoder
from models.recommender import SemanticRecommender
from data.dataset import load_amazon_data, SequentialRecommendationDataset
from utils.metrics import Recall, NDCG, HitRate, MRR, AUC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        device: str = 'cpu'
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            data_dir: 数据目录
            device: 设备
        """
        self.device = torch.device(device)
        self.data_dir = data_dir
        
        # 加载数据
        logger.info("加载数据...")
        self.text_features, self.visual_features, self.interactions, self.id_mappings = \
            load_amazon_data(data_dir)
        
        # 加载模型
        logger.info("加载模型...")
        self._load_model(model_path)
        
        # 初始化评估指标
        self.metrics = {
            'recall@5': Recall(k=5),
            'recall@10': Recall(k=10),
            'recall@20': Recall(k=20),
            'ndcg@5': NDCG(k=5),
            'ndcg@10': NDCG(k=10),
            'ndcg@20': NDCG(k=20),
            'hit_rate@10': HitRate(k=10),
            'hit_rate@20': HitRate(k=20),
            'mrr': MRR(),
            'auc': AUC()
        }
        
    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 创建MMOE编码器
        self.mmoe_encoder = MultiModalFeatureEncoder(
            text_input_dim=768,
            visual_input_dim=2048,
            hidden_dim=256,
            output_dim=128,
            num_experts=4
        ).to(self.device)
        
        # 创建RQ-VAE
        self.rq_vae = RQVAE(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=128,
            num_quantization_layers=3,
            codebook_size=256
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.rq_vae.load_state_dict(checkpoint['model_state_dict'])
        elif 'rq_vae' in checkpoint:
            self.rq_vae.load_state_dict(checkpoint['rq_vae'])
        
        self.mmoe_encoder.eval()
        self.rq_vae.eval()
        
        # 预计算所有商品的语义ID
        self._precompute_semantic_ids()
        
    def _precompute_semantic_ids(self):
        """预计算所有商品的语义ID"""
        logger.info("预计算语义ID...")
        
        with torch.no_grad():
            text_tensor = torch.tensor(
                self.text_features, dtype=torch.float32
            ).to(self.device)
            visual_tensor = torch.tensor(
                self.visual_features, dtype=torch.float32
            ).to(self.device)
            
            # 分批处理
            batch_size = 256
            all_semantic_ids = []
            
            for i in range(0, len(text_tensor), batch_size):
                batch_text = text_tensor[i:i+batch_size]
                batch_visual = visual_tensor[i:i+batch_size]
                
                fused, _, _, _ = self.mmoe_encoder(batch_text, batch_visual)
                _, semantic_ids, _, _ = self.rq_vae(fused)
                all_semantic_ids.append(semantic_ids.cpu())
            
            self.all_semantic_ids = torch.cat(all_semantic_ids, dim=0)
        
        logger.info(f"预计算完成: {len(self.all_semantic_ids)} 个商品")
    
    def evaluate_rq_vae(self) -> Dict:
        """评估RQ-VAE重建质量"""
        logger.info("评估RQ-VAE重建质量...")
        
        with torch.no_grad():
            text_tensor = torch.tensor(
                self.text_features[:1000], dtype=torch.float32
            ).to(self.device)
            visual_tensor = torch.tensor(
                self.visual_features[:1000], dtype=torch.float32
            ).to(self.device)
            
            fused, _, _, _ = self.mmoe_encoder(text_tensor, visual_tensor)
            z_q, _, vq_loss, info = self.rq_vae(fused)
            
            # 计算重建误差
            recon_error = F.mse_loss(z_q, fused).item()
            
            # 计算码本使用率
            codebook_usage = info.get('codebook_usage', {})
            
        return {
            'reconstruction_error': recon_error,
            'vq_loss': vq_loss.item(),
            'codebook_usage': codebook_usage
        }
    
    def evaluate_recommendation(
        self,
        test_loader: DataLoader,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """评估推荐效果"""
        logger.info("评估推荐效果...")
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中"):
                # 获取目标商品语义ID
                target_items = batch['target_item'].numpy()
                target_semantic_ids = self.all_semantic_ids[target_items]
                
                # 简单评分：使用语义ID的距离
                # 这里可以替换为更复杂的推荐模型
                user_ids = batch['user_id'].numpy()
                
                # 为每个用户计算候选商品得分
                batch_scores = []
                batch_labels = []
                
                for i, user_id in enumerate(user_ids):
                    target_id = target_items[i]
                    
                    # 随机采样候选商品
                    num_candidates = 100
                    candidates = np.random.choice(
                        len(self.all_semantic_ids),
                        size=num_candidates,
                        replace=False
                    )
                    candidates[0] = target_id  # 确保目标商品在候选中
                    
                    # 计算语义ID距离
                    target_semantic = self.all_semantic_ids[target_id].unsqueeze(0)
                    candidate_semantics = self.all_semantic_ids[candidates]
                    
                    # 距离转换为得分
                    distances = torch.sum(
                        torch.abs(candidate_semantics - target_semantic), dim=1
                    )
                    scores = 1.0 / (1.0 + distances)
                    
                    # 创建标签
                    labels = torch.zeros(num_candidates)
                    labels[0] = 1  # 第一个是正样本
                    
                    batch_scores.append(scores)
                    batch_labels.append(labels)
                
                all_scores.append(torch.stack(batch_scores))
                all_labels.append(torch.stack(batch_labels))
        
        # 合并所有结果
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # 计算指标
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric(scores, labels)
            except Exception as e:
                logger.warning(f"计算 {name} 失败: {e}")
                results[name] = 0.0
        
        return results
    
    def evaluate_cold_start(self) -> Dict:
        """评估冷启动效果"""
        logger.info("评估冷启动效果...")
        
        # 找出交互少的商品
        item_counts = self.interactions['item_id'].value_counts()
        cold_items = item_counts[item_counts <= 2].index.tolist()[:100]
        
        if len(cold_items) == 0:
            return {'cold_start_recall': 'N/A'}
        
        # 对于冷启动商品，检查语义ID是否合理
        cold_semantic_ids = self.all_semantic_ids[cold_items]
        
        # 计算冷启动商品之间的语义相似度
        if len(cold_semantic_ids) > 1:
            similarity_matrix = 1 - torch.cdist(
                cold_semantic_ids.float(),
                cold_semantic_ids.float()
            ) / (3 * 256)  # 归一化
            
            avg_similarity = (similarity_matrix.sum() - len(cold_semantic_ids)) / \
                           (len(cold_semantic_ids) * (len(cold_semantic_ids) - 1))
        else:
            avg_similarity = 0
        
        return {
            'num_cold_items': len(cold_items),
            'avg_semantic_similarity': avg_similarity.item()
        }
    
    def run_full_evaluation(self) -> Dict:
        """运行完整评估"""
        results = {}
        
        # 1. RQ-VAE评估
        rq_vae_results = self.evaluate_rq_vae()
        results['rq_vae'] = rq_vae_results
        
        # 2. 推荐评估
        # 创建测试数据加载器
        from data.dataset import SequentialRecommendationDataset
        import pandas as pd
        
        test_interactions = self.interactions.sample(n=min(1000, len(self.interactions)))
        test_dataset = SequentialRecommendationDataset(
            test_interactions,
            self.text_features,
            self.visual_features,
            self.id_mappings.get('user_id_to_idx', {}),
            self.id_mappings.get('item_id_to_idx', {}),
            max_seq_length=50,
            num_negatives=4,
            mode='test'
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        rec_results = self.evaluate_recommendation(test_loader)
        results['recommendation'] = rec_results
        
        # 3. 冷启动评估
        cold_results = self.evaluate_cold_start()
        results['cold_start'] = cold_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description='RQ-VAE Recommender Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default='data/amazon_videogames', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='logs/evaluation', help='输出目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # 运行评估
    logger.info("开始评估...")
    results = evaluator.run_full_evaluation()
    
    # 打印结果
    logger.info("\n" + "="*60)
    logger.info("评估结果")
    logger.info("="*60)
    
    logger.info("\n【RQ-VAE重建质量】")
    for key, value in results['rq_vae'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n【推荐效果】")
    for key, value in results['recommendation'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n【冷启动效果】")
    for key, value in results['cold_start'].items():
        logger.info(f"  {key}: {value}")
    
    # 保存结果
    output_path = os.path.join(
        args.output_dir,
        f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
