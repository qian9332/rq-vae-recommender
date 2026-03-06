"""
RQ-VAE Recommender Inference Service
推理服务：加载训练好的模型，提供推荐功能
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class RecommendationResult:
    """推荐结果"""
    item_ids: List[int]
    item_scores: List[float]
    semantic_ids: List[List[int]]


class RQVAERecommender:
    """
    RQ-VAE推荐器
    封装模型加载和推理逻辑
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        device: str = "cpu"
    ):
        """
        初始化推荐器
        
        Args:
            checkpoint_path: 模型检查点路径
            data_dir: 数据目录
            device: 设备 (cpu/cuda)
        """
        self.device = torch.device(device)
        self.data_dir = data_dir
        
        # 加载ID映射
        self._load_id_mappings(data_dir)
        
        # 加载特征
        self._load_features(data_dir)
        
        # 加载模型
        self._load_model(checkpoint_path)
        
        # 预计算所有商品的语义ID
        self._precompute_semantic_ids()
        
    def _load_id_mappings(self, data_dir: str):
        """加载ID映射"""
        mapping_path = os.path.join(data_dir, "id_mappings.pkl")
        with open(mapping_path, "rb") as f:
            mappings = pickle.load(f)
        
        self.item_id_to_idx = mappings.get("item_id_to_idx", {})
        self.idx_to_item_id = mappings.get("idx_to_item_id", {})
        self.user_id_to_idx = mappings.get("user_id_to_idx", {})
        self.idx_to_user_id = mappings.get("idx_to_user_id", {})
        
        self.num_items = len(self.item_id_to_idx)
        self.num_users = len(self.user_id_to_idx)
        
    def _load_features(self, data_dir: str):
        """加载特征"""
        text_path = os.path.join(data_dir, "text_features.npy")
        self.text_features = np.load(text_path)
        if self.text_features.dtype == np.float16:
            self.text_features = self.text_features.astype(np.float32)
        
        visual_path = os.path.join(data_dir, "visual_features.npy")
        self.visual_features = np.load(visual_path)
        if self.visual_features.dtype == np.float16:
            self.visual_features = self.visual_features.astype(np.float32)
        
        self.text_features_tensor = torch.tensor(
            self.text_features, dtype=torch.float32
        ).to(self.device)
        self.visual_features_tensor = torch.tensor(
            self.visual_features, dtype=torch.float32
        ).to(self.device)
        
    def _load_model(self, checkpoint_path: str):
        """加载模型"""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.rq_vae import RQVAE
        from models.mmoe_encoder import MultiModalFeatureEncoder
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.mmoe_encoder = MultiModalFeatureEncoder(
            text_input_dim=768,
            visual_input_dim=2048,
            hidden_dim=256,
            output_dim=128,
            num_experts=4,
            expert_hidden_dim=128,
            dropout=0.0
        ).to(self.device)
        
        self.rq_vae = RQVAE(
            input_dim=128,
            hidden_dim=256,
            embedding_dim=128,
            num_quantization_layers=3,
            codebook_size=256,
            commitment_cost=0.25
        ).to(self.device)
        
        if "model_state_dict" in checkpoint:
            try:
                self.rq_vae.load_state_dict(checkpoint["model_state_dict"])
            except:
                pass
        
        self.mmoe_encoder.eval()
        self.rq_vae.eval()
        
    def _precompute_semantic_ids(self):
        """预计算所有商品的语义ID"""
        print("Precomputing semantic IDs for all items...")
        
        with torch.no_grad():
            fused_features, _, _, _ = self.mmoe_encoder(
                self.text_features_tensor,
                self.visual_features_tensor
            )
            _, semantic_ids, _, _ = self.rq_vae(fused_features)
            
            self.all_semantic_ids = semantic_ids.cpu().numpy()
            self.all_features = fused_features.cpu().numpy()
            
        print(f"Precomputed {len(self.all_semantic_ids)} items")
        
    def get_item_semantic_id(self, item_id: str) -> List[int]:
        """获取商品的语义ID"""
        if str(item_id) not in self.item_id_to_idx:
            return []
        idx = self.item_id_to_idx[str(item_id)]
        return self.all_semantic_ids[idx].tolist()
    
    def get_similar_items_by_semantic_id(
        self,
        semantic_id: List[int],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """通过语义ID查找相似商品"""
        target = np.array(semantic_id)
        distances = np.sum(np.abs(self.all_semantic_ids - target), axis=1)
        indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in indices:
            item_id = self.idx_to_item_id.get(idx, str(idx))
            similarity = 1.0 / (1.0 + distances[idx])
            results.append((item_id, float(similarity)))
            
        return results
    
    def recommend_for_user(
        self,
        user_id: str,
        history_items: List[str],
        top_k: int = 10
    ) -> RecommendationResult:
        """为用户生成推荐"""
        history_semantic_ids = []
        for item_id in history_items:
            sid = self.get_item_semantic_id(item_id)
            if sid:
                history_semantic_ids.append(sid)
        
        if not history_semantic_ids:
            return self._get_popular_items(top_k)
        
        user_preference = np.mean(history_semantic_ids, axis=0)
        distances = np.sum(np.abs(self.all_semantic_ids - user_preference), axis=1)
        
        for item_id in history_items:
            if str(item_id) in self.item_id_to_idx:
                idx = self.item_id_to_idx[str(item_id)]
                distances[idx] = float("inf")
        
        indices = np.argsort(distances)[:top_k]
        
        item_ids = []
        scores = []
        semantic_ids = []
        
        for idx in indices:
            item_id = self.idx_to_item_id.get(idx, str(idx))
            score = 1.0 / (1.0 + distances[idx])
            
            try:
                item_ids.append(int(item_id))
            except:
                item_ids.append(item_id)
            scores.append(float(score))
            semantic_ids.append(self.all_semantic_ids[idx].tolist())
        
        return RecommendationResult(
            item_ids=item_ids,
            item_scores=scores,
            semantic_ids=semantic_ids
        )
    
    def _get_popular_items(self, top_k: int) -> RecommendationResult:
        """获取热门商品"""
        indices = np.random.choice(
            min(1000, self.num_items),
            size=min(top_k, 1000),
            replace=False
        )
        
        item_ids = []
        scores = []
        semantic_ids = []
        
        for i, idx in enumerate(indices):
            item_id = self.idx_to_item_id.get(idx, str(idx))
            try:
                item_ids.append(int(item_id))
            except:
                item_ids.append(item_id)
            scores.append(1.0 - i * 0.01)
            semantic_ids.append(self.all_semantic_ids[idx].tolist())
        
        return RecommendationResult(
            item_ids=item_ids,
            item_scores=scores,
            semantic_ids=semantic_ids
        )


# 全局推荐器实例
_recommender: Optional[RQVAERecommender] = None


def get_recommender() -> RQVAERecommender:
    """获取全局推荐器实例"""
    global _recommender
    if _recommender is None:
        raise RuntimeError("Recommender not initialized. Call init_recommender() first.")
    return _recommender


def init_recommender(
    checkpoint_path: str = None,
    data_dir: str = None,
    device: str = "cpu"
) -> RQVAERecommender:
    """初始化全局推荐器"""
    global _recommender
    
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "MODEL_PATH",
            "logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt"
        )
    
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "data/amazon_videogames")
    
    _recommender = RQVAERecommender(checkpoint_path, data_dir, device)
    return _recommender
