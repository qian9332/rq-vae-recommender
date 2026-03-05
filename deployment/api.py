"""
RQ-VAE Recommender API Service
FastAPI接口服务
"""

import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import init_recommender, get_recommender


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 请求/响应模型
class RecommendRequest(BaseModel):
    """推荐请求"""
    user_id: str = Field(..., description="用户ID")
    history_items: List[str] = Field(default=[], description="用户历史交互商品ID列表")
    top_k: int = Field(default=10, ge=1, le=100, description="返回推荐数量")


class SimilarItemsRequest(BaseModel):
    """相似商品请求"""
    item_id: str = Field(..., description="商品ID")
    top_k: int = Field(default=10, ge=1, le=100, description="返回相似商品数量")


class SemanticIDRequest(BaseModel):
    """语义ID请求"""
    item_ids: List[str] = Field(..., description="商品ID列表")


class RecommendationResponse(BaseModel):
    """推荐响应"""
    item_ids: List[int] = Field(..., description="推荐商品ID列表")
    scores: List[float] = Field(..., description="推荐分数列表")
    semantic_ids: List[List[int]] = Field(..., description="语义ID列表")


class SimilarItemsResponse(BaseModel):
    """相似商品响应"""
    item_id: str = Field(..., description="查询商品ID")
    similar_items: List[dict] = Field(..., description="相似商品列表")


class SemanticIDResponse(BaseModel):
    """语义ID响应"""
    item_ids: List[str] = Field(..., description="商品ID列表")
    semantic_ids: List[List[int]] = Field(..., description="语义ID列表")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    num_items: int
    num_users: int


# 应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("Loading model...")
    try:
        recommender = init_recommender(
            checkpoint_path=os.environ.get(
                "MODEL_PATH",
                "logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt"
            ),
            data_dir=os.environ.get("DATA_DIR", "data/amazon_videogames"),
            device=os.environ.get("DEVICE", "cpu")
        )
        logger.info(f"Model loaded successfully. Items: {recommender.num_items}, Users: {recommender.num_users}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down...")


# 创建FastAPI应用
app = FastAPI(
    title="RQ-VAE Recommender API",
    description="基于RQ-VAE的多模态推荐系统API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API端点
@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "RQ-VAE Recommender API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    try:
        recommender = get_recommender()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            num_items=recommender.num_items,
            num_users=recommender.num_users
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            num_items=0,
            num_users=0
        )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
async def recommend(request: RecommendRequest):
    """
    为用户生成推荐
    
    - **user_id**: 用户ID
    - **history_items**: 用户历史交互商品ID列表
    - **top_k**: 返回推荐数量
    """
    try:
        recommender = get_recommender()
        result = recommender.recommend_for_user(
            user_id=request.user_id,
            history_items=request.history_items,
            top_k=request.top_k
        )
        
        return RecommendationResponse(
            item_ids=result.item_ids,
            scores=result.item_scores,
            semantic_ids=result.semantic_ids
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarItemsResponse, tags=["Recommendation"])
async def get_similar_items(request: SimilarItemsRequest):
    """
    获取相似商品
    
    - **item_id**: 商品ID
    - **top_k**: 返回相似商品数量
    """
    try:
        recommender = get_recommender()
        semantic_id = recommender.get_item_semantic_id(request.item_id)
        
        if not semantic_id:
            raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")
        
        similar = recommender.get_similar_items_by_semantic_id(
            semantic_id=semantic_id,
            top_k=request.top_k
        )
        
        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=[
                {"item_id": item_id, "score": score}
                for item_id, score in similar
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar items query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic-id", response_model=SemanticIDResponse, tags=["Item"])
async def get_semantic_ids(request: SemanticIDRequest):
    """
    获取商品的语义ID
    
    - **item_ids**: 商品ID列表
    """
    try:
        recommender = get_recommender()
        semantic_ids = []
        
        for item_id in request.item_ids:
            sid = recommender.get_item_semantic_id(item_id)
            semantic_ids.append(sid if sid else [])
        
        return SemanticIDResponse(
            item_ids=request.item_ids,
            semantic_ids=semantic_ids
        )
    except Exception as e:
        logger.error(f"Semantic ID query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/item/{item_id}/semantic-id", tags=["Item"])
async def get_item_semantic_id(item_id: str):
    """
    获取单个商品的语义ID
    """
    try:
        recommender = get_recommender()
        semantic_id = recommender.get_item_semantic_id(item_id)
        
        if not semantic_id:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        return {
            "item_id": item_id,
            "semantic_id": semantic_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic ID query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# 运行入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
