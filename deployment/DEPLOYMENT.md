# RQ-VAE Recommender 部署指南

## 快速开始

### 方式一：直接运行

```bash
cd deployment
chmod +x start.sh
./start.sh
```

### 方式二：Docker部署

```bash
cd deployment
docker build -t rq-vae-recommender -f Dockerfile ..
docker run -d -p 8000:8000 --name rq-vae-api rq-vae-recommender
```

### 方式三：Docker Compose

```bash
cd deployment
docker-compose up -d
```

## API接口

服务启动后访问：http://localhost:8000

### 1. 健康检查

```bash
GET /health

Response:
{
    "status": "healthy",
    "model_loaded": true,
    "num_items": 16297,
    "num_users": 10537
}
```

### 2. 获取推荐

```bash
POST /recommend

Request:
{
    "user_id": "user123",
    "history_items": ["0", "1", "2"],
    "top_k": 10
}

Response:
{
    "item_ids": [123, 456, 789],
    "scores": [0.95, 0.89, 0.82],
    "semantic_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
}
```

### 3. 获取相似商品

```bash
POST /similar

Request:
{
    "item_id": "123",
    "top_k": 10
}

Response:
{
    "item_id": "123",
    "similar_items": [
        {"item_id": "456", "score": 0.95},
        {"item_id": "789", "score": 0.89}
    ]
}
```

### 4. 获取语义ID

```bash
POST /semantic-id

Request:
{
    "item_ids": ["123", "456"]
}

Response:
{
    "item_ids": ["123", "456"],
    "semantic_ids": [[1, 2, 3], [4, 5, 6]]
}
```

## 使用示例

### Python客户端

```python
import requests

BASE_URL = "http://localhost:8000"

# 获取推荐
response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "user_id": "user123",
        "history_items": ["0", "1", "2"],
        "top_k": 10
    }
)
print(response.json())
```

### cURL示例

```bash
# 健康检查
curl http://localhost:8000/health

# 获取推荐
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user123", "history_items": ["0", "1", "2"], "top_k": 10}'

# 获取相似商品
curl -X POST http://localhost:8000/similar \
    -H "Content-Type: application/json" \
    -d '{"item_id": "0", "top_k": 5}'
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | logs/rq_vae_pretraining_.../best_model.pt | 模型路径 |
| DATA_DIR | data/amazon_videogames | 数据目录 |
| DEVICE | cpu | 设备 (cpu/cuda) |
| PORT | 8000 | 服务端口 |

## 生产环境建议

1. 使用Gunicorn + Uvicorn
2. 使用Nginx反向代理
3. 启用HTTPS
4. 添加认证和限流
