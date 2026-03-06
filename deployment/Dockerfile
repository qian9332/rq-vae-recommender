# RQ-VAE Recommender API Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# 复制应用代码
COPY models/ ./models/
COPY configs/ ./configs/
COPY utils/ ./utils/
COPY deployment/ ./deployment/
COPY __init__.py .

# 复制数据和模型
COPY data/amazon_videogames/ ./data/amazon_videogames/
COPY logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt ./models/best_model.pt

# 设置环境变量
ENV MODEL_PATH=/app/models/best_model.pt
ENV DATA_DIR=/app/data/amazon_videogames
ENV DEVICE=cpu
ENV PORT=8000

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "deployment.api"]
