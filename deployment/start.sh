#!/bin/bash
# RQ-VAE Recommender 启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== RQ-VAE Recommender 启动脚本 ===${NC}"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}Checking dependencies...${NC}"
pip3 install -q fastapi uvicorn 2>/dev/null || true

# 设置环境变量
export MODEL_PATH=${MODEL_PATH:-"logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt"}
export DATA_DIR=${DATA_DIR:-"data/amazon_videogames"}
export DEVICE=${DEVICE:-"cpu"}
export PORT=${PORT:-8000}

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_DIR"
echo "  Device: $DEVICE"
echo "  Port: $PORT"

# 启动服务
echo -e "${GREEN}Starting API server...${NC}"
cd "$(dirname "$0")"
python3 api.py
