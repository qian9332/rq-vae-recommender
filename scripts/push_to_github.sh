#!/bin/bash
# 推送部署代码到GitHub
# 使用方法: ./push_to_github.sh

set -e

cd "$(dirname "$0")/.."

echo "=== RQ-VAE Recommender - 推送到GitHub ==="

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "有未提交的更改，请先提交"
    git status
    exit 1
fi

# 显示将要推送的提交
echo -e "\n=== 将要推送的提交 ==="
git log origin/feature/rq-vae-implementation..HEAD --oneline

# 推送
echo -e "\n=== 推送到GitHub ==="
git push origin feature/rq-vae-implementation

echo -e "\n=== 推送完成 ==="
echo "GitHub仓库: https://github.com/qian9332/rq-vae-recommender"
echo "分支: feature/rq-vae-implementation"
