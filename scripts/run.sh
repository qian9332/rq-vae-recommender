#!/bin/bash

# RQ-VAE Recommender Training Script

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
DATA_DIR="./data"
LOG_DIR="./logs"
MODE="all"  # pretrain, finetune, all

# Create directories
mkdir -p $DATA_DIR
mkdir -p $LOG_DIR

# Generate synthetic data if not exists
if [ ! -f "$DATA_DIR/text_features.npy" ]; then
    echo "Generating synthetic data..."
    python -c "
from data.dataset import generate_synthetic_data, save_processed_data
data = generate_synthetic_data(num_items=10000, num_users=1000)
save_processed_data(data, '$DATA_DIR')
print('Data generated successfully!')
"
fi

# Run training
echo "Starting training with mode: $MODE"
python -m training.train \
    --mode $MODE \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --num_epochs 50 \
    --batch_size 256 \
    --lr 0.0001 \
    --device cuda \
    --seed 42

echo "Training completed!"
