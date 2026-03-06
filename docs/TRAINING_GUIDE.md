# RQ-VAE Recommender 训练指南

## 目录
1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [RQ-VAE预训练](#3-rq-vae预训练)
4. [推荐模型微调](#4-推荐模型微调)
5. [评估与测试](#5-评估与测试)
6. [常见问题](#6-常见问题)

---

## 1. 环境配置

### 1.1 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU训练)
- 内存: 16GB+
- 磁盘: 10GB+

### 1.2 安装依赖

```bash
# 克隆项目
git clone https://github.com/qian9332/rq-vae-recommender.git
cd rq-vae-recommender

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 1.3 验证安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

---

## 2. 数据准备

### 2.1 数据格式

项目需要以下数据文件：

```
data/
└── your_dataset/
    ├── text_features.npy      # 文本特征 [N, 768]
    ├── visual_features.npy    # 视觉特征 [N, 2048]
    ├── interactions.csv       # 交互数据
    └── id_mappings.pkl        # ID映射
```

### 2.2 交互数据格式

`interactions.csv` 应包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| user_id | int | 用户ID |
| item_id | int | 商品ID |
| timestamp | int | 时间戳（可选） |
| rating | float | 评分（可选） |

### 2.3 特征提取

#### 文本特征提取

```python
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# 加载BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_features(texts, batch_size=32):
    """提取文本特征"""
    features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS]向量作为文本特征
            batch_features = outputs.last_hidden_state[:, 0, :].numpy()
        features.append(batch_features)
    return np.vstack(features)

# 使用示例
texts = ["商品描述1", "商品描述2", ...]
text_features = extract_text_features(texts)
np.save('text_features.npy', text_features.astype(np.float16))
```

#### 视觉特征提取

```python
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

# 加载ResNet模型
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后分类层
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def extract_visual_features(image_paths, batch_size=32):
    """提取视觉特征"""
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images)
        with torch.no_grad():
            batch_features = model(batch_tensor).squeeze(-1).squeeze(-1).numpy()
        features.append(batch_features)
    return np.vstack(features)

# 使用示例
image_paths = ["image1.jpg", "image2.jpg", ...]
visual_features = extract_visual_features(image_paths)
np.save('visual_features.npy', visual_features.astype(np.float16))
```

### 2.4 使用Amazon数据集

项目已包含Amazon Video_Games数据集：

```bash
# 数据位置
data/amazon_videogames/

# 数据统计
- 商品数: 16,297
- 用户数: 10,537
- 交互数: 110,079
```

---

## 3. RQ-VAE预训练

### 3.1 预训练目的

RQ-VAE预训练的目标是学习将商品的多模态特征编码为离散语义ID序列。

### 3.2 运行预训练

```bash
# 基础训练
python training/train.py --mode pretrain \
    --data_dir data/amazon_videogames \
    --num_epochs 50

# 带详细日志的训练
python training/train_with_logging.py \
    --data_dir data/amazon_videogames \
    --num_epochs 50 \
    --batch_size 256 \
    --learning_rate 1e-3
```

### 3.3 预训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_epochs | 50 | 训练轮数 |
| batch_size | 256 | 批次大小 |
| learning_rate | 1e-3 | 学习率 |
| num_quantization_layers | 3 | 量化层数 |
| codebook_size | 256 | 码本大小 |
| embedding_dim | 128 | 嵌入维度 |

### 3.4 预训练结果

```
logs/rq_vae_pretraining_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pt          # 最佳模型
│   └── checkpoint_epoch_*.pt  # 周期检查点
├── tensorboard/               # TensorBoard日志
├── config.json               # 配置文件
├── training_history.json     # 训练历史
└── *.log                     # 训练日志
```

### 3.5 监控训练

```bash
# 使用TensorBoard监控
tensorboard --logdir logs/rq_vae_pretraining_YYYYMMDD_HHMMSS/tensorboard

# 浏览器访问
http://localhost:6006
```

---

## 4. 推荐模型微调

### 4.1 微调目的

微调阶段将RQ-VAE与推荐任务联合训练，优化推荐效果。

### 4.2 运行微调

```bash
# 微调训练
python training/finetune.py \
    --data_dir data/amazon_videogames \
    --pretrained_model logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt \
    --num_epochs 20 \
    --batch_size 128 \
    --learning_rate 1e-4

# 快速测试
python training/quick_finetune.py
```

### 4.3 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_epochs | 20 | 训练轮数 |
| batch_size | 128 | 批次大小 |
| learning_rate | 1e-4 | 学习率 |
| reconstruction_weight | 0.5 | 重建损失权重 |
| recommendation_weight | 1.0 | 推荐损失权重 |
| max_seq_length | 50 | 最大序列长度 |

### 4.4 微调策略

#### 端到端微调

```python
# 联合优化RQ-VAE和推荐模型
loss = recommendation_weight * L_rec + reconstruction_weight * L_vq
```

#### 两阶段微调

```python
# 阶段1: 冻结RQ-VAE，训练推荐模型
for param in rq_vae.parameters():
    param.requires_grad = False

# 阶段2: 解冻RQ-VAE，联合微调
for param in rq_vae.parameters():
    param.requires_grad = True
```

---

## 5. 评估与测试

### 5.1 评估指标

| 指标 | 说明 |
|------|------|
| Recall@K | 召回率 |
| Precision@K | 准确率 |
| NDCG@K | 归一化折损累积增益 |
| Hit Rate@K | 命中率 |
| MRR | 平均倒数排名 |
| AUC | ROC曲线下面积 |

### 5.2 运行评估

```python
from utils.metrics import MetricsCalculator

# 创建评估器
metrics = MetricsCalculator(['recall@10', 'ndcg@10', 'hit_rate@10', 'mrr'])

# 计算指标
results = metrics.compute(scores, labels)
print(results)
```

### 5.3 离线评估

```bash
# 评估预训练模型
python training/evaluate.py \
    --model_path logs/rq_vae_pretraining_20260304_104338/checkpoints/best_model.pt \
    --data_dir data/amazon_videogames
```

---

## 6. 常见问题

### Q1: CUDA内存不足

```bash
# 减小batch_size
python training/train.py --batch_size 128

# 使用混合精度训练
python training/train.py --fp16
```

### Q2: 码本使用率低

```bash
# 增加训练轮数
python training/train.py --num_epochs 100

# 调整死码重置阈值
# 在config.py中修改 dead_code_threshold
```

### Q3: 训练不收敛

```bash
# 降低学习率
python training/train.py --learning_rate 1e-4

# 使用学习率预热
python training/train.py --warmup_steps 1000
```

### Q4: 如何使用GPU训练

```bash
# 指定GPU
CUDA_VISIBLE_DEVICES=0 python training/train.py

# 多GPU训练
python training/train.py --gpus 0,1,2,3
```

### Q5: 如何加载预训练模型

```python
import torch
from models.rq_vae import RQVAE

# 加载模型
checkpoint = torch.load('best_model.pt')
model = RQVAE(...)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 附录

### A. 完整训练流程

```bash
# 1. 准备数据
python scripts/prepare_data.py --dataset amazon_videogames

# 2. RQ-VAE预训练
python training/train.py --mode pretrain --num_epochs 50

# 3. 推荐模型微调
python training/finetune.py --num_epochs 20

# 4. 评估
python training/evaluate.py

# 5. 部署
cd deployment && docker-compose up -d
```

### B. 训练时间估计

| 阶段 | 数据规模 | GPU | 时间 |
|------|---------|-----|------|
| 预训练 | 16K商品 | RTX 3090 | ~30分钟 |
| 微调 | 110K交互 | RTX 3090 | ~1小时 |
| 评估 | - | RTX 3090 | ~5分钟 |

### C. 模型大小

| 组件 | 参数量 | 大小 |
|------|--------|------|
| MMOE编码器 | 1.5M | 5.9 MB |
| RQ-VAE | 0.2M | 0.9 MB |
| 推荐模型 | 2.1M | 8.2 MB |
| **总计** | **3.8M** | **15 MB** |

---

*最后更新: 2026-03-06*
