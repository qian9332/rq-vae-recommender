# RQ-VAE Recommender

基于残差量化的多模态推荐系统，使用语义ID替代传统Hash ID解决冷启动语义孤岛问题。

## 🌟 核心特性

### 1. 残差量化VAE (RQ-VAE)
- **多层残差量化**：2-3层量化，每层码本大小256
- **语义ID序列**：将Item多模态特征编码为离散语义ID序列
- **解决冷启动**：新物品通过语义相似性获得相近ID，打破语义孤岛

### 2. EMA码本更新 + 死码重置
- **EMA更新**：decay=0.99，平滑更新码本向量
- **死码检测**：跟踪每个码本向量的使用频率
- **死码重置**：将长期未使用的码本向量重置为当前batch的活跃向量
- **利用率提升**：从约10%-20%提升至85%+

### 3. MMOE多模态编码器
- **多专家网络**：多个专家网络学习通用特征变换
- **任务特定门控**：文本和视觉模态独立门控，实现协同与解耦
- **跨模态融合**：自适应模态权重融合

### 4. 行为感知微调
- **软索引**：Gumbel-Softmax实现可微分量化
- **STE梯度回传**：直通估计器允许推荐任务梯度回传至RQ-VAE
- **联合优化**：同时优化重建损失与推荐损失

## 📁 项目结构

```
rq-vae-recommender/
├── configs/
│   └── config.py          # 配置文件
├── models/
│   ├── __init__.py
│   ├── ema_codebook.py    # EMA码本模块
│   ├── rq_vae.py          # 残差量化VAE
│   ├── mmoe_encoder.py    # MMOE多模态编码器
│   ├── recommender.py     # 推荐模型
│   └── behavior_aware_finetuning.py  # 行为感知微调
├── data/
│   ├── __init__.py
│   └── dataset.py         # 数据处理
├── training/
│   ├── train.py           # 训练脚本
│   └── train_with_logging.py  # 带日志的训练脚本
├── deployment/            # 部署服务
│   ├── inference.py       # 推理服务
│   ├── api.py             # FastAPI接口
│   ├── Dockerfile         # Docker配置
│   ├── docker-compose.yml
│   ├── start.sh           # 启动脚本
│   └── DEPLOYMENT.md      # 部署文档
├── logs/                  # 训练日志和检查点
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # 评估指标
│   └── common.py          # 工具函数
├── scripts/
│   └── run.sh             # 运行脚本
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

数据格式要求：
- `text_features.npy`: 文本特征 [N, 768]
- `visual_features.npy`: 视觉特征 [N, 2048]
- `interactions.csv`: 交互数据 (user_id, item_id, timestamp)

或使用合成数据进行测试：

```python
from data.dataset import generate_synthetic_data, save_processed_data

data = generate_synthetic_data(
    num_items=10000,
    num_users=1000,
    text_dim=768,
    visual_dim=2048
)
save_processed_data(data, './data')
```

### 训练模型

#### 1. RQ-VAE预训练

```bash
python -m training.train --mode pretrain --data_dir ./data --num_epochs 50
```

#### 2. 推荐模型微调

```bash
python -m training.train --mode finetune --data_dir ./data --num_epochs 30
```

#### 3. 完整训练流程

```bash
python -m training.train --mode all --data_dir ./data
```

## 🐳 部署服务

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

### API接口

服务启动后访问：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

#### 获取推荐

```bash
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user123", "history_items": ["0", "1", "2"], "top_k": 10}'
```

#### 获取相似商品

```bash
curl -X POST http://localhost:8000/similar \
    -H "Content-Type: application/json" \
    -d '{"item_id": "0", "top_k": 5}'
```

详细部署说明请查看 [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)

## 📊 训练结果

### Amazon Video_Games 数据集

| 指标 | 数值 |
|------|------|
| 商品数 | 16,297 |
| 用户数 | 10,537 |
| 交互数 | 110,079 |
| 训练损失 | 0.6341 → 0.0128 |
| 验证损失 | 0.0011 |
| 码本使用率 | 94-98% |

### 代码示例

```python
import torch
from models import RQVAE, MultiModalFeatureEncoder, SemanticRecommender

# 1. 创建MMOE编码器
mmoe_encoder = MultiModalFeatureEncoder(
    text_input_dim=768,
    visual_input_dim=2048,
    hidden_dim=256,
    output_dim=128,
    num_experts=4
)

# 2. 创建RQ-VAE
rq_vae = RQVAE(
    input_dim=128,
    hidden_dim=256,
    embedding_dim=128,
    num_quantization_layers=3,
    codebook_size=256,
    ema_decay=0.99
)

# 3. 编码物品特征
text_features = torch.randn(100, 768)
visual_features = torch.randn(100, 2048)

fused_features, _, _, _ = mmoe_encoder(text_features, visual_features)
semantic_ids = rq_vae.encode_to_ids(fused_features)

print(f"Semantic IDs shape: {semantic_ids.shape}")  # [100, 3]
print(f"Example semantic ID: {semantic_ids[0]}")    # e.g., [45, 128, 201]

# 4. 创建推荐模型
recommender = SemanticRecommender(
    num_users=1000,
    num_items=10000,
    num_quantization_layers=3,
    codebook_size=256,
    embedding_dim=128
)

# 5. 推理
user_ids = torch.tensor([0, 1, 2])
history_ids = torch.randint(0, 256, (3, 50, 3))
history_mask = torch.ones(3, 50)
candidate_ids = torch.randint(0, 256, (3, 10, 3))

scores, info = recommender(
    user_ids, history_ids, history_mask, candidate_ids
)
print(f"Prediction scores: {scores.shape}")  # [3, 10]
```

## 📊 核心算法

### 残差量化

```
输入: z ∈ R^{B×D}
输出: 语义ID序列 [id_1, id_2, id_3]

第1层: r_1 = z,           id_1 = argmin_k ||r_1 - e_k^1||
第2层: r_2 = r_1 - e_{id_1}^1,  id_2 = argmin_k ||r_2 - e_k^2||
第3层: r_3 = r_2 - e_{id_2}^2,  id_3 = argmin_k ||r_3 - e_k^3||

重建: z_q = e_{id_1}^1 + e_{id_2}^2 + e_{id_3}^3
```

### EMA码本更新

```
N_t = γ * N_{t-1} + (1-γ) * n_t          # 使用次数
m_t = γ * m_{t-1} + (1-γ) * Σ z_i        # 向量和
e = m_t / N_t                            # 码本向量

死码重置:
if steps_unused > threshold:
    e_dead = z_sample + noise            # 用当前样本重置
```

### 软索引 + STE

```
软索引: p = Gumbel-Softmax(-||z - e||, τ)
软量化: z_q = Σ p_k * e_k

STE: 前向 z_q_hard, 反向 ∂L/∂z_q_soft
```

## 📈 评估指标

支持以下推荐系统评估指标：

| 指标 | 说明 |
|------|------|
| Recall@K | 召回率 |
| Precision@K | 准确率 |
| NDCG@K | 归一化折损累积增益 |
| Hit Rate@K | 命中率 |
| MRR | 平均倒数排名 |
| AUC | ROC曲线下面积 |
| Coverage | 覆盖率 |
| Diversity | 多样性 |

## 🔧 配置说明

```yaml
# config.yaml
codebook:
  num_layers: 3              # 残差量化层数
  codebook_size: 256         # 码本大小
  embedding_dim: 128         # 嵌入维度
  ema_decay: 0.99            # EMA衰减系数
  dead_code_threshold: 100   # 死码判定阈值

mmoe:
  text_input_dim: 768        # 文本特征维度
  visual_input_dim: 2048     # 视觉特征维度
  hidden_dim: 256            # 隐藏层维度
  num_experts: 4             # 专家数量

recommender:
  num_users: 10000           # 用户数量
  num_items: 50000           # 物品数量
  max_seq_length: 50         # 最大序列长度

training:
  batch_size: 256
  learning_rate: 0.0001
  pretrain_epochs: 50
  finetune_epochs: 30
```

## 📚 参考文献

1. **RQ-VAE**: Lee et al., "Autoregressive Image Generation using Residual Quantization", CVPR 2022
2. **MMOE**: Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD 2018
3. **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
4. **Semantic ID**: Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！
