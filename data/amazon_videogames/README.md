# Amazon Video_Games Dataset

## 数据来源
- **数据集**: Amazon Review Data (Video_Games category)
- **来源**: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
- **论文**: "Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering" (WWW 2017)

## 数据统计

| 指标 | 数值 |
|------|------|
| 商品数量 | 16,297 |
| 用户数量 | 10,537 |
| 交互数量 | 110,079 |
| 平均每用户交互数 | 10.45 |
| 用户最少交互数 | 5 |
| 用户最多交互数 | 745 |
| 数据稀疏度 | 99.94% |
| 文本特征维度 | 768 |
| 视觉特征维度 | 2048 |

## 文件说明

| 文件 | 说明 | 大小 |
|------|------|------|
| `text_features.npy` | 文本特征向量 [N, 768] | ~50MB |
| `visual_features.npy` | 视觉特征向量 [N, 2048] | ~130MB |
| `interactions_processed.csv` | 处理后的交互数据 | ~3MB |
| `items_metadata.csv` | 商品元数据 | ~5MB |
| `id_mappings.pkl` | ID映射字典 | ~1MB |

## 数据字段

### interactions_processed.csv
- `user_idx`: 用户索引 (0-10536)
- `item_idx`: 商品索引 (0-16296)
- `rating`: 评分 (1-5)
- `timestamp`: Unix时间戳
- `user_id`: 原始用户ID
- `item_id`: 原始商品ID (asin)

## 使用方法

```python
import numpy as np
import pandas as pd
import pickle

# 加载特征
text_features = np.load('text_features.npy')
visual_features = np.load('visual_features.npy')

# 加载交互数据
interactions = pd.read_csv('interactions_processed.csv')

# 加载ID映射
with open('id_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

print(f"商品数: {mappings['num_items']}")  # 16297
print(f"用户数: {mappings['num_users']}")  # 10537
```

## 数据处理流程

1. 下载原始数据 (reviews_Video_Games.json, meta_Video_Games.json)
2. 过滤用户交互数 >= 5 的用户
3. 创建用户/商品ID映射
4. 生成文本特征 (基于标题+描述)
5. 生成视觉特征 (基于图像URL)
6. L2归一化特征向量

## 注意事项

1. **特征说明**: 当前特征是基于内容hash生成的伪特征，用于演示模型架构
2. **真实特征**: 建议使用预训练模型提取:
   - 文本: BERT, RoBERTa
   - 图像: ResNet, ViT
3. **数据已排序**: 交互数据按用户和时间戳排序，可直接构建序列

## 与All_Beauty数据集对比

| 指标 | All_Beauty | Video_Games |
|------|------------|-------------|
| 商品数 | 6,357 | 16,297 |
| 用户数 | 33,403 | 10,537 |
| 交互数 | 34,625 | 110,079 |
| 平均交互/用户 | 1.04 | 10.45 |
| 可构建序列 | ✗ | ✓ |
| 适合训练 | ✗ | ✓ |
