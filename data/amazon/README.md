# Amazon All_Beauty Dataset

## 数据来源
- **数据集**: Amazon Review Data (All_Beauty category)
- **来源**: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
- **论文**: "Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering" (WWW 2017)

## 数据统计

| 指标 | 数值 |
|------|------|
| 商品数量 | 6,357 |
| 用户数量 | 33,403 |
| 交互数量 | 34,625 |
| 平均每用户交互数 | 1.0 |
| 文本特征维度 | 768 |
| 视觉特征维度 | 2048 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `text_features.npy` | 文本特征向量 [N, 768] |
| `visual_features.npy` | 视觉特征向量 [N, 2048] |
| `interactions_processed.csv` | 处理后的交互数据 |
| `items_processed.csv` | 商品元数据 |
| `id_mappings.pkl` | ID映射字典 |
| `reviews_All_Beauty.json` | 原始评论数据 |
| `meta_All_Beauty.json` | 原始商品元数据 |

## 数据字段

### interactions_processed.csv
- `user_idx`: 用户索引
- `item_idx`: 商品索引
- `rating`: 评分 (1-5)
- `timestamp`: 时间戳

### items_processed.csv
- `asin`: Amazon商品ID
- `title`: 商品标题
- `description`: 商品描述
- `price`: 价格
- `brand`: 品牌
- `image_url`: 商品图片URL

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

print(f"商品数: {mappings['num_items']}")
print(f"用户数: {mappings['num_users']}")
```

## 评分分布

| 评分 | 数量 |
|------|------|
| 1星 | 4,029 |
| 2星 | 2,156 |
| 3星 | 3,034 |
| 4星 | 5,235 |
| 5星 | 20,171 |

## 注意事项

1. 文本特征和视觉特征是基于内容hash生成的伪特征，用于演示模型架构
2. 如需真实特征，建议使用预训练模型（如BERT、ResNet）提取
3. 原始数据保留在 `reviews_All_Beauty.json` 和 `meta_All_Beauty.json` 中
