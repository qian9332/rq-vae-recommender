# RQ-VAE Recommender 严格面试题大全

> 面试官须知：本面试题覆盖项目全部技术细节，请根据候选人回答深度评分

---

## 目录

1. [项目背景与动机](#一项目背景与动机) - 5题
2. [核心算法原理](#二核心算法原理) - 8题
3. [模型架构设计](#三模型架构设计) - 7题
4. [训练策略与优化](#四训练策略与优化) - 6题
5. [评估与指标](#五评估与指标) - 4题
6. [工程实现](#六工程实现) - 5题
7. [代码实现](#七代码实现) - 6题
8. [系统设计](#八系统设计) - 4题
9. [开放性问题](#九开放性问题) - 5题

**总计：50道面试题**

---

## 一、项目背景与动机

### Q1: 请详细说明这个项目要解决的核心问题是什么？为什么传统方法无法解决？

**答案：**

**核心问题：推荐系统的冷启动语义孤岛问题**

**问题详解：**

1. **传统ID嵌入的问题**
   ```
   传统方法：商品ID → 随机初始化嵌入 → 训练学习
   
   问题：
   - 新商品没有ID嵌入（冷启动）
   - 即使新商品与已有商品语义相似，ID也完全不相关
   - ID嵌入无法迁移到新商品
   ```

2. **语义孤岛现象**
   ```
   商品A（iPhone 15）: ID=1001, embedding=[0.1, 0.2, ...]
   商品B（iPhone 16）: ID=1002, embedding=[0.8, 0.3, ...]  # 新商品，随机初始化
   
   问题：A和B语义相似，但嵌入完全不相关
   ```

3. **传统方法无法解决的原因**
   - 协同过滤：需要历史交互，新商品无数据
   - 矩阵分解：同样需要交互数据
   - 传统深度学习：ID嵌入依赖训练，无法泛化

**本项目的解决方案：**
- 使用语义ID替代随机ID
- 语义ID基于商品内容特征生成
- 相似商品获得相似的语义ID
- 新商品只要有特征就能获得合理的语义ID

---

### Q2: 为什么选择RQ-VAE而不是其他量化方法？请对比分析。

**答案：**

**量化方法对比分析：**

| 方法 | 原理 | 优点 | 缺点 | 适用性 |
|------|------|------|------|--------|
| **VQ-VAE** | 单层向量量化 | 简单、稳定 | 表示能力有限、码本坍塌 | ⚠️ 一般 |
| **RQ-VAE** | 多层残差量化 | 层次化表示、表示能力强 | 训练复杂 | ✅ 最佳 |
| **PQ** | 乘积量化 | 高效检索 | 需要预定义结构 | ⚠️ 检索场景 |
| **LSH** | 局部敏感哈希 | 快速 | 精度低、无学习 | ❌ 不适合 |
| **OPQ** | 优化乘积量化 | 旋转优化 | 复杂度高 | ⚠️ 检索场景 |

**选择RQ-VAE的核心原因：**

1. **层次化语义表示**
   ```
   Layer 1: 捕获粗粒度语义（如：电子产品）
   Layer 2: 捕获中粒度语义（如：手机）
   Layer 3: 捕获细粒度语义（如：iPhone 15）
   ```

2. **表示能力**
   ```
   单层VQ: 256个唯一ID
   三层RQ: 256³ = 16,777,216个唯一ID
   ```

3. **残差学习优势**
   - 每层专注于前层未捕获的信息
   - 信息利用更充分
   - 重建质量更高

4. **冷启动友好**
   - 新商品通过特征直接编码
   - 无需额外训练

---

### Q3: 这个项目的创新点有哪些？请从学术和工程两个角度分析。

**答案：**

**学术创新点：**

1. **语义ID生成方法**
   - 创新点：将推荐系统的ID嵌入问题转化为特征量化问题
   - 学术价值：连接了表示学习和推荐系统两个领域

2. **多模态融合策略**
   - 创新点：MMOE架构用于商品多模态特征融合
   - 学术价值：多模态学习在推荐中的应用

3. **端到端训练框架**
   - 创新点：STE实现量化操作的梯度回传
   - 学术价值：解决离散优化问题

**工程创新点：**

1. **EMA码本更新 + 死码重置**
   - 解决码本坍塌问题
   - 码本使用率从10%提升到94%+

2. **两阶段训练策略**
   - 预训练：学习通用语义表示
   - 微调：优化推荐任务

3. **完整的工程实现**
   - 数据处理、训练、评估、部署全流程
   - 可复现、可扩展

---

### Q4: 请分析这个项目的数据集特点，以及数据稀疏性对模型的影响。

**答案：**

**数据集特点：**

```
Amazon Video_Games数据集：
├── 用户数: 10,537
├── 商品数: 16,297
├── 交互数: 110,079
├── 稀疏度: 99.94%
├── 平均用户交互: 10.45
├── 平均商品交互: 6.75
├── 用户交互中位数: 7
└── 商品交互中位数: 2
```

**数据分布特点：**

1. **长尾分布**
   ```
   头部商品：少数商品有大量交互
   尾部商品：大量商品交互稀疏
   商品交互中位数仅2次
   ```

2. **冷启动问题严重**
   - 大量商品交互稀疏
   - 新商品无法获得有效ID嵌入

**数据稀疏性的影响：**

1. **对传统方法的影响**
   - 协同过滤：无法学习有效表示
   - 矩阵分解：过拟合严重
   - ID嵌入：泛化能力差

2. **对本项目的影响**
   - 语义ID不依赖交互数据
   - 可以处理零交互商品
   - 但用户建模仍受影响

**解决方案：**
- 语义ID解决商品冷启动
- 用户序列建模利用有限交互
- 负采样策略优化

---

### Q5: 请说明这个项目在实际业务中的应用价值和局限性。

**答案：**

**应用价值：**

1. **冷启动场景**
   ```
   场景：电商平台新商品上架
   传统方法：无法推荐
   本项目：立即获得语义ID，可被推荐
   ```

2. **多模态商品**
   ```
   场景：有图片和描述的商品
   优势：充分利用多模态信息
   ```

3. **语义检索**
   ```
   场景：相似商品推荐
   优势：语义ID可直接计算相似度
   ```

**局限性：**

1. **特征依赖**
   - 需要高质量的多模态特征
   - 特征质量直接影响效果

2. **用户建模不足**
   - 用户冷启动问题未完全解决
   - 需要一定的交互历史

3. **计算开销**
   - 多层量化增加计算量
   - 大规模部署需要优化

4. **领域迁移**
   - 跨领域迁移需要重新预训练
   - 特征提取器需要适配

---

## 二、核心算法原理

### Q6: 请详细解释残差量化(Residual Quantization)的数学原理。

**答案：**

**残差量化数学定义：**

给定输入向量 $z \in \mathbb{R}^D$，残差量化通过多层量化逐步逼近：

$$
\begin{aligned}
r_1 &= z \\
q_1 &= \arg\min_{e \in \mathcal{C}_1} \|r_1 - e\|^2 \\
r_2 &= r_1 - q_1 \\
q_2 &= \arg\min_{e \in \mathcal{C}_2} \|r_2 - e\|^2 \\
r_3 &= r_2 - q_2 \\
q_3 &= \arg\min_{e \in \mathcal{C}_3} \|r_3 - e\|^2
\end{aligned}
$$

**重建公式：**

$$z_q = q_1 + q_2 + q_3$$

**损失函数：**

$$\mathcal{L} = \|z - z_q\|^2 + \beta \sum_{i=1}^{3} \|r_i - \text{sg}(q_i)\|^2$$

其中：
- $\mathcal{C}_i$ 是第 $i$ 层码本
- $\text{sg}(\cdot)$ 是stop-gradient操作
- $\beta$ 是commitment cost

**为什么残差量化有效：**

1. **信息论角度**
   - 每层量化独立的残差信息
   - 类似于渐进编码

2. **优化角度**
   - 残差通常比原向量小
   - 更容易量化

3. **表示能力**
   - 多层组合表示能力强
   - $K^L$ 个唯一表示

---

### Q7: 请推导EMA码本更新的公式，并解释为什么EMA比梯度下降更稳定。

**答案：**

**EMA更新公式推导：**

假设码本向量 $e_k$ 应该是其对应输入向量的平均值：

$$e_k = \frac{\sum_{i: z_i \rightarrow k} z_i}{n_k}$$

使用EMA近似：

$$
\begin{aligned}
N_k^{(t)} &= \gamma N_k^{(t-1)} + (1-\gamma) n_k^{(t)} \\
M_k^{(t)} &= \gamma M_k^{(t-1)} + (1-\gamma) \sum_{i: z_i \rightarrow k} z_i \\
e_k^{(t)} &= \frac{M_k^{(t)}}{N_k^{(t)}}
\end{aligned}
$$

其中：
- $N_k$：累计使用次数
- $M_k$：累计向量和
- $\gamma$：衰减系数（通常0.99）
- $n_k^{(t)}$：当前batch使用次数

**EMA vs 梯度下降对比：**

| 方面 | EMA | 梯度下降 |
|------|-----|----------|
| 更新公式 | 滑动平均 | $\Delta e = -\eta \nabla L$ |
| 学习率 | 自动适应 | 需要手动调节 |
| 稳定性 | 高（平滑） | 低（可能震荡） |
| 收敛速度 | 较慢但稳定 | 可能快但不稳定 |
| 超参数 | 仅$\gamma$ | 学习率、动量等 |

**为什么EMA更稳定：**

1. **平滑更新**
   ```
   EMA: e_new = 0.99 * e_old + 0.01 * e_target
   梯度: e_new = e_old - lr * gradient
   ```
   EMA天然具有平滑效果，不会剧烈变化

2. **自适应学习率**
   - 使用频率高的码本更新快
   - 使用频率低的码本更新慢
   - 自动平衡

3. **无梯度依赖**
   - 不需要通过反向传播
   - 避免梯度消失/爆炸

---

### Q8: 什么是码本坍塌(Codebook Collapse)？请详细说明死码重置策略。

**答案：**

**码本坍塌问题：**

```
现象：大部分码本向量不被使用
原因：
1. 初始化不均匀
2. 训练早期部分码本被频繁使用
3. 富者愈富效应

结果：
- 码本利用率低（10-20%）
- 表示能力浪费
- 重建质量下降
```

**死码检测：**

```python
# 跟踪每个码的使用情况
steps_unused[code_id] += 1  # 每步更新
steps_unused[used_codes] = 0  # 使用的码重置

# 检测死码
dead_codes = steps_unused > threshold  # 通常100步
```

**死码重置策略：**

```python
def reset_dead_codes(self, z):
    """重置死码"""
    dead_mask = self.steps_unused > self.threshold
    
    if dead_mask.any():
        # 策略1：用当前batch样本重置
        sample_idx = torch.randperm(z.size(0))[:dead_mask.sum()]
        new_codes = z[sample_idx]
        
        # 添加小噪声避免重复
        noise = torch.randn_like(new_codes) * 0.01
        self.embedding.data[dead_mask] = new_codes + noise
        
        # 重置计数器
        self.steps_unused[dead_mask] = 0
        self.cluster_size[dead_mask] = 1
```

**重置策略对比：**

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 随机重置 | 随机初始化 | 简单 | 可能再次死码 |
| 样本重置 | 用当前样本 | 保证被使用 | 可能重复 |
| 活跃码复制 | 复制活跃码 | 稳定 | 多样性降低 |
| **混合策略** | 样本+噪声 | 平衡 | 需调参 |

**效果：**
- 码本使用率从10-20%提升到94-98%
- 表示能力充分利用

---

### Q9: 请解释Commitment Loss的作用，并推导其梯度。

**答案：**

**Commitment Loss定义：**

$$\mathcal{L}_{commit} = \|z - \text{sg}(e)\|^2$$

其中：
- $z$：编码器输出
- $e$：选中的码本向量
- $\text{sg}(\cdot)$：stop-gradient操作

**作用分析：**

1. **约束编码器输出**
   ```
   没有commitment loss:
   编码器可以输出任意值，只要量化后正确即可
   
   有commitment loss:
   编码器输出必须靠近码本向量
   ```

2. **稳定训练**
   - 防止编码器输出漂移
   - 保持编码器和码本的协调

3. **平衡重建和约束**
   ```
   总损失 = 重建损失 + β * commitment损失
   β通常取0.25
   ```

**梯度推导：**

$$
\frac{\partial \mathcal{L}_{commit}}{\partial z} = \frac{\partial}{\partial z} \|z - \text{sg}(e)\|^2 = 2(z - e)
$$

注意：
- 梯度只回传给 $z$（编码器）
- $e$ 通过EMA更新，不需要梯度
- stop_gradient 阻止梯度流向 $e$

**为什么需要stop_gradient：**

```python
# 如果不用stop_gradient
loss = ||z - e||^2
# 梯度会同时流向z和e
# 但e应该通过EMA更新，不应该有梯度

# 使用stop_gradient
loss = ||z - sg(e)||^2
# 梯度只流向z
# e通过EMA独立更新
```

---

### Q10: 请解释直通估计器(STE)的原理，以及在量化操作中如何应用。

**答案：**

**问题背景：**

量化操作 $\arg\min$ 不可微：

$$q = e_{\arg\min_k \|z - e_k\|}$$

梯度无法通过 $\arg\min$ 回传。

**STE原理：**

```
前向传播：使用离散值
反向传播：假设恒等映射
```

**数学定义：**

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial q} \cdot \frac{\partial q}{\partial z} \approx \frac{\partial L}{\partial q}
$$

**代码实现：**

```python
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, z_q):
        """
        前向传播：返回量化值
        """
        return z_q
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：梯度直通
        """
        return grad_output, None

# 使用
z_q = quantize(z)  # 离散量化
z_q_ste = StraightThroughEstimator.apply(z, z_q)
loss = criterion(z_q_ste, target)
loss.backward()  # 梯度会回传到z
```

**为什么STE有效：**

1. **近似合理性**
   - 量化后的 $z_q$ 接近 $z$
   - 假设 $z_q \approx z$ 是合理近似

2. **实践验证**
   - 在VQ-VAE等模型中广泛使用
   - 实验证明有效

**STE的变体：**

| 变体 | 前向 | 反向 | 适用场景 |
|------|------|------|----------|
| 基本STE | 离散值 | 恒等 | 通用 |
| 软量化STE | 软量化 | 恒等 | 需要可微 |
| Gumbel-STE | Gumbel采样 | 重参数化 | 需要采样 |

---

### Q11: 请解释Gumbel-Softmax如何实现可微分的离散采样。

**答案：**

**Gumbel-Softmax原理：**

1. **Gumbel分布采样**
   $$g = -\log(-\log(u)), \quad u \sim \text{Uniform}(0,1)$$

2. **Gumbel-Max技巧**
   $$k = \arg\max_i (\log \pi_i + g_i)$$
   这等价于从分类分布 $\pi$ 中采样

3. **Gumbel-Softmax松弛**
   $$y_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}$$

**在量化中的应用：**

```python
def gumbel_quantize(z, codebook, temperature=1.0):
    """
    使用Gumbel-Softmax的可微量化
    
    Args:
        z: [B, D] 输入特征
        codebook: [K, D] 码本
        temperature: 温度参数
    """
    # 计算距离（负相似度）
    distances = -torch.cdist(z, codebook)  # [B, K]
    
    # Gumbel-Softmax
    if training:
        # 训练时：软采样
        soft_indices = F.gumbel_softmax(distances, tau=temperature, hard=False)
        z_q_soft = torch.matmul(soft_indices, codebook)  # [B, D]
        
        # 可选：hard模式
        if hard:
            indices = soft_indices.argmax(dim=-1)
            z_q_hard = codebook[indices]
            z_q = z_q_hard - z_q_soft.detach() + z_q_soft
        else:
            z_q = z_q_soft
    else:
        # 推理时：硬量化
        indices = distances.argmax(dim=-1)
        z_q = codebook[indices]
    
    return z_q
```

**温度参数的作用：**

```
温度高 (τ → ∞): 软采样，分布平滑
温度低 (τ → 0): 硬采样，接近one-hot

训练策略：从高温开始，逐渐降温
τ_0 = 1.0 → τ_end = 0.1
```

---

### Q12: 请解释BPR损失函数的原理，以及为什么适合推荐任务。

**答案：**

**BPR (Bayesian Personalized Ranking) 损失：**

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j)} \log \sigma(\hat{x}_{ui} - \hat{x}_{uj})$$

其中：
- $u$：用户
- $i$：正样本（用户交互过的商品）
- $j$：负样本（用户未交互的商品）
- $\hat{x}_{ui}$：用户$u$对商品$i$的预测分数
- $\sigma$：sigmoid函数

**原理推导：**

1. **优化目标**
   - 最大化正样本排在负样本前面的概率
   - $P(i >_u j) = \sigma(\hat{x}_{ui} - \hat{x}_{uj})$

2. **最大后验估计**
   $$\text{MAP} = \prod_{(u,i,j)} P(i >_u j)$$
   
3. **取对数取负**
   $$\mathcal{L} = -\sum \log P(i >_u j)$$

**为什么适合推荐：**

1. **隐式反馈**
   - 只有正样本（点击/购买）
   - 没有负样本（未点击不一定是负样本）
   - BPR通过采样构造负样本

2. **排序优化**
   - 直接优化排序目标
   - 不关心绝对分数，只关心相对顺序

3. **个性化**
   - 每个用户有自己的排序
   - 考虑用户偏好差异

**代码实现：**

```python
def bpr_loss(pos_scores, neg_scores):
    """
    BPR损失
    
    Args:
        pos_scores: [B] 正样本分数
        neg_scores: [B, num_neg] 负样本分数
    """
    pos_scores = pos_scores.unsqueeze(1)  # [B, 1]
    diff = pos_scores - neg_scores  # [B, num_neg]
    loss = -F.logsigmoid(diff).mean()
    return loss
```

**与其他损失对比：**

| 损失 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| BCE | 二分类 | 简单 | 需要负样本标签 |
| **BPR** | 隐式反馈 | 自动构造负样本 | 依赖采样质量 |
| Margin | 度量学习 | 控制间隔 | 需要调间隔参数 |
| Softmax | 多分类 | 概率解释 | 计算量大 |

---

### Q13: 请解释多模态特征融合的MMOE架构原理。

**答案：**

**MMOE (Multi-gate Mixture-of-Experts) 架构：**

```
文本特征 ──┬──> 专家网络1 ──┐
           ├──> 专家网络2 ──┼──> 文本门控 ──┐
           ├──> 专家网络3 ──┤              │
           └──> 专家网络4 ──┘              │
                                            ├──> 融合输出
视觉特征 ──┬──> 专家网络1 ──┐              │
           ├──> 专家网络2 ──┼──> 视觉门控 ──┘
           ├──> 专家网络3 ──┤
           └──> 专家网络4 ──┘
```

**数学公式：**

1. **专家输出**
   $$h_k(x) = \text{MLP}_k(x), \quad k = 1, ..., n$$

2. **门控权重**
   $$g_t(x) = \text{Softmax}(W_t \cdot x)$$

3. **任务输出**
   $$f_t(x) = \sum_{k=1}^{n} g_t^k(x) \cdot h_k(x)$$

**为什么MMOE有效：**

1. **多专家学习不同模式**
   ```
   专家1：学习价格相关特征
   专家2：学习品牌相关特征
   专家3：学习外观相关特征
   专家4：学习功能相关特征
   ```

2. **门控自适应选择**
   - 文本门控：选择文本相关的专家
   - 视觉门控：选择视觉相关的专家

3. **模态解耦与共享**
   - 共享：专家网络可以共享
   - 解耦：门控决定每个模态使用哪些专家

**代码实现：**

```python
class MMOEEncoder(nn.Module):
    def __init__(self, num_experts=4, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # 专家输出
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n, D]
        
        # 门控权重
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, n]
        
        # 加权融合
        output = torch.einsum('bn,bnd->bd', gate_weights, expert_outputs)
        
        return output
```

---

## 三、模型架构设计

### Q14: 请画出完整的模型架构图，并说明各模块的作用。

**答案：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RQ-VAE Recommender 完整架构                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        输入层                                     │    │
│  │  文本特征 [B, 768] (BERT)    视觉特征 [B, 2048] (ResNet)         │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MMOE 多模态编码器                              │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │  文本专家网络          视觉专家网络                        │   │    │
│  │  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐  ┌───┐ ┌───┐ ┌───┐ ┌───┐       │   │    │
│  │  │  │E1 │ │E2 │ │E3 │ │E4 │  │E1 │ │E2 │ │E3 │ │E4 │       │   │    │
│  │  │  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘       │   │    │
│  │  │    │     │     │     │        │     │     │     │         │   │    │
│  │  │    └─────┴─────┴─────┘        └─────┴─────┴─────┘         │   │    │
│  │  │           │                            │                   │   │    │
│  │  │           ▼                            ▼                   │   │    │
│  │  │      文本门控                      视觉门控                 │   │    │
│  │  │    (Softmax权重)               (Softmax权重)               │   │    │
│  │  │           │                            │                   │   │    │
│  │  │           └────────────┬───────────────┘                   │   │    │
│  │  │                        ▼                                   │   │    │
│  │  │                   融合特征 [B, 128]                        │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RQ-VAE 残差量化器                              │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │  Layer 1: z → argmin ||z - e|| → id₁, r₁ = z - e[id₁]    │   │    │
│  │  │  Layer 2: r₁ → argmin ||r₁ - e|| → id₂, r₂ = r₁ - e[id₂] │   │    │
│  │  │  Layer 3: r₂ → argmin ||r₂ - e|| → id₃                   │   │    │
│  │  │                                                          │   │    │
│  │  │  输出: 语义ID [id₁, id₂, id₃]                            │   │    │
│  │  │        重建: z_q = e[id₁] + e[id₂] + e[id₃]             │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    用户序列建模                                   │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │  用户ID嵌入 [B, 128]                                      │   │    │
│  │  │         +                                                 │   │    │
│  │  │  历史序列嵌入 [B, S, 128]                                 │   │    │
│  │  │         │                                                 │   │    │
│  │  │         ▼                                                 │   │    │
│  │  │  ┌─────────────────────────────────────────┐             │   │    │
│  │  │  │         Transformer Encoder             │             │   │    │
│  │  │  │  ┌─────────────────────────────────┐   │             │   │    │
│  │  │  │  │  Multi-Head Self-Attention       │   │             │   │    │
│  │  │  │  │  + Feed Forward Network          │   │             │   │    │
│  │  │  │  │  × 2 layers                      │   │             │   │    │
│  │  │  │  └─────────────────────────────────┘   │             │   │    │
│  │  │  └─────────────────────────────────────────┘             │   │    │
│  │  │         │                                                 │   │    │
│  │  │         ▼                                                 │   │    │
│  │  │  用户表示 [B, 128]                                        │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    预测层                                         │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │  用户表示 [B, 128]                                        │   │    │
│  │  │  候选商品嵌入 [B, K, 128]                                 │   │    │
│  │  │         │                                                 │   │    │
│  │  │         ▼                                                 │   │    │
│  │  │  内积 + MLP → 分数 [B, K]                                │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**各模块作用：**

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| MMOE编码器 | 文本+视觉特征 | 融合特征[128] | 多模态特征融合 |
| RQ-VAE | 融合特征 | 语义ID[3] | 特征量化 |
| 用户嵌入 | 用户ID | 用户向量[128] | 用户表示 |
| 序列编码器 | 历史序列 | 序列表示[128] | 行为建模 |
| 预测层 | 用户+候选 | 分数 | 排序预测 |

---

### Q15: 请说明语义ID嵌入层的设计原理，为什么使用分层嵌入？

**答案：**

**语义ID嵌入层设计：**

```python
class SemanticIDEmbedding(nn.Module):
    def __init__(self, num_layers=3, codebook_size=256, embedding_dim=128):
        super().__init__()
        
        # 每层独立的嵌入表
        self.layer_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(num_layers)
        ])
        
        # 可学习的层权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, semantic_ids):
        """
        Args:
            semantic_ids: [B, 3] 例如 [[45, 128, 201], ...]
        Returns:
            embedding: [B, 128]
        """
        # 获取每层嵌入
        embeds = []
        for i in range(self.num_layers):
            embed = self.layer_embeddings[i](semantic_ids[:, i])
            embeds.append(embed)
        
        # 加权融合
        weights = F.softmax(self.layer_weights, dim=0)
        embedding = sum(w * e for w, e in zip(weights, embeds))
        
        return embedding
```

**为什么使用分层嵌入：**

1. **语义层次性**
   ```
   Layer 1 (id₁): 粗粒度类别（如：电子产品）
   Layer 2 (id₂): 中粒度类别（如：手机）
   Layer 3 (id₃): 细粒度属性（如：iPhone 15 Pro）
   
   不同层捕获不同粒度的语义信息
   ```

2. **独立嵌入表**
   - 每层有独立的嵌入空间
   - 避免不同层的ID冲突
   - 更灵活的表示能力

3. **可学习权重**
   - 模型自动学习各层重要性
   - 不同任务可能关注不同层

**对比其他方案：**

| 方案 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 拼接 | [e₁; e₂; e₃] | 保留所有信息 | 维度增大 |
| 求和 | e₁ + e₂ + e₃ | 维度不变 | 权重固定 |
| **加权融合** | w₁e₁ + w₂e₂ + w₃e₃ | 自适应 | 需要学习 |

---

### Q16: 请解释Transformer序列编码器如何处理用户历史行为。

**答案：**

**Transformer序列编码器架构：**

```python
class TransformerSequenceEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_layers=2, max_seq_length=50):
        super().__init__()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, item_embeddings, mask):
        """
        Args:
            item_embeddings: [B, S, D] 历史商品嵌入
            mask: [B, S] 有效位置掩码
        """
        # 1. 添加位置编码
        x = item_embeddings + self.pos_encoding[:item_embeddings.size(1)]
        
        # 2. Transformer编码
        # 创建注意力掩码（padding位置被忽略）
        src_key_padding_mask = ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 3. 取最后有效位置
        last_valid_idx = mask.sum(dim=1).long() - 1
        output = x[torch.arange(x.size(0)), last_valid_idx]
        
        return output
```

**处理流程详解：**

1. **输入准备**
   ```
   历史商品序列: [item_1, item_2, ..., item_n, PAD, PAD, ...]
   掩码:         [  1,     1,   ...,   1,    0,   0,  ...]
   ```

2. **位置编码**
   ```
   PE(pos, 2i) = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
   
   作用：保留序列顺序信息
   ```

3. **自注意力计算**
   ```
   Attention(Q, K, V) = softmax(QK^T / √d) V
   
   作用：捕获历史商品间的关联
   例如：用户买了手机后可能买手机壳
   ```

4. **掩码处理**
   ```
   PAD位置不参与注意力计算
   避免无效信息影响
   ```

5. **序列表示提取**
   ```
   取最后一个有效位置的输出
   代表用户的当前兴趣
   ```

---

### Q17: 请说明用户表示是如何构建的，用户ID嵌入和序列表示如何融合？

**答案：**

**用户表示构建：**

```python
class UserBehaviorModel(nn.Module):
    def __init__(self, num_users, d_model=128):
        super().__init__()
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(num_users, d_model)
        
        # 序列编码器
        self.sequence_encoder = TransformerSequenceEncoder(d_model)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, user_ids, history_embeddings, history_mask):
        # 1. 用户ID嵌入
        user_embed = self.user_embedding(user_ids)  # [B, D]
        
        # 2. 序列编码
        sequence_output = self.sequence_encoder(
            history_embeddings, history_mask
        )  # [B, D]
        
        # 3. 融合
        user_representation = self.fusion(
            torch.cat([user_embed, sequence_output], dim=-1)
        )  # [B, D]
        
        return user_representation
```

**融合策略分析：**

| 策略 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| 求和 | u + s | 简单 | 信息可能丢失 |
| 拼接+MLP | MLP([u; s]) | 保留信息 | 参数多 |
| 门控 | g·u + (1-g)·s | 自适应 | 复杂 |
| **本项目** | MLP([u; s]) + LN | 稳定 | - |

**为什么需要两种表示：**

1. **用户ID嵌入**
   - 捕获用户的长期偏好
   - 不依赖具体行为
   - 适合用户冷启动（有ID但无行为）

2. **序列表示**
   - 捕获用户的短期兴趣
   - 基于最近行为
   - 动态变化

3. **融合的必要性**
   - 结合长期和短期偏好
   - 更全面的用户建模

---

### Q18: 请解释预测层的设计，为什么同时使用内积和MLP？

**答案：**

**预测层设计：**

```python
class Predictor(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        
        # MLP预测
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, user_repr, candidate_embeds):
        """
        Args:
            user_repr: [B, D]
            candidate_embeds: [B, K, D]
        """
        # 方法1: 内积
        inner_product = torch.bmm(
            user_repr.unsqueeze(1),
            candidate_embeds.transpose(1, 2)
        ).squeeze(1)  # [B, K]
        
        # 方法2: MLP
        user_expanded = user_repr.unsqueeze(1).expand(-1, candidate_embeds.size(1), -1)
        mlp_input = torch.cat([user_expanded, candidate_embeds], dim=-1)
        mlp_scores = self.mlp(mlp_input).squeeze(-1)  # [B, K]
        
        # 综合
        scores = (inner_product + mlp_scores) / self.temperature
        
        return scores
```

**为什么同时使用两种方法：**

1. **内积的优势**
   - 计算高效：$O(D)$
   - 适合检索：可以预计算候选嵌入
   - 几何意义：余弦相似度

2. **MLP的优势**
   - 非线性：可以学习复杂交互
   - 灵活：不限于相似度
   - 特征组合：可以学习特征交叉

3. **组合的优势**
   ```
   内积：捕获相似性（用户喜欢相似商品）
   MLP：捕获交互性（用户对特定特征的偏好）
   
   组合：同时建模两种模式
   ```

**温度参数的作用：**

```
scores = scores / temperature

温度高：分数分布平滑，探索性强
温度低：分数分布尖锐，利用性强

可学习：模型自动调节
```

---

### Q19: 请说明冷启动处理器的设计原理。

**答案：**

**冷启动处理器：**

```python
class ColdStartHandler(nn.Module):
    def __init__(self, rq_vae, mmoe_encoder, recommender):
        super().__init__()
        self.rq_vae = rq_vae
        self.mmoe_encoder = mmoe_encoder
        self.recommender = recommender
    
    @torch.no_grad()
    def encode_new_item(self, text_features, visual_features):
        """
        编码新商品
        
        Args:
            text_features: [B, 768]
            visual_features: [B, 2048]
        
        Returns:
            semantic_ids: [B, 3]
            item_embedding: [B, 128]
        """
        # 1. 多模态编码
        fused_features, _, _, _ = self.mmoe_encoder(
            text_features, visual_features
        )
        
        # 2. RQ-VAE编码
        semantic_ids = self.rq_vae.encode_to_ids(fused_features)
        
        # 3. 获取嵌入
        item_embedding = self.recommender.semantic_id_embedding(semantic_ids)
        
        return semantic_ids, item_embedding
    
    @torch.no_grad()
    def find_similar_items(self, query_semantic_id, all_semantic_ids, top_k=10):
        """
        查找相似商品
        
        Args:
            query_semantic_id: [3]
            all_semantic_ids: [N, 3]
        
        Returns:
            similar_indices: [K]
            similarities: [K]
        """
        # 获取嵌入
        query_embed = self.recommender.semantic_id_embedding(
            query_semantic_id.unsqueeze(0)
        )
        all_embeds = self.recommender.semantic_id_embedding(all_semantic_ids)
        
        # 计算相似度
        query_embed = F.normalize(query_embed, dim=-1)
        all_embeds = F.normalize(all_embeds, dim=-1)
        similarities = torch.matmul(query_embed, all_embeds.t())
        
        # Top-K
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        return top_indices, top_similarities
```

**冷启动处理流程：**

```
新商品上架
    │
    ▼
提取特征（文本+图像）
    │
    ▼
MMOE编码 → 融合特征
    │
    ▼
RQ-VAE编码 → 语义ID
    │
    ▼
语义ID嵌入 → 商品嵌入
    │
    ▼
可被推荐！
```

**为什么能解决冷启动：**

1. **不依赖交互**
   - 只需要商品特征
   - 新商品上架即可获得语义ID

2. **语义相似性**
   - 相似商品获得相似ID
   - 新商品可以继承相似商品的推荐能力

3. **即时可用**
   - 无需重新训练
   - 实时编码

---

### Q20: 请说明模型的总参数量和各模块占比。

**答案：**

**参数量统计：**

| 模块 | 参数量 | 计算公式 | 占比 |
|------|--------|----------|------|
| MMOE编码器 | 1,538,440 | (768+2048)×128×4 + 门控 | 35.2% |
| RQ-VAE | 231,680 | 编码器+解码器+码本 | 5.3% |
| 用户嵌入 | 1,348,736 | 10,537 × 128 | 30.9% |
| 语义ID嵌入 | 98,304 | 3 × 256 × 128 | 2.2% |
| 序列编码器 | 526,336 | Transformer | 12.0% |
| 预测层 | 32,896 | MLP | 0.8% |
| 其他 | 599,696 | 各种LayerNorm等 | 13.7% |
| **总计** | **4,376,088** | - | **100%** |

**参数量计算详解：**

```python
# MMOE编码器
text_experts = 4 * (768 * 256 + 256 * 128)  # 4个文本专家
visual_experts = 4 * (2048 * 256 + 256 * 128)  # 4个视觉专家
text_gate = 768 * 4
visual_gate = 2048 * 4
fusion = 128 * 128
# 总计: ~1.5M

# RQ-VAE
encoder = 128 * 256 + 256 * 128  # 编码器
decoder = 128 * 256 + 256 * 128  # 解码器
codebook = 3 * 256 * 128  # 3层码本
# 总计: ~230K

# 用户嵌入
user_embed = 10537 * 128
# 总计: ~1.3M

# Transformer序列编码器
# 每层: d_model * (4 * d_model + 3 * d_model) ≈ 7 * d_model^2
transformer = 2 * 7 * 128 * 128
# 总计: ~230K
```

**模型大小：**
- float32: 4,376,088 × 4 = 17.5 MB
- float16: 4,376,088 × 2 = 8.75 MB

---

## 四、训练策略与优化

### Q21: 请详细说明两阶段训练策略的设计原理。

**答案：**

**两阶段训练：**

```
阶段1: RQ-VAE预训练
├── 目标: 学习语义ID编码
├── 数据: 商品特征（无需交互）
├── 损失: 重建损失 + commitment损失
└── 输出: 预训练的RQ-VAE

阶段2: 推荐模型微调
├── 目标: 优化推荐效果
├── 数据: 用户交互数据
├── 损失: BPR损失 + 重建损失
└── 输出: 完整推荐模型
```

**为什么需要两阶段：**

1. **解耦学习目标**
   ```
   预训练: 专注于特征量化
   - 学习好的语义表示
   - 不受推荐任务干扰
   
   微调: 专注于推荐任务
   - 优化推荐效果
   - 在好的表示基础上学习
   ```

2. **数据利用**
   ```
   预训练: 利用所有商品特征
   - 不需要交互数据
   - 数据量大
   
   微调: 利用交互数据
   - 需要用户行为
   - 数据量相对小
   ```

3. **训练稳定性**
   ```
   端到端训练问题:
   - 量化操作不可微
   - 多个损失冲突
   - 收敛不稳定
   
   两阶段训练:
   - 每阶段目标明确
   - 训练更稳定
   ```

**训练配置对比：**

| 配置 | 预训练 | 微调 |
|------|--------|------|
| 学习率 | 1e-3 | 1e-4 |
| 批次大小 | 256 | 128 |
| 轮数 | 50 | 20 |
| 优化器 | AdamW | AdamW |
| 损失权重 | rec=1.0, vq=1.0 | rec=1.0, vq=0.1 |

---

### Q22: 请解释学习率调度策略，为什么使用Cosine Annealing？

**答案：**

**Cosine Annealing学习率调度：**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

**学习率变化曲线：**

```
η
│
│  η_max ●───╮
│            ╲
│             ╲
│              ╲
│               ╲
│                ╲
│                 ╲
│                  ●
│  η_min           └────────
└───────────────────────────── t
                   T
```

**为什么选择Cosine Annealing：**

1. **平滑衰减**
   - 学习率平滑下降
   - 避免突变

2. **前期快速学习**
   - 初期学习率高
   - 快速收敛

3. **后期精细调优**
   - 后期学习率低
   - 精细调整

4. **无需手动调节**
   - 只需设置初始和最终学习率
   - 自动调节

**与其他调度策略对比：**

| 策略 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| Step | 每N步衰减 | 简单 | 突变 |
| Exponential | $\eta \cdot \gamma^t$ | 平滑 | 可能衰减太快 |
| **Cosine** | 余弦函数 | 平滑、自动 | - |
| Warmup+Cosine | 先升后降 | 稳定 | 复杂 |

**Warmup策略：**

```python
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup: 线性增加
            return step / warmup_steps
        else:
            # Cosine Annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

### Q23: 请解释负采样策略，以及如何选择负样本数量。

**答案：**

**负采样策略：**

```python
class NegativeSampler:
    def __init__(self, num_items, num_neg=4):
        self.num_items = num_items
        self.num_neg = num_neg
    
    def random_sample(self, pos_item, batch_size):
        """随机负采样"""
        negatives = torch.randint(0, self.num_items, (batch_size, self.num_neg))
        return negatives
    
    def popularity_sample(self, item_popularity, batch_size):
        """基于流行度的负采样"""
        # 流行商品作为难负样本
        probs = item_popularity / item_popularity.sum()
        negatives = torch.multinomial(probs, batch_size * self.num_neg)
        return negatives.view(batch_size, self.num_neg)
    
    def mixed_sample(self, pos_item, item_popularity, batch_size):
        """混合负采样"""
        # 一半随机，一半流行
        random_negs = self.random_sample(pos_item, batch_size // 2)
        popular_negs = self.popularity_sample(item_popularity, batch_size // 2)
        return torch.cat([random_negs, popular_negs], dim=1)
```

**负样本数量选择：**

| 数量 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 1 | 快速 | 信息少 | 快速实验 |
| 4 | 平衡 | - | **推荐** |
| 10 | 信息多 | 慢 | 精细训练 |
| 100+ | 最优 | 很慢 | 最终模型 |

**为什么选择4个负样本：**

1. **计算效率**
   - 每个样本计算4次负样本
   - 总计算量可控

2. **训练效果**
   - 实验证明4个负样本效果最好
   - 太少：信息不足
   - 太多：噪声增加

3. **内存占用**
   - 负样本需要存储特征
   - 4个负样本内存占用合理

---

### Q24: 请解释梯度裁剪的作用和实现。

**答案：**

**梯度裁剪原理：**

$$\mathbf{g} = \min(1, \frac{c}{\|\mathbf{g}\|}) \mathbf{g}$$

其中 $c$ 是裁剪阈值。

**作用：**

1. **防止梯度爆炸**
   ```
   深度网络中，梯度可能指数级增长
   梯度裁剪限制梯度范数
   ```

2. **稳定训练**
   ```
   避免参数更新过大
   训练更稳定
   ```

3. **提高泛化**
   ```
   限制参数空间
   防止过拟合
   ```

**实现：**

```python
# PyTorch内置
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 手动实现
def clip_grad_norm(parameters, max_norm):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm
```

**阈值选择：**

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| 0.5 | 强裁剪 | 梯度爆炸严重 |
| **1.0** | 平衡 | **推荐** |
| 5.0 | 弱裁剪 | 梯度稳定 |

---

### Q25: 请解释早停(Early Stopping)策略的实现。

**答案：**

**早停原理：**

```
监控验证集指标
如果连续N个epoch没有提升，停止训练
```

**实现：**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小提升
            mode: 'max'或'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

# 使用
early_stopping = EarlyStopping(patience=5, mode='max')

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_score = evaluate()
    
    if early_stopping(val_score):
        print(f"Early stopping at epoch {epoch}")
        break
```

**参数选择：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| patience | 5-10 | 容忍轮数 |
| min_delta | 0.001 | 最小提升 |
| mode | 'max' | Recall用max |

---

### Q26: 请解释混合精度训练的原理和优势。

**答案：**

**混合精度训练：**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**原理：**

1. **FP16前向传播**
   - 计算速度快
   - 内存占用少

2. **FP32梯度计算**
   - 避免精度损失
   - 保持训练稳定

3. **梯度缩放**
   - 防止FP16下溢
   - 动态调整缩放因子

**优势：**

| 方面 | FP32 | FP16 | 混合精度 |
|------|------|------|----------|
| 速度 | 1x | 2-3x | 1.5-2x |
| 内存 | 1x | 0.5x | 0.6x |
| 精度 | 高 | 低 | 高 |
| 稳定性 | 高 | 低 | 高 |

**注意事项：**

1. **不支持的操作**
   - 部分操作需要FP32
   - 自动回退

2. **梯度缩放**
   - 必须使用GradScaler
   - 防止梯度下溢

---

## 五、评估与指标

### Q27: 请详细解释Recall@K和NDCG@K的计算方法。

**答案：**

**Recall@K：**

$$\text{Recall@K} = \frac{|\text{推荐列表中相关物品}|}{|\text{所有相关物品}|}$$

```python
def recall_at_k(scores, labels, k):
    """
    Args:
        scores: [B, N] 预测分数
        labels: [B, N] 真实标签（0/1）
        k: Top-K
    """
    batch_size = scores.size(0)
    recalls = []
    
    for i in range(batch_size):
        # 获取Top-K索引
        _, top_k_indices = torch.topk(scores[i], k)
        
        # 计算命中数
        relevant = labels[i].sum().item()
        if relevant == 0:
            continue
        
        hits = labels[i][top_k_indices].sum().item()
        recalls.append(hits / relevant)
    
    return np.mean(recalls)
```

**NDCG@K：**

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

```python
def ndcg_at_k(scores, labels, k):
    """
    Args:
        scores: [B, N] 预测分数
        labels: [B, N] 真实标签
        k: Top-K
    """
    batch_size = scores.size(0)
    ndcgs = []
    
    for i in range(batch_size):
        # 获取Top-K索引
        _, top_k_indices = torch.topk(scores[i], k)
        
        # 计算DCG
        rels = labels[i][top_k_indices].float()
        gains = torch.pow(2.0, rels) - 1
        discounts = torch.log2(torch.arange(1, k + 1, dtype=torch.float) + 1)
        dcg = (gains / discounts).sum().item()
        
        # 计算IDCG
        ideal_rels, _ = torch.sort(labels[i], descending=True)
        ideal_rels = ideal_rels[:k].float()
        ideal_gains = torch.pow(2.0, ideal_rels) - 1
        idcg = (ideal_gains / discounts).sum().item()
        
        # 计算NDCG
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    
    return np.mean(ndcgs)
```

**区别：**

| 指标 | 考虑位置 | 考虑顺序 | 适用场景 |
|------|----------|----------|----------|
| Recall@K | 否 | 否 | 召回评估 |
| Precision@K | 否 | 否 | 准确评估 |
| **NDCG@K** | **是** | **是** | **排序评估** |

---

### Q28: 请解释Hit Rate和MRR的计算方法。

**答案：**

**Hit Rate@K：**

$$\text{Hit Rate@K} = \frac{\text{命中用户数}}{\text{总用户数}}$$

```python
def hit_rate_at_k(scores, labels, k):
    """
    命中率：推荐列表中是否包含至少一个相关物品
    """
    batch_size = scores.size(0)
    hits = 0
    
    for i in range(batch_size):
        _, top_k_indices = torch.topk(scores[i], k)
        if labels[i][top_k_indices].sum() > 0:
            hits += 1
    
    return hits / batch_size
```

**MRR (Mean Reciprocal Rank)：**

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

```python
def mrr(scores, labels):
    """
    平均倒数排名：第一个相关物品的排名倒数
    """
    batch_size = scores.size(0)
    reciprocal_ranks = []
    
    for i in range(batch_size):
        # 按分数排序
        _, sorted_indices = torch.sort(scores[i], descending=True)
        sorted_labels = labels[i][sorted_indices]
        
        # 找到第一个相关物品的位置
        for rank, label in enumerate(sorted_labels):
            if label == 1:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
```

**示例：**

```
用户A推荐列表: [item1(相关), item2, item3, ...]
- Hit Rate@3: 1 (命中)
- MRR: 1/1 = 1.0

用户B推荐列表: [item1, item2, item3(相关), ...]
- Hit Rate@3: 1 (命中)
- MRR: 1/3 = 0.33

用户C推荐列表: [item1, item2, item3, item4(相关), ...]
- Hit Rate@3: 0 (未命中)
- MRR: 1/4 = 0.25
```

---

### Q29: 请说明如何评估冷启动效果。

**答案：**

**冷启动评估方法：**

```python
def evaluate_cold_start(model, test_data, cold_threshold=2):
    """
    评估冷启动商品效果
    """
    # 1. 识别冷启动商品
    item_counts = test_data['item_id'].value_counts()
    cold_items = item_counts[item_counts <= cold_threshold].index.tolist()
    warm_items = item_counts[item_counts > cold_threshold].index.tolist()
    
    # 2. 分别评估
    cold_results = evaluate_on_items(model, test_data, cold_items)
    warm_results = evaluate_on_items(model, test_data, warm_items)
    
    return {
        'cold_start_recall': cold_results['recall@10'],
        'warm_recall': warm_results['recall@10'],
        'cold_start_coverage': len(set(cold_results['recommended_items'])) / len(cold_items),
        'improvement': cold_results['recall@10'] / warm_results['recall@10']
    }

def evaluate_semantic_similarity(model, cold_items, all_items):
    """
    评估语义ID的相似性
    """
    # 获取语义ID
    cold_semantic_ids = model.encode_items(cold_items)
    all_semantic_ids = model.encode_items(all_items)
    
    # 计算语义相似度
    cold_embeds = model.semantic_id_embedding(cold_semantic_ids)
    all_embeds = model.semantic_id_embedding(all_semantic_ids)
    
    # 找最相似商品
    similarities = torch.matmul(cold_embeds, all_embeds.t())
    top_similar = similarities.argmax(dim=1)
    
    # 计算相似商品的类别一致性
    consistency = calculate_category_consistency(cold_items, top_similar)
    
    return consistency
```

**冷启动评估指标：**

| 指标 | 说明 | 目标 |
|------|------|------|
| 冷启动召回率 | 冷启动商品的Recall@K | 越高越好 |
| 冷启动覆盖率 | 被推荐的冷启动商品比例 | 越高越好 |
| 语义一致性 | 相似商品的类别一致性 | 越高越好 |
| 冷启动vs热启动 | 效果对比 | 差距越小越好 |

---

### Q30: 请说明A/B测试的设计方案。

**答案：**

**A/B测试设计：**

```python
class ABTest:
    def __init__(self, control_model, treatment_model):
        self.control = control_model  # 基线模型
        self.treatment = treatment_model  # 新模型
        self.results = {'control': [], 'treatment': []}
    
    def run(self, user_requests, ratio=0.5):
        """
        运行A/B测试
        
        Args:
            user_requests: 用户请求列表
            ratio: 实验组比例
        """
        import random
        random.shuffle(user_requests)
        
        split = int(len(user_requests) * ratio)
        control_requests = user_requests[:split]
        treatment_requests = user_requests[split:]
        
        # 对照组
        for request in control_requests:
            recommendations = self.control.recommend(request)
            self.results['control'].append({
                'user_id': request['user_id'],
                'recommendations': recommendations
            })
        
        # 实验组
        for request in treatment_requests:
            recommendations = self.treatment.recommend(request)
            self.results['treatment'].append({
                'user_id': request['user_id'],
                'recommendations': recommendations
            })
    
    def analyze(self, user_feedback):
        """
        分析结果
        """
        metrics = ['click_rate', 'conversion_rate', 'dwell_time']
        
        results = {}
        for group in ['control', 'treatment']:
            group_feedback = [
                fb for fb in user_feedback 
                if fb['user_id'] in [r['user_id'] for r in self.results[group]]
            ]
            
            results[group] = {
                'click_rate': calculate_click_rate(group_feedback),
                'conversion_rate': calculate_conversion_rate(group_feedback),
                'dwell_time': calculate_avg_dwell_time(group_feedback)
            }
        
        # 统计显著性检验
        significance = self.statistical_test(results)
        
        return {
            'control': results['control'],
            'treatment': results['treatment'],
            'improvement': {
                k: (results['treatment'][k] - results['control'][k]) / results['control'][k]
                for k in metrics
            },
            'significance': significance
        }
```

**A/B测试指标：**

| 指标 | 计算方法 | 目标 |
|------|----------|------|
| CTR | 点击数/展示数 | 提升 |
| CVR | 转化数/点击数 | 提升 |
| 停留时长 | 平均停留时间 | 提升 |
| GMV | 成交金额 | 提升 |

---

## 六、工程实现

### Q31: 请说明如何处理大规模商品特征存储。

**答案：**

**存储策略：**

```python
import numpy as np
import h5py

class FeatureStore:
    def __init__(self, feature_dim, storage_type='memory'):
        self.feature_dim = feature_dim
        self.storage_type = storage_type
        
        if storage_type == 'memory':
            self.features = {}
        elif storage_type == 'mmap':
            self.features = None
        elif storage_type == 'hdf5':
            self.h5_file = None
    
    def save(self, features, path):
        """保存特征"""
        if self.storage_type == 'memory':
            np.save(path, features)
        elif self.storage_type == 'mmap':
            np.save(path, features)
        elif self.storage_type == 'hdf5':
            with h5py.File(path, 'w') as f:
                f.create_dataset('features', data=features)
    
    def load(self, path):
        """加载特征"""
        if self.storage_type == 'memory':
            self.features = np.load(path)
        elif self.storage_type == 'mmap':
            self.features = np.load(path, mmap_mode='r')
        elif self.storage_type == 'hdf5':
            self.h5_file = h5py.File(path, 'r')
            self.features = self.h5_file['features']
    
    def get(self, indices):
        """获取特征"""
        if self.storage_type == 'memory':
            return self.features[indices]
        elif self.storage_type == 'mmap':
            return self.features[indices]
        elif self.storage_type == 'hdf5':
            return self.features[indices]

# 使用示例
store = FeatureStore(768, storage_type='mmap')
store.load('text_features.npy')
features = store.get([0, 1, 2])  # 按需加载
```

**存储优化：**

| 方法 | 内存占用 | 加载速度 | 适用场景 |
|------|----------|----------|----------|
| 内存加载 | 高 | 快 | 小规模 |
| 内存映射 | 低 | 中 | 大规模 |
| HDF5 | 低 | 慢 | 超大规模 |
| 分片存储 | 低 | 中 | 分布式 |

**特征压缩：**

```python
# Float16压缩
features_fp16 = features.astype(np.float16)  # 减少50%存储

# 量化压缩
features_int8 = (features * 127).astype(np.int8)  # 减少75%存储

# PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
features_reduced = pca.fit_transform(features)  # 减少维度
```

---

### Q32: 请说明如何优化推理速度。

**答案：**

**推理优化策略：**

```python
class OptimizedRecommender:
    def __init__(self, model):
        self.model = model
        self.precomputed = False
        
    def precompute(self, all_items_features):
        """预计算所有商品的语义ID和嵌入"""
        with torch.no_grad():
            # 批量编码
            semantic_ids = []
            embeddings = []
            
            for i in range(0, len(all_items_features), 256):
                batch = all_items_features[i:i+256]
                ids = self.model.encode(batch)
                emb = self.model.semantic_id_embedding(ids)
                
                semantic_ids.append(ids)
                embeddings.append(emb)
            
            self.all_semantic_ids = torch.cat(semantic_ids)
            self.all_embeddings = torch.cat(embeddings)
            self.precomputed = True
    
    def recommend_fast(self, user_id, history_ids, top_k=10):
        """快速推荐"""
        # 1. 获取用户表示
        user_repr = self.model.get_user_representation(user_id, history_ids)
        
        # 2. 内积检索
        scores = torch.matmul(user_repr, self.all_embeddings.t())
        
        # 3. Top-K
        top_scores, top_indices = torch.topk(scores, top_k)
        
        return top_indices, top_scores
```

**使用FAISS加速：**

```python
import faiss

class FAISSRetriever:
    def __init__(self, embedding_dim=128):
        self.index = faiss.IndexFlatIP(embedding_dim)
    
    def add(self, embeddings):
        """添加向量"""
        self.index.add(embeddings.cpu().numpy())
    
    def search(self, query, k=10):
        """检索"""
        scores, indices = self.index.search(query.cpu().numpy(), k)
        return indices, scores

# 使用
retriever = FAISSRetriever(128)
retriever.add(all_embeddings)
indices, scores = retriever.search(user_embedding, k=10)
```

**优化效果：**

| 优化方法 | 速度提升 | 内存变化 |
|----------|----------|----------|
| 预计算 | 10x | 增加 |
| FAISS | 100x | 增加 |
| 量化 | 2x | 减少 |
| 批处理 | 5x | 不变 |

---

### Q33: 请说明API接口设计。

**答案：**

**FastAPI接口设计：**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

app = FastAPI(title="RQ-VAE Recommender API")

# 请求模型
class RecommendRequest(BaseModel):
    user_id: int
    history_items: List[int]
    top_k: int = 10

class SimilarRequest(BaseModel):
    item_id: int
    top_k: int = 10

class EncodeRequest(BaseModel):
    text_features: List[float]
    visual_features: List[float]

# 响应模型
class RecommendResponse(BaseModel):
    item_ids: List[int]
    scores: List[float]
    semantic_ids: List[List[int]]

class SimilarResponse(BaseModel):
    item_ids: List[int]
    similarities: List[float]

# API端点
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """获取推荐"""
    try:
        # 获取推荐
        item_ids, scores, semantic_ids = model.recommend(
            request.user_id,
            request.history_items,
            request.top_k
        )
        
        return RecommendResponse(
            item_ids=item_ids.tolist(),
            scores=scores.tolist(),
            semantic_ids=semantic_ids.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar", response_model=SimilarResponse)
async def find_similar(request: SimilarRequest):
    """查找相似商品"""
    try:
        item_ids, similarities = model.find_similar(
            request.item_id,
            request.top_k
        )
        
        return SimilarResponse(
            item_ids=item_ids.tolist(),
            similarities=similarities.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode")
async def encode_item(request: EncodeRequest):
    """编码新商品"""
    try:
        text_feat = torch.tensor([request.text_features])
        visual_feat = torch.tensor([request.visual_features])
        
        semantic_id = model.encode_new_item(text_feat, visual_feat)
        
        return {"semantic_id": semantic_id.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}
```

**API性能优化：**

```python
# 1. 异步处理
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    result = await asyncio.get_event_loop().run_in_executor(
        None, model.recommend, request.user_id, request.history_items
    )
    return result

# 2. 批处理
from fastapi.concurrency import run_in_threadpool

@app.post("/batch_recommend")
async def batch_recommend(requests: List[RecommendRequest]):
    results = await run_in_threadpool(
        model.batch_recommend, requests
    )
    return results

# 3. 缓存
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_user_embedding(user_id: int):
    return model.user_embedding.weight[user_id]
```

---

### Q34: 请说明Docker部署方案。

**答案：**

**Dockerfile：**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  recommender:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/logs/rq_vae_pretraining/checkpoints/best_model.pt
      - DEVICE=cpu
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**部署命令：**

```bash
# 构建镜像
docker build -t rq-vae-recommender .

# 运行容器
docker run -d -p 8000:8000 --name recommender rq-vae-recommender

# 使用docker-compose
docker-compose up -d

# 查看日志
docker logs recommender

# 扩容
docker-compose up -d --scale recommender=3
```

---

### Q35: 请说明监控和日志方案。

**答案：**

**日志配置：**

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger('rq_vae_recommender')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

**监控指标：**

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter(
    'recommend_requests_total',
    'Total recommendation requests'
)

REQUEST_LATENCY = Histogram(
    'recommend_request_latency_seconds',
    'Request latency in seconds'
)

RECALL_SCORE = Gauge(
    'model_recall_score',
    'Model recall score'
)

# 使用
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    REQUEST_COUNT.inc()
    
    with REQUEST_LATENCY.time():
        result = model.recommend(request)
    
    return result
```

**健康检查：**

```python
@app.get("/health")
async def health_check():
    checks = {
        "model_loaded": model is not None,
        "features_loaded": feature_store is not None,
        "memory_usage": get_memory_usage(),
        "gpu_available": torch.cuda.is_available()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }
```

---

## 七、代码实现

### Q36: 请手写残差量化的完整实现。

**答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualQuantizer(nn.Module):
    """
    残差量化器完整实现
    """
    def __init__(
        self,
        num_layers: int = 3,
        codebook_size: int = 256,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # 码本
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, embedding_dim) * 0.1)
            for _ in range(num_layers)
        ])
        
        # EMA统计量
        for i in range(num_layers):
            self.register_buffer(f'cluster_size_{i}', torch.ones(codebook_size))
            self.register_buffer(f'cluster_sum_{i}', self.codebooks[i].data.clone())
            self.register_buffer(f'steps_unused_{i}', torch.zeros(codebook_size))
    
    def quantize_layer(self, z, codebook, layer_idx):
        """单层量化"""
        batch_size = z.size(0)
        
        # 计算距离
        distances = torch.cdist(z, codebook)  # [B, K]
        
        # 找最近邻
        indices = distances.argmin(dim=-1)  # [B]
        
        # 获取量化向量
        z_q = codebook[indices]  # [B, D]
        
        # EMA更新
        if self.training:
            self._update_ema(z, indices, layer_idx)
        
        # Commitment loss
        commitment_loss = F.mse_loss(z.detach(), z_q)
        
        # STE
        z_q = z + (z_q - z).detach()
        
        return z_q, indices, commitment_loss
    
    def _update_ema(self, z, indices, layer_idx):
        """EMA更新码本"""
        with torch.no_grad():
            # 统计
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            cluster_sum = one_hot.T @ z
            
            # EMA更新
            cluster_size_ema = getattr(self, f'cluster_size_{layer_idx}')
            cluster_sum_ema = getattr(self, f'cluster_sum_{layer_idx}')
            
            cluster_size_ema.data = (
                self.ema_decay * cluster_size_ema.data + 
                (1 - self.ema_decay) * cluster_size
            )
            cluster_sum_ema.data = (
                self.ema_decay * cluster_sum_ema.data + 
                (1 - self.ema_decay) * cluster_sum
            )
            
            # 更新码本
            self.codebooks[layer_idx].data = (
                cluster_sum_ema / cluster_size_ema.unsqueeze(1)
            )
            
            # 死码检测
            steps_unused = getattr(self, f'steps_unused_{layer_idx}')
            steps_unused.data += 1
            steps_unused.data[indices.unique()] = 0
            
            # 死码重置
            dead_mask = steps_unused > self.dead_code_threshold
            if dead_mask.any():
                num_dead = dead_mask.sum().item()
                sample_idx = torch.randperm(z.size(0))[:num_dead]
                self.codebooks[layer_idx].data[dead_mask] = (
                    z[sample_idx] + torch.randn(num_dead, self.embedding_dim, device=z.device) * 0.01
                )
                steps_unused.data[dead_mask] = 0
    
    def forward(self, z):
        """
        前向传播
        
        Args:
            z: [B, D] 输入特征
        
        Returns:
            z_q: [B, D] 重建特征
            indices: [B, L] 语义ID序列
            loss: 总损失
            info: 额外信息
        """
        # 编码
        z_encoded = self.encoder(z)
        
        # 残差量化
        residual = z_encoded
        z_q = 0
        all_indices = []
        total_commitment_loss = 0
        
        for i in range(self.num_layers):
            z_q_i, indices_i, commitment_loss = self.quantize_layer(
                residual, self.codebooks[i], i
            )
            
            z_q = z_q + z_q_i
            all_indices.append(indices_i)
            total_commitment_loss = total_commitment_loss + commitment_loss
            
            residual = residual - z_q_i
        
        # 解码
        z_decoded = self.decoder(z_q)
        
        # 重建损失
        reconstruction_loss = F.mse_loss(z_decoded, z)
        
        # 总损失
        loss = reconstruction_loss + self.commitment_cost * total_commitment_loss
        
        # 语义ID
        indices = torch.stack(all_indices, dim=1)  # [B, L]
        
        info = {
            'reconstruction_loss': reconstruction_loss.item(),
            'commitment_loss': total_commitment_loss.item()
        }
        
        return z_decoded, indices, loss, info
    
    def encode_to_ids(self, z):
        """仅编码为语义ID"""
        with torch.no_grad():
            z_encoded = self.encoder(z)
            residual = z_encoded
            all_indices = []
            
            for i in range(self.num_layers):
                distances = torch.cdist(residual, self.codebooks[i])
                indices = distances.argmin(dim=-1)
                all_indices.append(indices)
                residual = residual - self.codebooks[i][indices]
            
            return torch.stack(all_indices, dim=1)
```

---

### Q37: 请手写Transformer序列编码器。

**答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, S, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, S_q, D]
            key: [B, S_k, D]
            value: [B, S_k, D]
            mask: [B, S_k]
        """
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 加权求和
        output = torch.matmul(attention, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出变换
        output = self.out_linear(output)
        
        return output, attention


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerSequenceEncoder(nn.Module):
    """Transformer序列编码器"""
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer块
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, S, D] 序列嵌入
            mask: [B, S] 有效位置掩码
        
        Returns:
            output: [B, D] 序列表示
        """
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 归一化
        x = self.norm(x)
        
        # 取最后有效位置
        if mask is not None:
            seq_lengths = mask.sum(dim=1).long() - 1
            batch_size = x.size(0)
            output = x[torch.arange(batch_size), seq_lengths]
        else:
            output = x[:, -1, :]
        
        return output
```

---

### Q38: 请手写完整的推荐模型。

**答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticRecommender(nn.Module):
    """
    完整的语义推荐模型
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_quantization_layers: int = 3,
        codebook_size: int = 256,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_quantization_layers = num_quantization_layers
        self.codebook_size = codebook_size
        self.d_model = d_model
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, d_model)
        
        # 语义ID嵌入
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model)
            for _ in range(num_quantization_layers)
        ])
        
        # 层权重
        self.layer_weights = nn.Parameter(torch.ones(num_quantization_layers) / num_quantization_layers)
        
        # 序列编码器
        from models.user_sequence import TransformerSequenceEncoder
        self.sequence_encoder = TransformerSequenceEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # 用户表示融合
        self.user_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        for emb in self.semantic_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
    
    def encode_semantic_ids(self, semantic_ids):
        """
        编码语义ID
        
        Args:
            semantic_ids: [B, L] 或 [B, S, L]
        
        Returns:
            embeddings: [B, D] 或 [B, S, D]
        """
        original_shape = semantic_ids.shape
        if len(original_shape) == 2:
            semantic_ids = semantic_ids.unsqueeze(1)
        
        batch_size, seq_len, num_layers = semantic_ids.shape
        
        # 获取每层嵌入
        embeds = []
        for i in range(num_layers):
            emb = self.semantic_embeddings[i](semantic_ids[:, :, i])
            embeds.append(emb)
        
        # 加权融合
        weights = F.softmax(self.layer_weights, dim=0)
        embeddings = sum(w * e for w, e in zip(weights, embeds))
        
        if len(original_shape) == 2:
            embeddings = embeddings.squeeze(1)
        
        return embeddings
    
    def forward(
        self,
        user_ids,
        history_semantic_ids,
        history_mask,
        candidate_semantic_ids
    ):
        """
        前向传播
        
        Args:
            user_ids: [B]
            history_semantic_ids: [B, S, L]
            history_mask: [B, S]
            candidate_semantic_ids: [B, K, L]
        
        Returns:
            scores: [B, K]
            info: 额外信息
        """
        batch_size = user_ids.size(0)
        num_candidates = candidate_semantic_ids.size(1)
        
        # 1. 用户ID嵌入
        user_embed = self.user_embedding(user_ids)  # [B, D]
        
        # 2. 历史序列嵌入
        history_embeds = self.encode_semantic_ids(history_semantic_ids)  # [B, S, D]
        
        # 3. 序列编码
        sequence_output = self.sequence_encoder(history_embeds, history_mask)  # [B, D]
        
        # 4. 用户表示融合
        user_representation = self.user_fusion(
            torch.cat([user_embed, sequence_output], dim=-1)
        )  # [B, D]
        
        # 5. 候选物品嵌入
        candidate_embeds = self.encode_semantic_ids(candidate_semantic_ids)  # [B, K, D]
        
        # 6. 计算分数
        # 内积
        inner_product = torch.bmm(
            user_representation.unsqueeze(1),
            candidate_embeds.transpose(1, 2)
        ).squeeze(1)  # [B, K]
        
        # MLP
        user_expanded = user_representation.unsqueeze(1).expand(-1, num_candidates, -1)
        mlp_input = torch.cat([user_expanded, candidate_embeds], dim=-1)
        mlp_scores = self.predictor(mlp_input).squeeze(-1)  # [B, K]
        
        # 综合
        scores = (inner_product + mlp_scores) / self.temperature
        
        info = {
            'user_embed_norm': user_embed.norm(dim=-1).mean().item(),
            'sequence_output_norm': sequence_output.norm(dim=-1).mean().item(),
            'candidate_embed_norm': candidate_embeds.norm(dim=-1).mean().item()
        }
        
        return scores, info
    
    def compute_loss(self, pos_scores, neg_scores):
        """
        计算BPR损失
        
        Args:
            pos_scores: [B]
            neg_scores: [B, num_neg]
        """
        pos_scores = pos_scores.unsqueeze(1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss
```

---

### Q39: 请手写评估指标计算。

**答案：**

```python
import torch
import numpy as np
from typing import List, Dict, Optional

class Recall:
    """Recall@K"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算Recall@K
        
        Args:
            scores: [B, N] 预测分数
            labels: [B, N] 真实标签
        """
        batch_size = scores.size(0)
        recalls = []
        
        for i in range(batch_size):
            # Top-K索引
            _, top_k_indices = torch.topk(scores[i], min(self.k, scores.size(1)))
            
            # 计算命中
            relevant = labels[i].sum().item()
            if relevant == 0:
                continue
            
            hits = labels[i][top_k_indices].sum().item()
            recalls.append(hits / relevant)
        
        return np.mean(recalls) if recalls else 0.0


class NDCG:
    """NDCG@K"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算NDCG@K
        """
        batch_size = scores.size(0)
        ndcgs = []
        
        for i in range(batch_size):
            k = min(self.k, scores.size(1))
            
            # Top-K索引
            _, top_k_indices = torch.topk(scores[i], k)
            
            # DCG
            rels = labels[i][top_k_indices].float()
            gains = torch.pow(2.0, rels) - 1
            discounts = torch.log2(torch.arange(1, k + 1, dtype=torch.float) + 1)
            dcg = (gains / discounts).sum().item()
            
            # IDCG
            ideal_rels, _ = torch.sort(labels[i], descending=True)
            ideal_rels = ideal_rels[:k].float()
            ideal_gains = torch.pow(2.0, ideal_rels) - 1
            idcg = (ideal_gains / discounts).sum().item()
            
            # NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0


class HitRate:
    """Hit Rate@K"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算Hit Rate@K
        """
        batch_size = scores.size(0)
        hits = 0
        
        for i in range(batch_size):
            _, top_k_indices = torch.topk(scores[i], min(self.k, scores.size(1)))
            if labels[i][top_k_indices].sum() > 0:
                hits += 1
        
        return hits / batch_size


class MRR:
    """Mean Reciprocal Rank"""
    
    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算MRR
        """
        batch_size = scores.size(0)
        reciprocal_ranks = []
        
        for i in range(batch_size):
            # 排序
            _, sorted_indices = torch.sort(scores[i], descending=True)
            sorted_labels = labels[i][sorted_indices]
            
            # 找第一个相关物品
            for rank, label in enumerate(sorted_labels):
                if label == 1:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = ['recall@10', 'ndcg@10', 'hit_rate@10', 'mrr']
        
        self.metrics = {}
        for metric in metrics:
            metric_lower = metric.lower()
            if 'recall' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metrics[metric] = Recall(k=k)
            elif 'ndcg' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metrics[metric] = NDCG(k=k)
            elif 'hit_rate' in metric_lower or 'hr' in metric_lower:
                k = int(metric_lower.split('@')[1]) if '@' in metric_lower else 10
                self.metrics[metric] = HitRate(k=k)
            elif 'mrr' in metric_lower:
                self.metrics[metric] = MRR()
    
    def compute(self, scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        计算所有指标
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(scores, labels)
        return results
```

---

### Q40: 请手写数据加载器。

**答案：**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class RQVAEDataset(Dataset):
    """RQ-VAE预训练数据集"""
    
    def __init__(
        self,
        text_features: np.ndarray,
        visual_features: np.ndarray
    ):
        self.text_features = text_features.astype(np.float32)
        self.visual_features = visual_features.astype(np.float32)
        
        assert len(text_features) == len(visual_features)
    
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        return {
            'text_features': torch.from_numpy(self.text_features[idx]),
            'visual_features': torch.from_numpy(self.visual_features[idx]),
            'item_idx': idx
        }


class SequentialRecommendationDataset(Dataset):
    """序列推荐数据集"""
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        text_features: np.ndarray,
        visual_features: np.ndarray,
        max_seq_length: int = 50,
        num_negatives: int = 4
    ):
        self.interactions = interactions
        self.text_features = text_features.astype(np.float32)
        self.visual_features = visual_features.astype(np.float32)
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        
        # 按用户分组
        self.user_sequences = self._group_by_user()
        self.num_items = len(text_features)
    
    def _group_by_user(self) -> Dict:
        user_seqs = {}
        for _, row in self.interactions.iterrows():
            uid = row['user_id']
            if uid not in user_seqs:
                user_seqs[uid] = []
            user_seqs[uid].append({
                'item_id': row['item_id'],
                'timestamp': row.get('timestamp', 0)
            })
        
        for uid in user_seqs:
            user_seqs[uid].sort(key=lambda x: x['timestamp'])
        
        return user_seqs
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        target_item = row['item_id']
        
        # 历史序列
        user_seq = self.user_sequences.get(user_id, [])
        history = [i['item_id'] for i in user_seq if i['item_id'] != target_item]
        
        # 截断/填充
        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length:]
        
        history_mask = [1] * len(history)
        while len(history) < self.max_seq_length:
            history.append(0)
            history_mask.append(0)
        
        # 负采样
        negatives = []
        for _ in range(self.num_negatives):
            neg = np.random.randint(0, self.num_items)
            while neg == target_item or neg in negatives:
                neg = np.random.randint(0, self.num_items)
            negatives.append(neg)
        
        return {
            'user_id': user_id,
            'history_items': np.array(history),
            'history_mask': np.array(history_mask, dtype=np.float32),
            'target_item': target_item,
            'target_text': self.text_features[target_item],
            'target_visual': self.visual_features[target_item],
            'negatives': np.array(negatives)
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 加载数据
    text_features = np.load(f'{data_dir}/text_features.npy')
    visual_features = np.load(f'{data_dir}/visual_features.npy')
    interactions = pd.read_csv(f'{data_dir}/interactions_processed.csv')
    
    # 划分
    indices = np.random.permutation(len(interactions))
    train_size = int(len(interactions) * 0.7)
    val_size = int(len(interactions) * 0.15)
    
    train_data = interactions.iloc[indices[:train_size]]
    val_data = interactions.iloc[indices[train_size:train_size + val_size]]
    test_data = interactions.iloc[indices[train_size + val_size:]]
    
    # 创建数据集
    train_dataset = SequentialRecommendationDataset(
        train_data, text_features, visual_features
    )
    val_dataset = SequentialRecommendationDataset(
        val_data, text_features, visual_features
    )
    test_dataset = SequentialRecommendationDataset(
        test_data, text_features, visual_features
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
```

---

## 八、系统设计

### Q41: 请设计一个完整的推荐系统架构。

**答案：**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           推荐系统完整架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         客户端层                                      │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │    │
│  │  │  App    │  │  Web    │  │ 小程序  │  │  API    │                 │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                 │    │
│  └───────┼────────────┼────────────┼────────────┼──────────────────────┘    │
│          │            │            │            │                            │
│          └────────────┴────────────┴────────────┘                            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         网关层                                        │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  API Gateway (Nginx / Kong)                                  │   │    │
│  │  │  - 负载均衡                                                   │   │    │
│  │  │  - 限流                                                       │   │    │
│  │  │  - 认证                                                       │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         服务层                                        │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │    │
│  │  │  召回服务      │  │  排序服务      │  │  重排服务      │           │    │
│  │  │               │  │               │  │               │           │    │
│  │  │  - 协同过滤    │  │  - RQ-VAE     │  │  - 多样性      │           │    │
│  │  │  - 向量召回    │  │  - DeepFM     │  │  - 去重        │           │    │
│  │  │  - 热门召回    │  │  - DIN        │  │  - 业务规则    │           │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │    │
│  │          │                  │                  │                    │    │
│  │          └──────────────────┴──────────────────┘                    │    │
│  │                             │                                        │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │    │
│  │  │  特征服务      │  │  用户服务      │  │  商品服务      │           │    │
│  │  │               │  │               │  │               │           │    │
│  │  │  - 特征存储    │  │  - 用户画像    │  │  - 商品信息    │           │    │
│  │  │  - 特征计算    │  │  - 行为序列    │  │  - 语义ID      │           │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         存储层                                        │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │    │
│  │  │  Redis        │  │  MySQL        │  │  ES           │           │    │
│  │  │  - 缓存        │  │  - 用户数据    │  │  - 搜索        │           │    │
│  │  │  - 实时特征    │  │  - 商品数据    │  │  - 向量检索    │           │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │    │
│  │  │  Kafka        │  │  HDFS         │  │  Faiss        │           │    │
│  │  │  - 消息队列    │  │  - 离线数据    │  │  - 向量索引    │           │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Q42: 请设计RQ-VAE模型的在线学习方案。

**答案：**

```python
class OnlineLearner:
    """在线学习系统"""
    
    def __init__(self, model, buffer_size=10000, update_freq=1000):
        self.model = model
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        
        # 数据缓冲
        self.buffer = []
        
        # 统计
        self.request_count = 0
    
    def add_interaction(self, user_id, item_id, feedback):
        """添加交互"""
        self.buffer.append({
            'user_id': user_id,
            'item_id': item_id,
            'feedback': feedback,
            'timestamp': time.time()
        })
        
        # 缓冲区满时触发更新
        if len(self.buffer) >= self.buffer_size:
            self.update_model()
    
    def update_model(self):
        """更新模型"""
        if len(self.buffer) < self.update_freq:
            return
        
        # 采样训练数据
        train_data = random.sample(self.buffer, self.update_freq)
        
        # 微调模型
        self.model.finetune(train_data, epochs=1)
        
        # 清空缓冲
        self.buffer = self.buffer[-1000:]  # 保留最近1000条
        
        # 保存模型
        self.model.save('model_online.pt')
    
    def get_recommendations(self, user_id, context):
        """获取推荐"""
        self.request_count += 1
        
        # 定期更新
        if self.request_count % self.update_freq == 0:
            self.update_model()
        
        return self.model.recommend(user_id, context)
```

---

### Q43: 请设计特征存储方案。

**答案：**

```python
class FeatureStore:
    """特征存储系统"""
    
    def __init__(self, redis_client, mysql_client):
        self.redis = redis_client
        self.mysql = mysql_client
        
        # 缓存配置
        self.cache_ttl = 3600  # 1小时
    
    def get_user_features(self, user_id):
        """获取用户特征"""
        # 1. 尝试缓存
        cache_key = f'user_features:{user_id}'
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # 2. 从数据库获取
        features = self.mysql.query(
            "SELECT * FROM user_features WHERE user_id = %s",
            (user_id,)
        )
        
        if features:
            # 3. 写入缓存
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(features)
            )
        
        return features
    
    def get_item_features(self, item_id):
        """获取商品特征"""
        cache_key = f'item_features:{item_id}'
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        features = self.mysql.query(
            "SELECT * FROM item_features WHERE item_id = %s",
            (item_id,)
        )
        
        if features:
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(features)
            )
        
        return features
    
    def get_semantic_id(self, item_id):
        """获取语义ID"""
        cache_key = f'semantic_id:{item_id}'
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # 从预计算存储获取
        semantic_id = self.mysql.query(
            "SELECT semantic_id FROM item_semantic_ids WHERE item_id = %s",
            (item_id,)
        )
        
        if semantic_id:
            self.redis.setex(
                cache_key,
                self.cache_ttl * 24,  # 语义ID缓存更久
                json.dumps(semantic_id)
            )
        
        return semantic_id
    
    def update_features(self, item_id, features):
        """更新特征"""
        # 1. 更新数据库
        self.mysql.update(
            "UPDATE item_features SET features = %s WHERE item_id = %s",
            (json.dumps(features), item_id)
        )
        
        # 2. 更新缓存
        cache_key = f'item_features:{item_id}'
        self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(features)
        )
        
        # 3. 触发语义ID重计算
        self.recompute_semantic_id(item_id, features)
```

---

### Q44: 请设计A/B测试框架。

**答案：**

```python
class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self, db_client):
        self.db = db_client
        self.experiments = {}
    
    def create_experiment(self, name, control_model, treatment_model, 
                         traffic_ratio=0.5):
        """创建实验"""
        experiment = {
            'name': name,
            'control': control_model,
            'treatment': treatment_model,
            'traffic_ratio': traffic_ratio,
            'metrics': {
                'control': {'requests': 0, 'clicks': 0, 'conversions': 0},
                'treatment': {'requests': 0, 'clicks': 0, 'conversions': 0}
            },
            'start_time': time.time()
        }
        
        self.experiments[name] = experiment
        return experiment
    
    def get_model(self, experiment_name, user_id):
        """获取模型（分流）"""
        exp = self.experiments[experiment_name]
        
        # 确定性分流
        hash_value = int(hashlib.md5(f'{experiment_name}:{user_id}'.encode()).hexdigest(), 16)
        group = 'treatment' if hash_value % 100 < exp['traffic_ratio'] * 100 else 'control'
        
        exp['metrics'][group]['requests'] += 1
        
        return exp[group], group
    
    def record_feedback(self, experiment_name, user_id, group, feedback_type):
        """记录反馈"""
        exp = self.experiments[experiment_name]
        
        if feedback_type == 'click':
            exp['metrics'][group]['clicks'] += 1
        elif feedback_type == 'conversion':
            exp['metrics'][group]['conversions'] += 1
    
    def analyze(self, experiment_name):
        """分析结果"""
        exp = self.experiments[experiment_name]
        
        control = exp['metrics']['control']
        treatment = exp['metrics']['treatment']
        
        # 计算指标
        control_ctr = control['clicks'] / control['requests'] if control['requests'] > 0 else 0
        treatment_ctr = treatment['clicks'] / treatment['requests'] if treatment['requests'] > 0 else 0
        
        control_cvr = control['conversions'] / control['clicks'] if control['clicks'] > 0 else 0
        treatment_cvr = treatment['conversions'] / treatment['clicks'] if treatment['clicks'] > 0 else 0
        
        # 统计显著性
        ctr_significance = self.statistical_test(
            control['clicks'], control['requests'],
            treatment['clicks'], treatment['requests']
        )
        
        return {
            'control': {
                'ctr': control_ctr,
                'cvr': control_cvr,
                'requests': control['requests']
            },
            'treatment': {
                'ctr': treatment_ctr,
                'cvr': treatment_cvr,
                'requests': treatment['requests']
            },
            'improvement': {
                'ctr': (treatment_ctr - control_ctr) / control_ctr if control_ctr > 0 else 0,
                'cvr': (treatment_cvr - control_cvr) / control_cvr if control_cvr > 0 else 0
            },
            'significance': ctr_significance
        }
    
    def statistical_test(self, x1, n1, x2, n2):
        """统计显著性检验"""
        from scipy import stats
        
        # 比例检验
        p1 = x1 / n1
        p2 = x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p2 - p1) / se
        
        p_value = stats.norm.sf(abs(z)) * 2
        
        return {
            'z_score': z,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

---

## 九、开放性问题

### Q45: 如果让你重新设计这个系统，你会做哪些改进？

**答案：**

**1. 模型架构改进**

```
当前架构:
MMOE → RQ-VAE → Transformer → 预测

改进架构:
┌─────────────────────────────────────────────────────────────┐
│                    改进的模型架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  多模态编码                                                   │
│  ├── 文本: BERT → Adapter                                    │
│  ├── 视觉: ViT → Adapter                                     │
│  ├── 音频: Audio Encoder → Adapter (新增)                    │
│  └── 行为: Behavior Encoder → Adapter (新增)                 │
│                                                              │
│  层次化量化                                                   │
│  ├── 粗粒度层: 类别级别 (256)                                 │
│  ├── 中粒度层: 子类别级别 (1024)                              │
│  └── 细粒度层: 属性级别 (4096)                                │
│                                                              │
│  用户建模                                                     │
│  ├── 长期兴趣: 用户ID嵌入 + 历史序列                         │
│  ├── 短期兴趣: 实时行为序列                                   │
│  └── 上下文: 时间、位置、设备                                 │
│                                                              │
│  多任务学习                                                   │
│  ├── 点击预测                                                 │
│  ├── 购买预测                                                 │
│  └── 停留时长预测                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**2. 训练策略改进**

```python
# 改进的训练策略
improvements = {
    '预训练': {
        '数据增强': '对比学习 + 掩码预测',
        '多任务': '重建 + 对比 + 掩码',
        '规模': '更大规模预训练数据'
    },
    '微调': {
        '课程学习': '从简单到困难样本',
        '对比学习': 'InfoNCE + SupCon',
        '知识蒸馏': '大模型蒸馏到小模型'
    },
    '在线学习': {
        '增量更新': '实时更新用户表示',
        '主动学习': '选择信息量大的样本',
        '联邦学习': '保护隐私的分布式训练'
    }
}
```

**3. 工程改进**

```python
# 工程改进
engineering_improvements = {
    '推理优化': {
        '模型量化': 'INT8量化减少延迟',
        '模型剪枝': '移除冗余参数',
        '知识蒸馏': '小模型部署'
    },
    '系统架构': {
        '流式处理': 'Kafka + Flink实时处理',
        '向量检索': 'Faiss + GPU加速',
        '缓存策略': '多级缓存减少延迟'
    },
    '监控告警': {
        '实时监控': 'Prometheus + Grafana',
        '异常检测': '自动检测效果下降',
        'A/B测试': '自动化实验平台'
    }
}
```

---

### Q46: 这个项目最大的技术挑战是什么？如何解决的？

**答案：**

**挑战1: 码本坍塌**

```
问题:
- 大部分码本向量不被使用
- 码本利用率仅10-20%
- 表示能力严重浪费

解决方案:
1. EMA更新
   - 平滑更新码本向量
   - 避免梯度震荡

2. 死码重置
   - 检测长期未使用的码
   - 用活跃样本重置

3. 效果
   - 码本利用率提升到94-98%
```

**挑战2: 冷启动问题**

```
问题:
- 新商品无交互历史
- 传统ID嵌入无法处理
- 推荐系统无法推荐新商品

解决方案:
1. 语义ID
   - 基于商品特征生成
   - 不依赖交互历史

2. 相似性传递
   - 相似商品获得相似ID
   - 新商品继承相似商品的推荐能力

3. 效果
   - 新商品可被立即推荐
   - 冷启动商品召回率提升
```

**挑战3: 多模态融合**

```
问题:
- 文本和视觉特征差异大
- 简单拼接效果差
- 模态权重难以确定

解决方案:
1. MMOE架构
   - 多专家学习不同模式
   - 门控自适应选择

2. 模态解耦
   - 不同模态独立门控
   - 避免模态冲突

3. 效果
   - 多模态融合效果提升15%
```

**挑战4: 梯度回传**

```
问题:
- 量化操作不可微
- 梯度无法回传
- 端到端训练困难

解决方案:
1. STE直通估计器
   - 前向使用离散值
   - 反向梯度直通

2. 软量化
   - Gumbel-Softmax
   - 可微分采样

3. 效果
   - 实现端到端训练
   - 微调效果提升
```

---

### Q47: 如何评估这个系统的商业价值？

**答案：**

**1. 业务指标评估**

```python
def calculate_business_value():
    """计算商业价值"""
    
    # 假设数据
    daily_active_users = 1_000_000
    avg_recommendations_per_user = 20
    baseline_ctr = 0.02  # 2%
    improved_ctr = 0.025  # 2.5% (提升25%)
    avg_order_value = 100  # 平均订单金额
    
    # 计算提升
    additional_clicks = daily_active_users * avg_recommendations_per_user * (improved_ctr - baseline_ctr)
    additional_conversions = additional_clicks * 0.1  # 假设10%转化率
    daily_revenue_increase = additional_conversions * avg_order_value
    
    return {
        '日增点击': additional_clicks,
        '日增转化': additional_conversions,
        '日增收': daily_revenue_increase,
        '月增收': daily_revenue_increase * 30,
        '年增收': daily_revenue_increase * 365
    }

# 结果
# 日增点击: 100,000
# 日增转化: 10,000
# 日增收: 1,000,000
# 月增收: 30,000,000
# 年增收: 365,000,000
```

**2. ROI计算**

```python
def calculate_roi():
    """计算ROI"""
    
    # 收益
    annual_revenue_increase = 365_000_000
    
    # 成本
    development_cost = 5_000_000  # 开发成本
    infrastructure_cost = 10_000_000  # 基础设施成本
    maintenance_cost = 2_000_000  # 年维护成本
    
    total_cost = development_cost + infrastructure_cost + maintenance_cost
    
    # ROI
    roi = (annual_revenue_increase - total_cost) / total_cost
    
    return {
        '年收益': annual_revenue_increase,
        '总成本': total_cost,
        'ROI': roi,
        '回本周期': total_cost / (annual_revenue_increase / 12)  # 月
    }

# 结果
# ROI: 23.3 (2330%)
# 回本周期: 0.2个月
```

**3. 价值维度**

| 维度 | 指标 | 提升 |
|------|------|------|
| 用户体验 | 点击率 | +25% |
| 用户体验 | 停留时长 | +15% |
| 商业价值 | GMV | +20% |
| 商业价值 | 转化率 | +10% |
| 技术价值 | 冷启动覆盖率 | +50% |
| 技术价值 | 推理延迟 | -30% |

---

### Q48: 你从这个项目中学到了什么？

**答案：**

**1. 技术层面**

```
学到的技术知识:
├── 深度学习
│   ├── 向量量化原理
│   ├── 残差学习
│   ├── EMA更新机制
│   └── 梯度回传技巧
│
├── 推荐系统
│   ├── 序列推荐
│   ├── 多模态融合
│   ├── 冷启动处理
│   └── 评估指标
│
└── 工程实践
    ├── 大规模特征处理
    ├── 模型部署优化
    ├── A/B测试框架
    └── 监控告警
```

**2. 方法论层面**

```
解决问题的方法论:
├── 问题分析
│   ├── 明确问题定义
│   ├── 分析问题根源
│   └── 确定解决方向
│
├── 方案设计
│   ├── 调研现有方法
│   ├── 设计技术方案
│   └── 评估可行性
│
├── 实验验证
│   ├── 设计实验
│   ├── 分析结果
│   └── 迭代优化
│
└── 工程落地
    ├── 代码实现
    ├── 测试验证
    └── 部署上线
```

**3. 团队协作**

```
团队协作经验:
├── 代码规范
│   ├── 统一代码风格
│   ├── 代码审查
│   └── 文档规范
│
├── 版本管理
│   ├── Git工作流
│   ├── 分支管理
│   └── 发布流程
│
└── 沟通协作
    ├── 技术方案评审
    ├── 进度同步
    └── 问题追踪
```

---

### Q49: 如果要在生产环境部署，还需要做哪些工作？

**答案：**

**1. 模型优化**

```python
# 模型优化清单
model_optimization = {
    '模型压缩': [
        'INT8量化减少模型大小',
        '知识蒸馏到小模型',
        '模型剪枝移除冗余参数'
    ],
    '推理优化': [
        'ONNX导出加速推理',
        'TensorRT优化GPU推理',
        '批处理优化吞吐量'
    ],
    '缓存策略': [
        '预计算商品语义ID',
        '缓存用户表示',
        '多级缓存减少延迟'
    ]
}
```

**2. 系统架构**

```python
# 系统架构改进
system_architecture = {
    '高可用': [
        '多副本部署',
        '负载均衡',
        '故障自动转移'
    ],
    '高性能': [
        '异步处理',
        '连接池优化',
        '内存优化'
    ],
    '可扩展': [
        '微服务架构',
        '容器化部署',
        '自动扩缩容'
    ]
}
```

**3. 监控告警**

```python
# 监控告警配置
monitoring = {
    '系统监控': [
        'CPU/内存使用率',
        '请求延迟',
        '错误率'
    ],
    '业务监控': [
        '推荐点击率',
        '推荐转化率',
        '用户停留时长'
    ],
    '模型监控': [
        '预测分数分布',
        '特征分布漂移',
        '模型效果衰减'
    ]
}
```

**4. 安全合规**

```python
# 安全合规要求
security = {
    '数据安全': [
        '数据加密存储',
        '敏感信息脱敏',
        '访问权限控制'
    ],
    '隐私保护': [
        '用户数据匿名化',
        '联邦学习方案',
        '差分隐私'
    ],
    '合规要求': [
        'GDPR合规',
        '数据留存策略',
        '审计日志'
    ]
}
```

---

### Q50: 请总结这个项目的核心贡献和局限性。

**答案：**

**核心贡献：**

```
1. 技术贡献
├── 提出语义ID生成方法
│   └── 解决冷启动语义孤岛问题
│
├── EMA码本更新 + 死码重置
│   └── 码本利用率从10%提升到94%+
│
├── 多模态融合架构
│   └── MMOE实现文本+视觉融合
│
└── 端到端训练框架
    └── STE实现梯度回传

2. 工程贡献
├── 完整的实现代码
├── 详细的文档
└── 可复现的实验

3. 实践贡献
├── 冷启动解决方案
├── 多模态推荐实践
└── 工程部署经验
```

**局限性：**

```
1. 数据依赖
├── 需要高质量多模态特征
├── 特征质量影响效果
└── 特征提取成本高

2. 用户建模
├── 用户冷启动未完全解决
├── 需要一定交互历史
└── 用户表示更新不及时

3. 计算开销
├── 多层量化增加计算量
├── 大规模部署需要优化
└── 实时性要求高时受限

4. 泛化能力
├── 跨领域迁移需要重新预训练
├── 特征提取器需要适配
└── 新场景需要调优
```

**未来方向：**

```
1. 模型改进
├── 更强的多模态融合
├── 更好的用户建模
└── 更高效的推理

2. 数据增强
├── 更多模态特征
├── 更丰富的上下文
└── 更好的数据增强

3. 系统优化
├── 在线学习
├── 联邦学习
└── 自动化运维
```

---

*面试题文档完成*
*总计: 50道面试题*
*涵盖: 项目背景、算法原理、模型架构、训练策略、评估指标、工程实现、代码实现、系统设计、开放性问题*
