# 日常总结 · Day 10/20

> 2026-10-20 ｜ 今日内容：Python 字典取值、PyTorch 张量维度与广播、Git 提交规范

## 一、Python 字典查值

| 写法 | 查不到时 |
| --- | --- |
| `vocab[ch]` | 直接报错 `KeyError` |
| `vocab.get(ch, 默认值)` | 不报错，返回给定的默认值 |

**要点**：不确定 key 是否存在时，用 `.get()` 更安全。

## 二、PyTorch：张量与 shape

- `tensor` 就是多维数组，`shape` 描述它有几行几列。
- `B` 是 batch size（一次处理几条文本），`C=2` 是类别数。

| shape | 含义 |
| --- | --- |
| `[4, 2]` | 4 行 2 列：4 条样本 × 每条 2 个类别得分 |
| `[4, 1]` | 4 行 1 列：4 条样本 × 每条 1 个数 |
| `[4]` | 一维，4 个数 |
| `[]` | 标量，单个数字（如 loss） |

### 二维张量的 dim 方向

- `dim=0`：竖着 / 跨行，每一列从上往下合并 → 结果长度 = 列数。
- `dim=1`：横着 / 跨列，每一行从左往右合并 → 结果长度 = 行数。

### shape[0] 与 shape[1]

以 `X_train.shape = torch.Size([800, 54])` 为例：

- 第 0 维 `800` = 行数 = 样本数；
- 第 1 维 `54` = 列数 = 每个样本的特征维度（词表大小）。

| 写法 | 取的是什么 | 值 |
| --- | --- | --- |
| `X_train.shape` | 整个形状 | `torch.Size([800, 54])` |
| `X_train.shape[0]` | 第 0 维大小 = 行数 | `800` |
| `X_train.shape[1]` | 第 1 维大小 = 列数 | `54` |
| `X_train[0]` | 第 0 行（第一条样本） | `[0, 1, 1, 0, ..., 0]`（长度 54） |

**易混点**：

- `shape[0]` 问的是“第 0 个维度有多大” → `800`；
- `X_train[0]` 是“取第 0 个元素” → 第一条样本向量；
- 两者完全不是一回事。

同理，`logits.shape[0]` = 4（行数 = batch 大小 B），`logits.shape[1]` = 2（列数 = 类别数）。

## 三、PyTorch：max、argmax 与 keepdim

### max(dim=1) 返回两个值

```python
result = logits.max(dim=1)
result.values   # 最大值本身
result.indices  # 最大值在第几列
```

- `tensor.max(dim=1)` 返回 `(最大值, 最大值的位置索引)`。
- `tensor.argmax(dim=1)` 只返回最大值的位置索引。
- 只要最大值本身时用 `.values`；不用 `.values` 会拿到一个 tuple，无法直接参与运算。

### keepdim=True 的作用

- 不加 `keepdim`：`max(dim=1)` 的结果是 `[4]`。
- 加上 `keepdim=True`：结果是 `[4, 1]`，保留被压掉的维度，但长度变为 1。

**为什么必须加**：下一步要算 `logits - row_max`，即 `[4, 2] - [4, 1]`。

- PyTorch 广播从最右边的维度开始对齐；
- `[4, 1]` 的第 1 列复制 2 次 → `[4, 2]`，再逐元素相减；
- 效果是每一行的两个数都减去这一行的最大值：
  - `[0.39, 0.70] - 0.70 → [-0.31, 0]`
  - `[-1.20, 0.50] - 0.50 → [-1.70, 0]`
- 不加 `keepdim` 得到 `[4]`，`[4, 2] - [4]` 最右维 2 和 4 对不上，直接报错。

**一句话**：`keepdim=True` 是为了让后面的减法能正确广播。

## 四、Git 提交信息规范

`commit -m` 后用类型前缀说明改动性质：

| 类型 | 含义 | 举例 |
| --- | --- | --- |
| `feat` | 新功能 | `feat: 实现 MLP 前向传播` |
| `fix` | 修 bug | `fix: 修复交叉熵数值溢出` |
| `docs` | 文档 / 注释 | `docs: 补充词表构建注释` |
| `refactor` | 重构（不改行为） | `refactor: 提取 tokenize 函数` |
| `test` | 测试 | `test: 添加 evaluate 单元测试` |
| `chore` | 杂项（构建 / 依赖 / 配置） | `chore: 添加 .gitignore` |
| `perf` | 性能优化 | `perf: 优化 batch 矩阵运算` |

## 五、今日重点回顾

1. 字典取值：`[]` 可能抛 `KeyError`，`.get(key, default)` 更安全。
2. `shape` 要结合“行 / 列、样本 / 特征”理解；`shape[0]` 是维度大小，不是取第 0 条数据。
3. `max(dim=1)` 同时返回值和索引，`argmax` 只返回索引；只取最大值用 `.values`。
4. `keepdim=True` 保留长度为 1 的维度，核心目的是让广播成立。
5. Git 提交按 `feat / fix / docs / refactor / test / chore / perf` 分类写清楚。
