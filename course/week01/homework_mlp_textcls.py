# -*- coding: utf-8 -*-
"""
第 1 周作业：手写 MLP 中文情感分类（纯 tensor + autograd）
==========================================================
目标：把「forward → loss → backward → step → zero_grad」训练五步写进肌肉记忆，
     并亲手验证 Softmax+交叉熵、链式法则、mini-batch 这些讲义概念。

环境：python3 + torch（本沙盒已装好；本地跑则 pip install torch）
规则（重要）：
  1. 全程禁止使用 nn.Module / nn.Linear / optim 包——本周只用 tensor + autograd
  2. 允许用 torch.nn.functional.cross_entropy 做数值对照（任务 4）
  3. 按任务顺序闯关，每过一个 assert 就是一关；卡住超过半天就标记跳过、周末清障
提交：全部通关后，把运行输出贴给老师，并完成文末 3 道自测题
"""
import random

import torch
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# ----------------------------------------------------------------------
# 数据（已提供，不用改）：模拟酒店评论，好评=1 / 差评=0
# 特意让两类共享大部分"方面词"（房间/服务/...），模型必须学会区分观点词
# ----------------------------------------------------------------------
ASPECTS = ["房间", "服务", "早餐", "位置", "环境", "前台", "卫生", "床品"]
POS_WORDS = ["很好", "干净", "热情", "方便", "满意", "贴心", "舒适", "周到", "不错", "棒"]
NEG_WORDS = ["很脏", "太吵", "差评", "简陋", "失望", "冷淡", "糟糕", "坑人", "不行", "后悔"]


def sample_text(words):
    k = random.randint(1, 3)
    return "，".join(random.choice(ASPECTS) + random.choice(words) for _ in range(k))


def make_dataset(n_per_class):
    texts, labels = [], []
    for _ in range(n_per_class):
        texts.append(sample_text(POS_WORDS)); labels.append(1)
        texts.append(sample_text(NEG_WORDS)); labels.append(0)
    pairs = list(zip(texts, labels))
    random.shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def tokenize(text):
    """字符级分词：去掉逗号，逐字切分。
    思考：字符级 vs 词级 vs 子词（BPE）各有什么优劣？这是第 1 篇后面会讲的。"""
    return [ch for ch in text if ch != "，"]


# ----------------------------------------------------------------------
# 任务 1（热身）：构建词表
# 要求：返回 dict {字符: 序号}，<unk> 固定占 0 号，词表只由训练集构建
# 为什么只用训练集？——提示：验证集里出现没见过的字符怎么办？(OOV 概念)
# ----------------------------------------------------------------------
def build_vocab(texts):
    # 初始化 vocab = {"<unk>": 0}
    vocab = {"<unk>": 0}    
    # 遍历每个文本的 tokenize 结果，把没见过的字符加入词表
    for text in texts:
        for ch in tokenize(text):
            if ch not in vocab:
                vocab[ch] = len(vocab) 
    return vocab    

# ----------------------------------------------------------------------
# 任务 2（热身）：文本 → 词袋向量（Bag of Words）
# 要求：返回形状 [len(vocab)] 的 float 张量；每个字符在对应位置计数
# OOV 字符计入 <unk>（0 号位）
# ----------------------------------------------------------------------
def text_to_bow(text, vocab):
    # torch.zeros(len(vocab)) 建零向量
    bow = torch.zeros(len(vocab))
    chars = tokenize(text)
    #  逐字符查词表计数，查不到的用 vocab["<unk>"]
    for ch in chars:
        idx = vocab.get(ch,"<unk>")
        bow[idx] += 1
    return bow
    # 思考：BoW 丢掉了什么信息？（"很好"和"好很"在这个表示下有区别吗？）


# ----------------------------------------------------------------------
# 任务 3（核心）：手动创建参数 + 手写 MLP 前向传播
# 模型：input[V] → ReLU(x@W1+b1) → @W2+b2 → logits[2]
# 禁止使用 nn.Linear / F.relu，ReLU 用 torch.clamp_min(h, 0.0) 代替
# ----------------------------------------------------------------------
def init_params(V, H=32, C=2):
    """创建 W1[V,H], b1[H], W2[H,C], b2[C]，全部 requires_grad=True。
    提示（可照抄风格）：
        g = torch.Generator().manual_seed(0)
        W1 = (torch.randn(V, H, generator=g) * 0.1).requires_grad_(True)
        b1 = torch.zeros(H, requires_grad=True)
    问题：W 为什么要 * 0.1？b 为什么初始化成 0？
    """
    # TODO: 按提示创建 4 个参数并 return W1, b1, W2, b2
    raise NotImplementedError


def mlp_forward(X, W1, b1, W2, b2):
    """X: [B, V] → 返回 logits: [B, 2]"""
    # TODO: 第一层线性 + ReLU + 第二层线性
    #       h = clamp_min(X @ W1 + b1, 0.0)
    #       logits = ?
    raise NotImplementedError


# ----------------------------------------------------------------------
# 任务 4（核心）：手写 Softmax + 交叉熵（batch 版，取均值）
# 输入 logits [B,2]、target [B]（0/1），输出标量 loss
# 三步提示：
#   ① 数值稳定：shifted = logits - 每行 max（为什么要减 max？先想再写）
#   ② log_softmax = shifted - log(sum(exp(shifted)))   （按行）
#   ③ loss = -log_softmax[每个样本的 target 位置] 的均值
# 完成后 main 里会自动和 F.cross_entropy 对数，误差必须 < 1e-6
# ----------------------------------------------------------------------
def my_cross_entropy(logits, target):
    # TODO: 实现三步
    raise NotImplementedError


# ----------------------------------------------------------------------
# 任务 5（核心★）：训练循环五步 + 评估
# 这是本周的肌肉记忆目标：合上讲义也能默写
# ----------------------------------------------------------------------
def evaluate(X, y, params):
    """返回 (准确率, 预测标签)。提醒：评估不需要梯度，用 torch.no_grad()。"""
    # TODO: forward → argmax → 与 y 比较求 mean
    #       同时返回 argmax 结果（任务 6 要用）
    raise NotImplementedError


def train(X_train, y_train, X_val, y_val, params, lr=0.5, batch_size=16, epochs=8):
    """mini-batch 训练。五步循环模板（把伪代码变成真代码）：
        每个 epoch：torch.randperm 打乱样本顺序
        每个批次：
            ① forward     logits = mlp_forward(X_batch, *params)
            ② loss        loss  = my_cross_entropy(logits, y_batch)
            ③ backward    loss.backward()          # autograd 沿计算图算出每个 p.grad
            ④ step        在 torch.no_grad() 里：p -= lr * p.grad（每个参数）
            ⑤ zero_grad   p.grad = None（每个参数）
        问题（自测题 1）：⑤ 如果忘写会发生什么？
    每轮结束打印：平均 loss、train_acc、val_acc
    """
    # TODO: 实现训练循环，最后 return
    raise NotImplementedError


# ----------------------------------------------------------------------
# 任务 6（选做）：正类（好评）的 Precision / Recall / F1
# 用 evaluate 返回的预测标签统计：TP / FP / FN，再套公式
# 这个指标会在第 4-5 周的项目里反复出现，现在手算一遍印象最深
# ----------------------------------------------------------------------
def positive_prf(y_true, y_pred):
    # TODO: tp = ((y_pred==1)&(y_true==1)).sum().item()，fp、fn 同理
    # TODO: precision = tp/(tp+fp)，recall = tp/(tp+fn)，f1 = 2PR/(P+R)
    # 注意分母为 0 的保护
    raise NotImplementedError


# ----------------------------------------------------------------------
# 闯关主线（不用改）：从上往下跑，卡在哪一关就修哪一关
# ----------------------------------------------------------------------
def main():
    train_texts, train_labels = make_dataset(400)
    val_texts, val_labels = make_dataset(100)
    
    vocab = build_vocab(train_texts)                                   # 关 1
    assert vocab["<unk>"] == 0 and len(vocab) > 50, "任务 1 未通过"
    print(f"[关 1] 词表大小 = {len(vocab)}")
    
    bow = text_to_bow(train_texts[0], vocab)                           # 关 2
    assert bow.shape == (len(vocab),) and bow.sum().item() == len(tokenize(train_texts[0])), "任务 2 未通过"
    X_train = torch.stack([text_to_bow(t, vocab) for t in train_texts])
    y_train = torch.tensor(train_labels)
    X_val = torch.stack([text_to_bow(t, vocab) for t in val_texts])
    y_val = torch.tensor(val_labels)
    print(f"[关 2] train {tuple(X_train.shape)} / val {tuple(X_val.shape)}（列数=词表大小）")

    params = init_params(X_train.shape[1])                             # 关 3
    W1, b1, W2, b2 = params
    demo = mlp_forward(X_train[:4], W1, b1, W2, b2)
    assert demo.shape == (4, 2), "任务 3 未通过：输出形状应为 [4, 2]"
    print(f"[关 3] 前向传播 OK，logits 形状 {tuple(demo.shape)}")

    demo_logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])               # 关 4
    demo_target = torch.tensor([0, 1])
    mine = my_cross_entropy(demo_logits, demo_target)
    ref = F.cross_entropy(demo_logits, demo_target)
    assert torch.allclose(mine, ref, atol=1e-6), f"任务 4 未通过：{mine.item()} vs {ref.item()}"
    print(f"[关 4] 手写 CE = {mine.item():.6f}，与 F.cross_entropy 完全一致")

    train(X_train, y_train, X_val, y_val, params)                      # 关 5
    val_acc, val_pred = evaluate(X_val, y_val, params)
    assert val_acc > 0.90, f"任务 5 未通过：val_acc={val_acc:.4f} < 0.90"
    print(f"[关 5] 训练五步 OK，val_acc = {val_acc:.4f}")

    p, r, f1 = positive_prf(y_val, val_pred)                           # 关 6（选做）
    print(f"[关 6] 正类 P={p:.4f} R={r:.4f} F1={f1:.4f}")
    print("\nPASS：全部检查点通过，把输出贴给老师吧！")


if __name__ == "__main__":
    main()

# ======================================================================
# 交给老师前，先回答这 3 道自测题（写在提交消息里即可）：
#   1. 训练五步是哪五步？忘了第 ⑤ 步 zero_grad 会发生什么？
#   2. 手写交叉熵时为什么要先减去每行的 max？不减会出什么问题？
#   3. 如果训练集里 95% 是差评，一个无脑全预测"差评"的模型准确率是多少？
#      它的 Precision/Recall/F1（正类=好评）呢？这说明 Accuracy 有什么坑？
# ======================================================================
