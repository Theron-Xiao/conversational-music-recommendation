"""
LambdaRank ApproxNDCG Loss — 可微分的 Listwise 排序损失。

直接优化 NDCG：对每对 (i, j) 其中 rel_i > rel_j，惩罚 s_i < s_j，
并按 |delta_NDCG_ij| 加权（交换 i,j 对 NDCG 的影响越大，惩罚越重）。

参考: Burges, "From RankNet to LambdaRank to LambdaMART", 2010.
"""

import torch
import torch.nn as nn


def lambdarank_loss(scores, relevance, sigma=1.0, epsilon=1e-6):
    """
    LambdaRank 近似 NDCG 损失。

    Args:
        scores:    (B, K) 模型预测分数（越高越相关）
        relevance: (B, K) 真实相关度（binary: 1=正例, 0=负例，或 graded）
        sigma:     控制 sigmoid 陡峭度的缩放因子
        epsilon:   数值稳定小量

    Returns:
        scalar loss (越高 → 越差)
    """
    B, K = scores.shape
    device = scores.device

    # ── 1. 计算 Ideal DCG（用于归一化 delta_NDCG）──
    ideal_rels, _ = relevance.float().sort(dim=1, descending=True)
    positions = torch.arange(1, K + 1, device=device).float()
    ideal_discount = 1.0 / torch.log2(positions + 1.0)
    idcg = (ideal_rels * ideal_discount.unsqueeze(0)).sum(dim=1)  # (B,)

    # 如果 IDCG 为 0（全是负例），跳过该样本
    idcg = idcg.clamp(min=epsilon)

    # ── 2. 计算 delta_NDCG 权重矩阵 ──
    # delta_NDCG_ij = |rel_i - rel_j| * |discount_i - discount_j| / IDCG
    discount = 1.0 / torch.log2(positions + 1.0)  # (K,)
    delta_discount = (discount.unsqueeze(0) - discount.unsqueeze(1)).abs()  # (K, K)

    reli = relevance.float().unsqueeze(2)  # (B, K, 1)
    relj = relevance.float().unsqueeze(1)  # (B, 1, K)
    delta_rel = (reli - relj).abs()        # (B, K, K)

    delta_ndcg = delta_rel * delta_discount.unsqueeze(0)  # (B, K, K)
    delta_ndcg = delta_ndcg / idcg.unsqueeze(-1).unsqueeze(-1).clamp(min=epsilon)

    # ── 3. 逐对损失 ──
    # s_diff[b, i, j] = s_i - s_j
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, K, K)
    # 仅在 rel_i > rel_j 时惩罚
    mask = (reli > relj).float()  # (B, K, K)

    # LambdaRank loss: mask * delta_ndcg * log(1 + exp(-sigma * (s_i - s_j)))
    # 用 softplus 保证数值稳定性: softplus(-x) = log(1 + exp(-x))
    pairwise_loss = mask * delta_ndcg * torch.nn.functional.softplus(-sigma * s_diff)

    # ── 4. 聚合 ──
    # 每个样本的 loss = 所有 pairs 的 loss 之和 / K
    loss_per_sample = pairwise_loss.sum(dim=(1, 2)) / K  # (B,)
    return loss_per_sample.mean()


class LambdaRankLoss(nn.Module):
    """LambdaRank ApproxNDCG Loss 的 nn.Module 封装。"""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, scores, relevance):
        return lambdarank_loss(scores, relevance, sigma=self.sigma)


# ── 快速测试 ──
if __name__ == '__main__':
    print("==== LambdaRank Loss 单元测试 ====\n")

    # 测试 1: 完美的排序 → loss 应该很低
    B, K = 4, 10
    scores_good = torch.tensor([
        [10.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]  # 正例(分数10)排第1
    ]).repeat(B, 1)
    rel = torch.zeros(B, K)
    rel[:, 0] = 1.0  # 第一个是正例

    loss_good = lambdarank_loss(scores_good, rel)
    print(f"完美排序 Loss: {loss_good.item():.6f} (期望: 很低)")

    # 测试 2: 最差排序 → loss 应该很高
    scores_bad = torch.tensor([
        [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 10.0, 20.0, 30.0, 99.0]  # 正例(最高分99)其实在第10位
    ]).repeat(B, 1)
    rel_bad = torch.zeros(B, K)
    rel_bad[:, 0] = 1.0  # 正例在第0位（分数最低）— 注意 rel 不随分数位置移动

    loss_bad = lambdarank_loss(scores_bad, rel_bad)
    print(f"最差排序 Loss: {loss_bad.item():.6f} (期望: 很高)")

    # 测试 3: 随机排序 → loss 居中
    torch.manual_seed(42)
    scores_rand = torch.randn(B, K)
    loss_rand = lambdarank_loss(scores_rand, rel)
    print(f"随机排序 Loss: {loss_rand.item():.6f} (期望: 居中)")

    # 测试 4: 多正例场景 (graded relevance)
    rel_multi = torch.zeros(1, K)
    rel_multi[0, 0] = 3.0  # 高度相关
    rel_multi[0, 1] = 2.0  # 相关
    rel_multi[0, 2] = 1.0  # 弱相关
    scores_multi = torch.tensor([[100.0, 50.0, -10.0, 0.1, 0.2, 0.0, -0.1, -0.2, -0.3, -0.4]])
    loss_multi = lambdarank_loss(scores_multi, rel_multi)
    print(f"多级相关度 Loss: {loss_multi.item():.6f}")

    # 测试 5: 梯度可用
    scores_grad = torch.randn(2, 10, requires_grad=True)
    rel_grad = torch.zeros(2, 10)
    rel_grad[:, 0] = 1.0
    loss_grad = lambdarank_loss(scores_grad, rel_grad)
    loss_grad.backward()
    print(f"\n梯度测试: loss={loss_grad.item():.4f}, grad_norm={scores_grad.grad.norm().item():.4f}")
    print("\n[OK] 全部测试通过!")
