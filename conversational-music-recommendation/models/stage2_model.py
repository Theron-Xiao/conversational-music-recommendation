"""
Stage 2: Cross-Attention Re-Ranker — 精排序模型。

Query（文本+用户+历史）通过 Cross-Attention 对 K 个候选曲目打分。
训练时用 LambdaRank ApproxNDCG 损失直接优化排序质量。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from config.default_config import (
    TEXT_DIM, AUDIO_DIM, USER_EMB_DIM,
    STAGE2_D_MODEL, STAGE2_NUM_HEADS, STAGE2_NUM_LAYERS,
    STAGE2_DROPOUT, STAGE2_MLP_HIDDEN,
)


class CrossAttentionBlock(nn.Module):
    """单层 Cross-Attention + FFN + Residual + LayerNorm。"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, cand):
        """
        Args:
            query: (B, 1, d_model) — 上下文 query
            cand:  (B, K, d_model) — 候选特征

        Returns:
            (B, K, d_model) — query-attended 候选表示
        """
        # Cross-Attention: query attends to candidates
        attn_out, _ = self.cross_attn(query=query, key=cand, value=cand)
        cand = self.norm1(cand + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(cand)
        cand = self.norm2(cand + self.dropout(ffn_out))
        return cand


class CrossAttentionReranker(nn.Module):
    """
    Cross-Attention 精排序模型。

    输入:
      - text_emb:       (B, 384)   上下文文本嵌入
      - user_emb:       (B, 128)   用户偏好嵌入
      - banned_center:  (B, 512)   历史曲目均值嵌入
      - cand_audio:     (B, K, 512) 候选曲目音频嵌入
      - cand_thought:   (B, K, 384) 候选曲目 thought 嵌入

    输出:
      - scores: (B, K)  每个候选的相关性分数
    """

    def __init__(
        self,
        text_dim=TEXT_DIM,
        user_emb_dim=USER_EMB_DIM,
        history_dim=AUDIO_DIM,
        audio_dim=AUDIO_DIM,
        thought_dim=TEXT_DIM,
        d_model=STAGE2_D_MODEL,
        num_heads=STAGE2_NUM_HEADS,
        num_layers=STAGE2_NUM_LAYERS,
        dropout=STAGE2_DROPOUT,
        mlp_hidden=STAGE2_MLP_HIDDEN,
    ):
        super().__init__()
        self.d_model = d_model

        query_in_dim = text_dim + user_emb_dim + history_dim  # 384 + 128 + 512 = 1024
        cand_in_dim = audio_dim + thought_dim                  # 512 + 384 = 896

        # Query 投影
        self.query_proj = nn.Sequential(
            nn.Linear(query_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Candidate 投影
        self.cand_proj = nn.Sequential(
            nn.Linear(cand_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 堆叠 Cross-Attention 层
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 打分 MLP
        self.scorer = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, text_emb, user_emb, banned_center, cand_audio, cand_thought):
        """
        Args:
            text_emb:       (B, text_dim)       上下文向量
            user_emb:       (B, user_emb_dim)   用户向量
            banned_center:  (B, history_dim)    历史中心向量
            cand_audio:     (B, K, audio_dim)   候选音频向量
            cand_thought:   (B, K, thought_dim) 候选 thought 向量

        Returns:
            scores: (B, K)  每个候选的得分（越高越相关）
        """
        B, K = cand_audio.shape[:2]

        # 1. 构建 Query
        query = torch.cat([text_emb, user_emb, banned_center], dim=-1)  # (B, 1024)
        query = self.query_proj(query).unsqueeze(1)                      # (B, 1, d_model)

        # 2. 构建 Candidate 特征
        cand = torch.cat([cand_audio, cand_thought], dim=-1)  # (B, K, 896)
        cand = self.cand_proj(cand)                            # (B, K, d_model)

        # 3. 多层 Cross-Attention
        for layer in self.cross_attn_layers:
            cand = layer(query, cand)  # (B, K, d_model)

        # 4. 打分
        scores = self.scorer(cand).squeeze(-1)  # (B, K)
        return scores


# ── 形状验证 ──
if __name__ == '__main__':
    print("==== CrossAttentionReranker 形状验证 ====\n")

    B, K = 4, 20
    model = CrossAttentionReranker()
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 构造假输入
    text_emb = torch.randn(B, TEXT_DIM)
    user_emb = torch.randn(B, USER_EMB_DIM)
    banned_center = torch.randn(B, AUDIO_DIM)
    cand_audio = torch.randn(B, K, AUDIO_DIM)
    cand_thought = torch.randn(B, K, TEXT_DIM)

    # Forward
    model.eval()
    with torch.no_grad():
        scores = model(text_emb, user_emb, banned_center, cand_audio, cand_thought)

    print(f"输入: text={text_emb.shape}, user={user_emb.shape}, banned={banned_center.shape}")
    print(f"候选: audio={cand_audio.shape}, thought={cand_thought.shape}")
    print(f"输出: scores={scores.shape}")
    assert scores.shape == (B, K), f"期望 (B={B}, K={K})，实际 {scores.shape}"
    print("\n[OK] 形状验证通过！")

    # 梯度验证
    model.train()
    scores = model(text_emb, user_emb, banned_center, cand_audio, cand_thought)
    loss = scores.mean()
    loss.backward()
    print(f"[OK] 梯度验证通过！loss={loss.item():.4f}")
