"""
两阶段评估 Pipeline：Stage 1 粗排 → Stage 2 精排 → MRR。

对比:
  - Stage 1 Only: 纯 thought 检索的 MRR（全量 43K 候选）
  - Two-Stage:   Stage 1 Top-K + Stage 2 Cross-Attention 重排序
  - Baseline LR:  Text+User → Linear → Audio（1000 候选池）

用法:
  python pipeline.py
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from config.default_config import (
    PROXY_ENV,
    DEVICE,
    TEXT_DIM, AUDIO_DIM, USER_EMB_DIM,
    STAGE2_D_MODEL, STAGE2_NUM_HEADS, STAGE2_NUM_LAYERS,
    STAGE2_DROPOUT, STAGE2_MLP_HIDDEN, STAGE2_K_RETRIEVAL,
)
from data.dataset import load_preprocessed_data
from data.loader import load_track_vectors, load_user_embeddings
from models.stage2_model import CrossAttentionReranker
from stage1_thought_retrieval import load_thought_index, retrieve_topk, evaluate as stage1_evaluate
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


@torch.no_grad()
def evaluate_two_stage(model, eval_df, track_vectors, thought_map,
                       thought_ids, thought_matrix, user_emb_dict,
                       K=STAGE2_K_RETRIEVAL, device=DEVICE, desc="Two-Stage"):
    """
    两阶段评估: Stage 1 召回 → Stage 2 重排序 → MRR。

    对每个样本:
      1. Stage 1 检索 Top-K
      2. 确保 ground truth 在候选集中
      3. Stage 2 打分并重排序
      4. 计算 MRR
    """
    model.eval()
    total_mrr = 0.0
    total_ndcg10 = 0.0
    count = 0
    hit_at_1 = 0
    hit_at_10 = 0

    for idx in tqdm(range(len(eval_df)), desc=desc):
        sample = eval_df.iloc[idx]
        true_track = sample['target_track_id']

        if true_track not in track_vectors:
            continue

        context_text = sample['context_text']

        # ── Stage 1: 召回 Top-K ──
        topk_ids, stage1_scores = retrieve_topk(
            context_text, thought_ids, thought_matrix, K=K, device=device
        )

        # 确保 ground truth 在候选集中
        if true_track not in topk_ids:
            topk_ids.insert(0, true_track)
            topk_ids = topk_ids[:K]

        K_actual = len(topk_ids)

        # ── 构建 query 特征 ──
        text_emb = get_text_embedding(context_text)
        uid = sample.get('user_id')
        user_emb = user_emb_dict.get(uid, np.zeros(USER_EMB_DIM, dtype=np.float32))

        history_tracks = sample.get('history_tracks', []) or []
        banned_vecs = [np.asarray(track_vectors[tid], dtype=np.float32)
                       for tid in history_tracks if tid in track_vectors]
        banned_center = (np.mean(np.stack(banned_vecs), axis=0)
                         if banned_vecs else np.zeros(AUDIO_DIM, dtype=np.float32))

        # ── 构建候选特征 ──
        cand_audio_list, cand_thought_list = [], []
        for tid in topk_ids:
            a_vec = np.asarray(track_vectors.get(tid, np.zeros(AUDIO_DIM)), dtype=np.float32)
            t_vec = thought_map.get(tid, np.zeros(TEXT_DIM, dtype=np.float32))
            cand_audio_list.append(a_vec)
            cand_thought_list.append(t_vec)

        cand_audio = np.stack(cand_audio_list)
        cand_thought = np.stack(cand_thought_list)

        # 转 tensor
        text_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)
        user_t = torch.tensor(np.asarray(user_emb, dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device)
        banned_t = torch.tensor(banned_center, dtype=torch.float32).unsqueeze(0).to(device)
        cand_a_t = torch.tensor(cand_audio, dtype=torch.float32).unsqueeze(0).to(device)
        cand_t_t = torch.tensor(cand_thought, dtype=torch.float32).unsqueeze(0).to(device)

        # ── Stage 2: 重排序 ──
        scores = model(text_t, user_t, banned_t, cand_a_t, cand_t_t)
        scores = scores.cpu().numpy()[0]

        # 按分数降序排列
        ranked_indices = np.argsort(scores)[::-1]
        rank = int(np.where(ranked_indices == topk_ids.index(true_track))[0][0]) + 1

        total_mrr += 1.0 / rank
        if rank <= 10:
            total_ndcg10 += 1.0 / np.log2(rank + 1)
        if rank == 1:
            hit_at_1 += 1
        if rank <= 10:
            hit_at_10 += 1
        count += 1

    if count == 0:
        return {}

    return {
        'count': count,
        'MRR': total_mrr / count,
        'NDCG@10': total_ndcg10 / count,
        'Hit@1': hit_at_1 / count,
        'Hit@10': hit_at_10 / count,
    }


def main():
    print("=" * 60)
    print("两阶段推荐系统评估 Pipeline")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n==== 加载测试数据 ====")
    _, test_df = load_preprocessed_data()
    test_pos = test_df[test_df['label'] == 1]
    # 评估时限制样本数（可通过配置调整）
    print(f"评估样本: {len(test_pos)} (全量 test 正例)")

    # ── 加载特征 ──
    print("\n==== 加载特征 ====")
    track_vectors = load_track_vectors()
    thought_map, thought_ids, thought_matrix = load_thought_index()
    user_emb_dict = load_user_embeddings()
    thought_map_id = {tid: i for i, tid in enumerate(thought_ids)}

    # ── Stage 1 Only 评估（全量 43K 候选）──
    print("\n==== Stage 1 Only (全量 Thought 检索) ====")
    stage1_evaluate(thought_ids, thought_matrix, thought_map_id, test_pos, desc="Stage1-Test")

    # ── 加载 Stage 2 模型 ──
    print("\n==== 加载 Stage 2 模型 ====")
    model = CrossAttentionReranker(
        d_model=STAGE2_D_MODEL,
        num_heads=STAGE2_NUM_HEADS,
        num_layers=STAGE2_NUM_LAYERS,
        dropout=STAGE2_DROPOUT,
        mlp_hidden=STAGE2_MLP_HIDDEN,
    ).to(DEVICE)

    model_path = 'models/stage2_model_best.pth'
    if not os.path.exists(model_path):
        model_path = 'models/stage2_model_final.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"加载模型: {model_path}")
    else:
        print("[WARNING] 未找到训练好的 Stage 2 模型，使用随机权重评估")
        print("请先运行: python train_stage2.py")

    # ── 两阶段评估 ──
    print(f"\n==== 两阶段评估 (Stage1 Top-{STAGE2_K_RETRIEVAL} + Stage2) ====")
    results = evaluate_two_stage(
        model, test_pos, track_vectors, thought_map,
        thought_ids, thought_matrix, user_emb_dict,
        K=STAGE2_K_RETRIEVAL, device=DEVICE, desc="Two-Stage"
    )

    # ── 结果汇总 ──
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
