"""
Stage 1: Thought-Driven Retrieval（纯语义匹配，无需训练）

逻辑: 用户对话 → MiniLM → 与全量 thought 向量做余弦匹配 → 排序 → MRR
优势: 完全绕过"歌曲只出现一次"问题，借 MiniLM 预训练语义直接检索
"""
import os
import numpy as np
import torch
from tqdm import tqdm

from config.default_config import PROXY_ENV, DEVICE, TEXT_DIM
from data.dataset import load_preprocessed_data
from data.loader import load_track_vectors
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


def load_thought_index():
    """加载预计算的 track → thought_avg 嵌入。"""
    ids = np.load('data/track_thought_ids.npy', allow_pickle=True)
    emb = np.load('data/track_thought_emb.npy')
    mapping = dict(zip(ids, emb))
    print(f"Thought 索引: {len(mapping)} 曲目, 维度 {emb.shape[1]}")
    return mapping, ids, emb


@torch.no_grad()
def evaluate(thought_ids, thought_matrix, thought_map_id, eval_df, desc="Eval"):
    """全量 thought 检索 MRR。"""
    thought_t = torch.tensor(thought_matrix, dtype=torch.float32).to(DEVICE)

    total_mrr = 0.0
    total_ndcg10 = 0.0
    hit_at_1 = 0
    hit_at_10 = 0
    count = 0
    missing = 0

    for idx in tqdm(range(len(eval_df)), desc=desc):
        sample = eval_df.iloc[idx]
        true_track = sample['target_track_id']
        if true_track not in thought_map_id:
            missing += 1
            continue

        query = get_text_embedding(sample['context_text'])
        query_t = torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 余弦相似度 → 全量排序
        sims = (torch.nn.functional.normalize(query_t, dim=1)
                @ torch.nn.functional.normalize(thought_t, dim=1).T).cpu().numpy()[0]
        rank = int(np.argsort(sims)[::-1].tolist().index(thought_map_id[true_track])) + 1
        total_mrr += 1.0 / rank
        if rank <= 10:
            total_ndcg10 += 1.0 / np.log2(rank + 1)
        if rank == 1:
            hit_at_1 += 1
        if rank <= 10:
            hit_at_10 += 1
        count += 1

    if count == 0:
        return None
    print(f"[{desc}] 全量 {len(thought_ids)} 候选 | {count} 条 (缺thought: {missing})")
    print(f"  MRR: {total_mrr/count:.4f} | NDCG@10: {total_ndcg10/count:.4f} | "
          f"Hit@1: {hit_at_1/count:.4f} | Hit@10: {hit_at_10/count:.4f}")


@torch.no_grad()
def retrieve_topk(context_text, thought_ids, thought_matrix,
                  K=200, device=DEVICE):
    """
    Stage 1: 为单个 context 检索 Top-K 候选曲目。

    Args:
        context_text:    str — 对话上下文
        thought_ids:     np.array — (N,) track IDs
        thought_matrix:  np.array — (N, 384) thought embeddings
        K:               int — 返回的候选数
        device:          torch.device

    Returns:
        topk_ids:    list[str] — Top-K track IDs（按相似度降序）
        topk_scores: list[float] — 对应余弦相似度
    """
    query = get_text_embedding(context_text)
    query_t = torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(device)
    thought_t = torch.tensor(thought_matrix, dtype=torch.float32).to(device)

    sims = (torch.nn.functional.normalize(query_t, dim=1)
            @ torch.nn.functional.normalize(thought_t, dim=1).T).cpu().numpy()[0]
    topk_indices = np.argsort(sims)[::-1][:K]
    topk_ids = [str(thought_ids[i]) for i in topk_indices]
    topk_scores = [float(sims[i]) for i in topk_indices]
    return topk_ids, topk_scores


def main():
    print("==== 加载数据 ====")
    train_df, test_df = load_preprocessed_data()
    train_pos = train_df[train_df['label'] == 1].copy()
    test_pos = test_df[test_df['label'] == 1].copy()

    print("\n==== 加载 Thought 索引 ====")
    thought_map, thought_ids, thought_matrix = load_thought_index()
    thought_map_id = {tid: i for i, tid in enumerate(thought_ids)}

    print("\n==== Thought 检索评估 ====")
    evaluate(thought_ids, thought_matrix, thought_map_id, train_pos, desc="Train")
    evaluate(thought_ids, thought_matrix, thought_map_id, test_pos, desc="Test")


if __name__ == '__main__':
    main()
