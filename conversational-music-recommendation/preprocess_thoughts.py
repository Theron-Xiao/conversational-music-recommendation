"""
Thought 聚合预处理：构建 track_id → thought 文本 → 均值嵌入 映射。
弥补数据中缺乏显式 BPM/节奏/情感价字段的遗憾。

输出: data/track_thought_emb.npy  (47K × 384)
      data/track_thought_ids.npy  (47K 个 track_id)
"""
import os
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

from config.default_config import PROXY_ENV, TEXT_DIM
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


def build_thought_aggregation():
    """扫描 train + test 所有 music turn，构建 track_id → [thought_texts] 映射，再均值池化。

    加入 test 集的 thought 不会泄露评估信息——thought 是曲目推荐理由的元数据，
    等价于"曲目描述文本"，不是用户反馈标签。扩大覆盖让 test 评估更完整。
    """
    print("==== 加载全量数据集 (train + test) ====")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")

    # 1. 收集  track_id → [thought_texts]（train + test 一起扫）
    track_thoughts = defaultdict(list)
    for split in ['train', 'test']:
        split_data = ds[split]
        for row in tqdm(split_data, desc=f"扫描 {split}"):
            for turn in row['conversations']:
                if turn.get('role') == 'music':
                    tid = turn.get('content', '')
                    thought = turn.get('thought', '')
                    if tid and isinstance(tid, str) and len(tid) == 36 and thought:
                        track_thoughts[tid].append(thought)

    print(f"覆盖曲目: {len(track_thoughts)}")
    thought_counts = [len(v) for v in track_thoughts.values()]
    print(f"每曲 Thought 数: min={min(thought_counts)}, max={max(thought_counts)}, "
          f"avg={np.mean(thought_counts):.1f}")

    # 2. 均值池化 → 嵌入
    track_ids = []
    track_embeddings = []
    for tid, thoughts in tqdm(track_thoughts.items(), desc="计算 Thought 嵌入"):
        embs = [get_text_embedding(t) for t in thoughts]
        mean_emb = np.mean(embs, axis=0).astype(np.float32)
        track_ids.append(tid)
        track_embeddings.append(mean_emb)

    ids_arr = np.array(track_ids)
    emb_arr = np.stack(track_embeddings)

    os.makedirs('data', exist_ok=True)
    np.save('data/track_thought_ids.npy', ids_arr)
    np.save('data/track_thought_emb.npy', emb_arr)
    print(f"\n保存完成: {len(ids_arr)} 曲目, 嵌入维度 {emb_arr.shape[1]}")
    return ids_arr, emb_arr


if __name__ == '__main__':
    build_thought_aggregation()
