"""
共享数据加载工具：加载 HuggingFace 数据集、构建候选池、计算排名指标。
"""
import numpy as np
from datasets import load_dataset
from config.default_config import AUDIO_DIM, USER_EMB_DIM, CANDIDATE_POOL_SIZE


def load_track_vectors():
    """加载音乐音频嵌入向量 (LAION-CLAP, 512d)。"""
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    df = ds['default'].to_pandas() if 'default' in ds else next(iter(ds.values())).to_pandas()
    col = 'audio-laion_clap' if 'audio-laion_clap' in df.columns else df.columns[-1]
    track_vectors = dict(zip(df['track_id'], df[col]))
    print(f"成功加载 {len(track_vectors)} 个音乐向量。")
    return track_vectors


def load_track_metadata():
    """加载音乐元数据（曲目→艺术家映射）。"""
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    df = ds['default'].to_pandas() if 'default' in ds else next(iter(ds.values())).to_pandas()
    return df


def load_user_embeddings():
    """加载用户协同过滤嵌入向量 (CF-BPR, 128d)，合并 train + test_warm + test_cold。"""
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_emb_dict = {}
    for split in ds.keys():
        df = ds[split].to_pandas()
        for _, row in df.iterrows():
            uid = row['user_id']
            emb = np.asarray(row['cf-bpr'], dtype=np.float32)
            if emb.shape[0] == USER_EMB_DIM:
                user_emb_dict[uid] = emb
    print(f"成功加载 {len(user_emb_dict)} 条用户嵌入向量（{len(ds.keys())} splits）。")
    return user_emb_dict


def build_track_to_artist(track_meta_df):
    """构建 track_id → artist 的安全映射。"""
    artist_col = 'artist_id' if 'artist_id' in track_meta_df.columns else 'artist'
    mapping = {}
    for _, row in track_meta_df.iterrows():
        tid = row['track_id']
        val = row[artist_col]
        if isinstance(val, (list, np.ndarray)):
            val_str = str(val[0]) if len(val) > 0 else 'unknown_artist'
        elif isinstance(val, dict):
            val_str = str(next(iter(val.values())))
        else:
            val_str = str(val)
        mapping[tid] = val_str.strip()
    print(f"成功构建 {len(mapping)} 条歌曲→歌手索引。")
    return mapping


def build_candidate_pool(track_vectors_dict, pool_size=CANDIDATE_POOL_SIZE):
    """从音乐向量字典中构建固定大小的候选池。"""
    ids, vectors = [], []
    for tid, vec in track_vectors_dict.items():
        if vec is not None and len(np.asarray(vec)) == AUDIO_DIM:
            ids.append(tid)
            vectors.append(np.asarray(vec, dtype=np.float32))
            if len(ids) >= pool_size:
                break
    matrix = np.stack(vectors) if vectors else np.zeros((0, AUDIO_DIM), dtype=np.float32)
    return ids, matrix


def compute_ranking_metrics(rank):
    """根据排名计算 MRR 和 NDCG。"""
    if rank <= 0:
        return 0.0, 0.0
    mrr = 1.0 / rank
    ndcg = 1.0 / np.log2(rank + 1)
    return mrr, ndcg
