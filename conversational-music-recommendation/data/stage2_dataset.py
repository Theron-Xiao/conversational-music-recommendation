"""
Stage 2 训练数据集 v2：候选集 = Stage 1 的 Top-K 检索结果。

关键改动（修复 train/eval 分布不匹配）：
  - 训练时不再随机采样负例
  - 而是对每个样本跑 retrieve_topk，取 Stage 1 的实际 Top-200 作为候选集
  - 保证训练和评估时 Stage 2 看到的候选分布完全一致
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config.default_config import (
    PROXY_ENV, DEVICE,
    TEXT_DIM, AUDIO_DIM, USER_EMB_DIM,
    STAGE2_K_RETRIEVAL,
)
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


class Stage2Dataset(Dataset):
    """
    Stage 2 训练数据集 — 候选来自 Stage 1 的真实 Top-K。

    预计算流程:
      1. 对所有样本的 context_text 编码 (MiniLM, 384d)
      2. 与全量 thought 矩阵做余弦相似度
      3. 取 Top-K (默认 200) + 确保 ground truth 在候选集中
      4. 缓存 K 个候选的 ID 列表
    """

    def __init__(
        self,
        df,
        track_vectors,
        thought_map,
        thought_ids,
        thought_matrix,
        user_emb_dict,
        K=STAGE2_K_RETRIEVAL,
        standard_audio_dim=AUDIO_DIM,
        user_emb_dim=USER_EMB_DIM,
        thought_dim=TEXT_DIM,
        cache_path='data/stage2_candidates.npz',
    ):
        self.K = K
        self.standard_audio_dim = standard_audio_dim
        self.user_emb_dim = user_emb_dim
        self.thought_dim = thought_dim

        # 加载或预计算候选列表
        if os.path.exists(cache_path):
            print(f"从缓存加载候选: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.cached_text_embs = [data['text_embs'][i] for i in range(len(data['text_embs']))]
            self.cached_banned_centers = [data['banned_centers'][i] for i in range(len(data['banned_centers']))]
            self.cached_user_embs = [data['user_embs'][i] for i in range(len(data['user_embs']))]
            self.cached_cand_ids = [list(data['cand_ids'][i]) for i in range(len(data['cand_ids']))]
            self.cached_target_track = [str(data['target_tracks'][i]) for i in range(len(data['target_tracks']))]
        else:
            print("预计算 Stage 1 Top-K 候选（一次性，后续从缓存加载）...")
            result = self._precompute_candidates(
                df, track_vectors, thought_map, thought_ids, thought_matrix,
                user_emb_dict, standard_audio_dim, user_emb_dim, thought_dim, cache_path
            )
            (self.cached_text_embs, self.cached_banned_centers,
             self.cached_user_embs, self.cached_cand_ids,
             self.cached_target_track) = result

        # 预加载 track vectors → numpy，方便 __getitem__ 查找
        self._track_audio = track_vectors
        self._thought_map = thought_map
        print(f"数据集: {len(self)} 样本, K={self.K}")

    def _precompute_candidates(self, df, track_vectors, thought_map,
                                thought_ids, thought_matrix, user_emb_dict,
                                audio_dim, user_dim, thought_dim, cache_path):
        """一次性预计算所有样本的 Stage 1 Top-K 候选。"""

        # 构建 thought 索引：track_id → index
        tid_to_idx = {str(tid): i for i, tid in enumerate(thought_ids)}
        thought_t = torch.tensor(thought_matrix, dtype=torch.float32).to(DEVICE)
        thought_t_norm = torch.nn.functional.normalize(thought_t, dim=1)

        text_embs_list = []
        banned_list = []
        user_list = []
        cand_ids_list = []
        target_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage2 预计算"):
            target_track = row['target_track_id']
            if target_track not in track_vectors:
                continue

            context_text = row['context_text']

            # Query 特征
            text_emb = get_text_embedding(context_text)

            history_tracks = row.get('history_tracks', []) or []
            banned_vecs = [np.asarray(track_vectors[tid], dtype=np.float32)
                           for tid in history_tracks if tid in track_vectors]
            banned_center = (np.mean(np.stack(banned_vecs), axis=0)
                             if banned_vecs else np.zeros(audio_dim, dtype=np.float32))

            uid = row.get('user_id')
            user_emb = np.asarray(user_emb_dict.get(uid, np.zeros(user_dim, dtype=np.float32)),
                                  dtype=np.float32)

            # Stage 1 检索
            query_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            query_t = torch.nn.functional.normalize(query_t, dim=1)
            sims = (query_t @ thought_t_norm.T).cpu().numpy()[0]
            topk_idx = np.argsort(sims)[::-1][:self.K]
            topk_ids = [str(thought_ids[i]) for i in topk_idx]

            # 确保 ground truth 在候选集中
            if target_track not in topk_ids:
                topk_ids.insert(0, target_track)
                topk_ids = topk_ids[:self.K]

            text_embs_list.append(text_emb)
            banned_list.append(banned_center)
            user_list.append(user_emb)
            cand_ids_list.append(topk_ids)
            target_list.append(target_track)

        # 缓存
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path,
                 text_embs=np.array(text_embs_list, dtype=np.float32),
                 banned_centers=np.array(banned_list, dtype=np.float32),
                 user_embs=np.array(user_list, dtype=np.float32),
                 cand_ids=np.array(cand_ids_list, dtype=object),
                 target_tracks=np.array(target_list, dtype=object))
        print(f"候选缓存已保存: {cache_path}")

        return text_embs_list, banned_list, user_list, cand_ids_list, target_list

    def __len__(self):
        return len(self.cached_text_embs)

    def __getitem__(self, idx):
        text_emb = self.cached_text_embs[idx]
        banned_center = self.cached_banned_centers[idx]
        user_emb = self.cached_user_embs[idx]
        topk_ids = self.cached_cand_ids[idx]
        target_track = self.cached_target_track[idx]

        K_actual = len(topk_ids)

        # 构建候选特征矩阵
        cand_audio = np.zeros((K_actual, self.standard_audio_dim), dtype=np.float32)
        cand_thought = np.zeros((K_actual, self.thought_dim), dtype=np.float32)
        relevance = np.zeros(K_actual, dtype=np.float32)

        for i, tid in enumerate(topk_ids):
            cand_audio[i] = np.asarray(
                self._track_audio.get(tid, np.zeros(self.standard_audio_dim)),
                dtype=np.float32
            )
            cand_thought[i] = np.asarray(
                self._thought_map.get(tid, np.zeros(self.thought_dim)),
                dtype=np.float32
            )
            if tid == target_track:
                relevance[i] = 1.0

        # 随机打乱
        perm = np.random.permutation(K_actual)
        cand_audio = cand_audio[perm]
        cand_thought = cand_thought[perm]
        relevance = relevance[perm]

        return {
            'text_emb': torch.tensor(text_emb, dtype=torch.float32),
            'user_emb': torch.tensor(user_emb, dtype=torch.float32),
            'banned_center': torch.tensor(banned_center, dtype=torch.float32),
            'cand_audio': torch.tensor(cand_audio, dtype=torch.float32),
            'cand_thought': torch.tensor(cand_thought, dtype=torch.float32),
            'relevance': torch.tensor(relevance, dtype=torch.float32),
        }
