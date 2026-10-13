"""
数据预处理：基于 GPA 金标，train/test 分离，逐层上下文 + 因果掩码。

预处理策略：
  - 每个 session 遍历全部 music 轮次，生成逐层累积样本
  - 保留所有有 GPA 标注的轮次：
      label=1: MOVES_TOWARD_GOAL       → 正例，MSE 训练目标
      label=0: DOES_NOT_MOVE_TOWARD    → 负例，排除（留作对比学习）
  - 首轮无评估 (None) → 跳过
  - history_tracks 仅含当前目标之前的曲目（因果掩码）
  - train: 15,199 sessions → ~106K 样本（正负各半）
  - test:   1,000 sessions →   ~7K 样本

用法：
  from data.dataset import load_preprocessed_data
  train_df, test_df = load_preprocessed_data()
  train_pos = train_df[train_df['label'] == 1]  # 仅正例用于 MSE
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from utils.text_utils import get_text_embedding
from config.default_config import AUDIO_DIM, USER_EMB_DIM


def _extract_gpa_samples(sessions_df, desc=""):
    """
    遍历 session，对每个 music 轮次生成逐层累积样本。
    保留所有有 GPA 标注的轮次：
      label=1: MOVES_TOWARD_GOAL     → 正例，MSE 目标
      label=0: DOES_NOT_MOVE_TOWARD  → 负例，排除（或用于对比学习）
    首轮无评估 (None) → 跳过。
    """
    rows = []

    for _, row in sessions_df.iterrows():
        session_id = row.get('session_id')
        user_id = row.get('user_id')
        turns = row.get('conversations', [])

        gpa_map = {}
        gpa_list = row.get('goal_progress_assessments')
        if gpa_list is not None and len(gpa_list) > 0:
            for a in gpa_list:
                gpa_map[a['turn_number']] = a['goal_progress_assessment']

        context_text = ""
        history_tracks = []

        for turn in turns:
            role = turn.get('role', '')
            content = turn.get('content', '')

            if role == 'user':
                context_text += f" User: {content}"
            elif role == 'music':
                if not (content and isinstance(content, str) and len(content) == 36):
                    continue
                tn = turn.get('turn_number')
                gpa = gpa_map.get(tn)
                if gpa is None:
                    history_tracks.append(content)
                    continue  # 首轮无评估

                label = 1 if gpa == 'MOVES_TOWARD_GOAL' else 0
                rows.append({
                    'session_id': session_id,
                    'user_id': user_id,
                    'context_text': context_text.strip(),
                    'target_track_id': content,
                    'history_tracks': list(history_tracks),
                    'label': label,
                })
                history_tracks.append(content)

            elif role == 'assistant':
                context_text += f" System: {content}"

    df = pd.DataFrame(rows)
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"[{desc}] {len(df)} 样本 (正例: {pos}, 负例: {neg})")
    return df


def load_preprocessed_data():
    """加载并预处理 train + test 数据。"""
    print("==== 加载原始数据集 ====")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    train_raw = ds['train'].to_pandas()
    test_raw = ds['test'].to_pandas()
    print(f"train sessions: {len(train_raw)}, test sessions: {len(test_raw)}")

    print("\n==== 预处理 train ====")
    train_df = _extract_gpa_samples(train_raw, desc="train")

    print("\n==== 预处理 test ====")
    test_df = _extract_gpa_samples(test_raw, desc="test")

    return train_df, test_df


class MultiModalRecDataset(Dataset):
    """多模态推荐数据集：文本 + 历史曲目 + 用户嵌入 → 目标音频嵌入。"""

    def __init__(self, df, track_vectors, user_emb_dict=None,
                 user_emb_dim=USER_EMB_DIM, standard_dim=AUDIO_DIM):
        self.samples = []

        print(f"构建数据集，候选样本: {len(df)}")
        for _, row in df.iterrows():
            target_track = row['target_track_id']
            if target_track not in track_vectors:
                continue
            target_vec = np.asarray(track_vectors[target_track], dtype=np.float32)
            if target_vec.shape[0] != standard_dim:
                continue

            user_id = row.get('user_id')
            history_tracks = row.get('history_tracks', []) or []

            # 历史曲目 → banned_center（均值池化）
            banned_vecs = []
            for tid in history_tracks:
                if tid in track_vectors and track_vectors[tid] is not None:
                    v = np.asarray(track_vectors[tid], dtype=np.float32)
                    if v.shape[0] == standard_dim:
                        banned_vecs.append(v)
            banned_center = (np.mean(np.stack(banned_vecs), axis=0)
                             if banned_vecs else np.zeros(standard_dim, dtype=np.float32))

            # 用户嵌入
            user_emb = np.zeros(user_emb_dim, dtype=np.float32)
            if user_emb_dict is not None and user_id in user_emb_dict:
                emb = np.asarray(user_emb_dict[user_id], dtype=np.float32)
                if emb.shape[0] == user_emb_dim:
                    user_emb = emb

            self.samples.append({
                'text': row['context_text'],
                'banned_center': banned_center.astype(np.float32),
                'target_vec': target_vec,
                'user_emb': user_emb,
            })

        print(f"[OK] 数据集构建完成: {len(self.samples)} 样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'text_emb': torch.tensor(get_text_embedding(s['text']), dtype=torch.float32),
            'banned_center': torch.tensor(s['banned_center'], dtype=torch.float32),
            'user_emb': torch.tensor(s['user_emb'], dtype=torch.float32),
            'target_vec': torch.tensor(s['target_vec'], dtype=torch.float32),
        }
