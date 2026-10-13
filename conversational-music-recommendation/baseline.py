"""
LR 基线：Text(384) + User(128) → Linear → Audio(512)
多模态回归，无掩码。训练目标 MSE，评估 MRR。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from config.default_config import (
    PROXY_ENV, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    DEVICE, TEXT_DIM, AUDIO_DIM, USER_EMB_DIM,
)
from data.dataset import load_preprocessed_data
from data.loader import load_track_vectors, load_user_embeddings, build_candidate_pool
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


class LRModel(nn.Module):
    """Text(384) + User(128) → Linear → Audio(512)"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(TEXT_DIM + USER_EMB_DIM, AUDIO_DIM)

    def forward(self, text_emb, user_emb):
        return self.linear(torch.cat([text_emb, user_emb], dim=1))


class LRDataset(Dataset):
    """LR 专用数据集：文本 + 用户 → 目标音频，无历史曲目。"""
    def __init__(self, df, track_vectors, user_emb_dict):
        self.samples = []
        for _, row in df.iterrows():
            tid = row['target_track_id']
            if tid not in track_vectors:
                continue
            uid = row.get('user_id')
            user_emb = user_emb_dict.get(uid, np.zeros(USER_EMB_DIM, dtype=np.float32))
            self.samples.append({
                'text': row['context_text'],
                'user_emb': np.asarray(user_emb, dtype=np.float32),
                'target_vec': np.asarray(track_vectors[tid], dtype=np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'text_emb': torch.tensor(get_text_embedding(s['text']), dtype=torch.float32),
            'user_emb': torch.tensor(s['user_emb'], dtype=torch.float32),
            'target_vec': torch.tensor(s['target_vec'], dtype=torch.float32),
        }


def evaluate_mrr(model, track_vectors_dict, eval_df, user_emb_dict,
                 device, desc="Eval"):
    """1000 候选池评估：MRR, NDCG@10, Hit@1, Hit@10。"""
    model.eval()
    candidate_ids, candidate_matrix = build_candidate_pool(track_vectors_dict)

    total_mrr = 0.0
    total_ndcg10 = 0.0
    hit_at_1 = 0
    hit_at_10 = 0
    count = 0

    for idx in range(len(eval_df)):
        sample = eval_df.iloc[idx]
        true_track = sample['target_track_id']
        if true_track not in track_vectors_dict:
            continue

        working_ids = list(candidate_ids)
        working_vectors = np.array(candidate_matrix, copy=True)
        if true_track not in working_ids:
            working_ids.append(true_track)
            working_vectors = np.vstack([
                working_vectors,
                np.asarray(track_vectors_dict[true_track], dtype=np.float32)
            ])

        text_emb = get_text_embedding(sample['context_text'])
        uid = sample.get('user_id')
        user_emb = user_emb_dict.get(uid, np.zeros(USER_EMB_DIM, dtype=np.float32))

        text_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)
        user_t = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output_vec = model(text_t, user_t).cpu().numpy()[0]

        sims = cosine_similarity(output_vec.reshape(1, -1), working_vectors)[0]
        rank_order = np.argsort(sims)[::-1]
        rank = [working_ids[i] for i in rank_order].index(true_track) + 1

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

    print(f"[{desc}] {count} 条 | MRR: {total_mrr/count:.4f} | "
          f"NDCG@10: {total_ndcg10/count:.4f} | "
          f"Hit@1: {hit_at_1/count:.4f} | Hit@10: {hit_at_10/count:.4f}")
    return total_mrr / count


def train():
    print("==== 预处理数据 ====")
    train_df, test_df = load_preprocessed_data()
    train_pos = train_df[train_df['label'] == 1].copy()   # 仅正例做 MSE
    test_pos = test_df[test_df['label'] == 1].copy()       # test set 正例

    track_vectors_dict = load_track_vectors()
    user_emb_dict = load_user_embeddings()

    train_dataset = LRDataset(train_pos, track_vectors_dict, user_emb_dict)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LRModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    print(f"[Start] 训练 {EPOCHS} epoch...\n")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            text_emb = batch['text_emb'].to(DEVICE)
            user_emb = batch['user_emb'].to(DEVICE)
            target_vec = batch['target_vec'].to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(text_emb, user_emb), target_vec)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Stats] Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

    # ── 保存模型 ──
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/baseline_model.pth')

    # ── 评估：Train + Test ──
    evaluate_mrr(model, track_vectors_dict, train_pos, user_emb_dict, DEVICE, desc="Train")
    evaluate_mrr(model, track_vectors_dict, test_pos, user_emb_dict, DEVICE, desc="Test")


if __name__ == '__main__':
    train()
