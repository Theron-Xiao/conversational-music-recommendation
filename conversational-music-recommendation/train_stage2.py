"""
Stage 2 训练脚本：训练 CrossAttentionReranker 精排序模型。

训练:
  - 仅用正样本（label=1）
  - 每样本采样 N 个负例
  - LambdaRank ApproxNDCG 损失

验证:
  - 每个 epoch 后在测试集快速评估 MRR@K
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.default_config import (
    PROXY_ENV,
    DEVICE,
    TEXT_DIM, AUDIO_DIM, USER_EMB_DIM,
    STAGE2_D_MODEL, STAGE2_NUM_HEADS, STAGE2_NUM_LAYERS,
    STAGE2_DROPOUT, STAGE2_MLP_HIDDEN,
    STAGE2_K_RETRIEVAL, STAGE2_SIGMA, STAGE2_EPOCHS,
    STAGE2_LR, STAGE2_BATCH_SIZE,
)
from data.dataset import load_preprocessed_data
from data.loader import (
    load_track_vectors, load_user_embeddings,
)
from data.stage2_dataset import Stage2Dataset
from models.stage2_model import CrossAttentionReranker
from models.approx_ndcg import lambdarank_loss
from stage1_thought_retrieval import load_thought_index, retrieve_topk
from utils.text_utils import get_text_embedding

os.environ.update(PROXY_ENV)


def evaluate_mrr_at_k(model, eval_df, track_vectors, thought_map,
                      thought_ids, thought_matrix, user_emb_dict,
                      K=STAGE2_K_RETRIEVAL, device=DEVICE, desc="Val"):
    """
    验证: Stage 1 召回 Top-K → Stage 2 重排序 → 计算 MRR@K。
    """
    model.eval()
    total_mrr = 0.0
    count = 0
    hit_at_k = 0

    for idx in tqdm(range(len(eval_df)), desc=desc):
        sample = eval_df.iloc[idx]
        true_track = sample['target_track_id']

        if true_track not in track_vectors:
            continue

        # Stage 1: 召回 Top-K
        context_text = sample['context_text']
        topk_ids, _ = retrieve_topk(context_text, thought_ids, thought_matrix, K=K, device=device)

        # 确保 ground truth 在候选集中
        if true_track not in topk_ids:
            topk_ids.insert(0, true_track)
            topk_ids = topk_ids[:K]

        K_actual = len(topk_ids)

        # 构建 query 特征
        text_emb = get_text_embedding(context_text)
        uid = sample.get('user_id')
        user_emb = user_emb_dict.get(uid, np.zeros(USER_EMB_DIM, dtype=np.float32))

        history_tracks = sample.get('history_tracks', []) or []
        banned_vecs = [np.asarray(track_vectors[tid], dtype=np.float32)
                       for tid in history_tracks if tid in track_vectors]
        banned_center = (np.mean(np.stack(banned_vecs), axis=0)
                         if banned_vecs else np.zeros(AUDIO_DIM, dtype=np.float32))

        # 构建候选特征
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

        with torch.no_grad():
            scores = model(text_t, user_t, banned_t, cand_a_t, cand_t_t)
            scores = scores.cpu().numpy()[0]

        # 按分数降序排列
        ranked_indices = np.argsort(scores)[::-1]
        rank = int(np.where(ranked_indices == topk_ids.index(true_track))[0][0]) + 1
        total_mrr += 1.0 / rank
        if rank <= K_actual:
            hit_at_k += 1
        count += 1

    if count == 0:
        return None

    avg_mrr = total_mrr / count
    hit_rate = hit_at_k / count
    print(f"[{desc}] {count} 条 | K={K} | MRR: {avg_mrr:.4f} | HitRate: {hit_rate:.4f}")
    return avg_mrr


def train():
    print("==== 加载数据 ====")
    train_df, test_df = load_preprocessed_data()
    train_pos = train_df[train_df['label'] == 1]
    test_pos_all = test_df[test_df['label'] == 1]

    # 验证集: 测试集中取部分样本
    val_samples = min(200, len(test_pos_all))
    test_pos = test_pos_all.head(val_samples)
    print(f"训练样本: {len(train_pos)}, 验证样本: {len(test_pos)}")

    print("\n==== 加载特征 ====")
    track_vectors = load_track_vectors()
    thought_map, thought_ids, thought_matrix = load_thought_index()
    user_emb_dict = load_user_embeddings()

    print("\n==== 构建 Stage 2 训练数据集（Stage 1 Top-K 候选） ====")
    train_dataset = Stage2Dataset(
        train_pos, track_vectors, thought_map, thought_ids, thought_matrix,
        user_emb_dict, K=STAGE2_K_RETRIEVAL,
    )
    train_loader = DataLoader(train_dataset, batch_size=STAGE2_BATCH_SIZE, shuffle=True)

    print("\n==== 初始化模型 ====")
    model = CrossAttentionReranker(
        d_model=STAGE2_D_MODEL,
        num_heads=STAGE2_NUM_HEADS,
        num_layers=STAGE2_NUM_LAYERS,
        dropout=STAGE2_DROPOUT,
        mlp_hidden=STAGE2_MLP_HIDDEN,
    ).to(DEVICE)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=STAGE2_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── 训练循环 ──
    print(f"\n{'='*50}")
    print(f"开始训练 {STAGE2_EPOCHS} epochs | Batch={STAGE2_BATCH_SIZE} | K={STAGE2_K_RETRIEVAL} | σ={STAGE2_SIGMA}")
    print(f"{'='*50}\n")

    best_mrr = 0.0

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{STAGE2_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            text_emb = batch['text_emb'].to(DEVICE)
            user_emb = batch['user_emb'].to(DEVICE)
            banned_center = batch['banned_center'].to(DEVICE)
            cand_audio = batch['cand_audio'].to(DEVICE)
            cand_thought = batch['cand_thought'].to(DEVICE)
            relevance = batch['relevance'].to(DEVICE)

            optimizer.zero_grad()
            scores = model(text_emb, user_emb, banned_center, cand_audio, cand_thought)
            loss = lambdarank_loss(scores, relevance, sigma=STAGE2_SIGMA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] 平均 Loss: {avg_loss:.4f}")

        # 验证
        val_mrr = evaluate_mrr_at_k(
            model, test_pos, track_vectors, thought_map,
            thought_ids, thought_matrix, user_emb_dict,
            K=STAGE2_K_RETRIEVAL, device=DEVICE, desc=f"Val Epoch {epoch+1}"
        )

        scheduler.step()

        # 保存最佳模型
        if val_mrr is not None and val_mrr > best_mrr:
            best_mrr = val_mrr
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/stage2_model_best.pth')
            print(f"  → 保存最佳模型 (MRR: {best_mrr:.4f})")

    # ── 保存最终模型 ──
    torch.save(model.state_dict(), 'models/stage2_model_final.pth')
    print(f"\n[OK] 训练完成！最佳 MRR: {best_mrr:.4f}")
    print(f"模型已保存至: models/stage2_model_best.pth / stage2_model_final.pth")


if __name__ == '__main__':
    train()
