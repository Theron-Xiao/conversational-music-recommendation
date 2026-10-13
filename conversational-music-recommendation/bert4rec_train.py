import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from config.default_config import PROXY_ENV

os.environ.update(PROXY_ENV)

from collections import defaultdict
import time

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
MAXLEN = 10
D_MODEL = 32
MAX_TRAIN_SAMPLES = 40000
REG = 1e-5
MASK_PROB = 0.2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def extract_sequences(ds_split):
    user_seqs = defaultdict(list)
    for sample in ds_split:
        user_id = sample['user_id']
        conversations = sample['conversations']
        track_ids = []
        for msg in conversations:
            if msg['role'] == 'music':
                track_ids.append(msg['content'])
        if len(track_ids) >= 2:
            user_seqs[user_id].append(track_ids)
    return user_seqs

print("\n" + "="*70)
print("步骤 1: 加载数据并提取交互序列")
print("="*70)

ds_challenge = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
train_seqs = extract_sequences(ds_challenge['train'])
test_seqs = extract_sequences(ds_challenge['test'])

all_user_seqs = defaultdict(list)
for uid, seqs in train_seqs.items():
    all_user_seqs[uid].extend(seqs)
for uid, seqs in test_seqs.items():
    all_user_seqs[uid].extend(seqs)

ds_user_emb = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
all_user_splits = {}
for split_name in ds_user_emb.keys():
    for row in ds_user_emb[split_name]:
        uid = row['user_id']
        all_user_splits[uid] = "test_warm" if split_name == "train" else split_name

all_track_ids = set()
for uid, seqs in all_user_seqs.items():
    for seq in seqs:
        all_track_ids.update(seq)

track_list = sorted(list(all_track_ids))
track2id = {tid: i+1 for i, tid in enumerate(track_list)}
id2track = {i+1: tid for i, tid in enumerate(track_list)}
NUM_ITEMS = len(track_list)

user_list = sorted(list(all_user_seqs.keys()))
user2id = {uid: i for i, uid in enumerate(user_list)}

print(f"用户数: {len(user_list)}")
print(f"曲目数: {NUM_ITEMS}")
print(f"train_warm 用户数: {sum(1 for v in all_user_splits.values() if v == 'train_warm')}")
print(f"test_warm 用户数: {sum(1 for v in all_user_splits.values() if v == 'test_warm')}")
print(f"test_cold 用户数: {sum(1 for v in all_user_splits.values() if v == 'test_cold')}")

total_seqs = sum(len(seqs) for seqs in all_user_seqs.values())
print(f"总会话序列数: {total_seqs}")

def build_training_data(user_seqs, user_splits, track2id, maxlen, max_samples=None):
    inputs = []
    targets = []
    all_users = sorted(user_seqs.keys())
    for uid in all_users:
        seqs = user_seqs[uid]
        for seq in seqs:
            seq_ids = [track2id[t] for t in seq if t in track2id]
            if len(seq_ids) < 2:
                continue
            for i in range(1, len(seq_ids)):
                target = seq_ids[i]
                input_seq = seq_ids[:i]
                if len(input_seq) > maxlen:
                    input_seq = input_seq[-maxlen:]
                inputs.append(input_seq)
                targets.append(target)
    if max_samples and len(inputs) > max_samples:
        indices = list(range(len(inputs)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        inputs = [inputs[i] for i in indices]
        targets = [targets[i] for i in indices]
    return inputs, targets

def build_test_data(user_seqs, user_splits, track2id, maxlen, split_filter=None):
    test_data = []
    for uid, seqs in user_seqs.items():
        if split_filter and user_splits.get(uid) != split_filter:
            continue
        for seq in seqs:
            seq_ids = [track2id[t] for t in seq if t in track2id]
            if len(seq_ids) < 2:
                continue
            target = seq_ids[-1]
            input_seq = seq_ids[:-1]
            if len(input_seq) > maxlen:
                input_seq = input_seq[-maxlen:]
            test_data.append((input_seq, target))
    return test_data

print("\n" + "="*70)
print("步骤 2: 构建训练和测试数据")
print("="*70)

train_inputs, train_targets = build_training_data(train_seqs, train_splits, track2id, MAXLEN, MAX_TRAIN_SAMPLES)
print(f"训练样本数: {len(train_inputs)}")

test_warm_data = build_test_data(test_seqs, test_splits, track2id, MAXLEN, 'test_warm')
test_cold_data = build_test_data(test_seqs, test_splits, track2id, MAXLEN, 'test_cold')
print(f"test_warm 样本数: {len(test_warm_data)}")
print(f"test_cold 样本数: {len(test_cold_data)}")

class BERTRecDataset(Dataset):
    def __init__(self, inputs, targets, num_items, maxlen, mask_prob=0.2):
        self.inputs = inputs
        self.targets = targets
        self.num_items = num_items
        self.maxlen = maxlen
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        seq = self.inputs[idx][:]
        target = self.targets[idx]
        masked_seq = []
        labels = []
        for item in seq:
            prob = random.random()
            if prob < self.mask_prob:
                masked_seq.append(self.mask_token)
                labels.append(item)
            else:
                masked_seq.append(item)
                labels.append(0)
        masked_seq.append(self.mask_token)
        labels.append(target)
        if len(masked_seq) > self.maxlen:
            masked_seq = masked_seq[-self.maxlen:]
            labels = labels[-self.maxlen:]
        padding_len = self.maxlen - len(masked_seq)
        masked_seq = [0] * padding_len + masked_seq
        labels = [0] * padding_len + labels
        return (
            torch.tensor(masked_seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )

class BERT4Rec(nn.Module):
    def __init__(self, num_items, d_model, maxlen, n_layers=2, n_heads=2, dropout=0.1):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.maxlen = maxlen
        vocab_size = num_items + 2
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        padding_mask = (input_ids == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        logits = self.out(x)
        return logits

def evaluate(model, test_data, num_items, maxlen, k_list=[1, 10], batch_size=128):
    model.eval()
    all_mrr = []
    all_ndcg = {k: [] for k in k_list}
    all_hit = {k: [] for k in k_list}
    mask_token = num_items + 1
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            seqs = []
            targets = []
            for seq, target in batch:
                seq_with_mask = seq + [mask_token]
                if len(seq_with_mask) > maxlen:
                    seq_with_mask = seq_with_mask[-maxlen:]
                padding_len = maxlen - len(seq_with_mask)
                seq_with_mask = [0] * padding_len + seq_with_mask
                seqs.append(seq_with_mask)
                targets.append(target)
            input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
            logits = model(input_ids)
            mask_positions = (input_ids == mask_token)
            for j in range(len(batch)):
                mask_idx = torch.where(mask_positions[j])[0]
                if len(mask_idx) == 0:
                    continue
                last_mask = mask_idx[-1]
                scores = logits[j, last_mask, 1:num_items+1]
                target = targets[j] - 1
                rank = (scores > scores[target]).sum().item() + 1
                all_mrr.append(1.0 / rank)
                for k in k_list:
                    hit = 1.0 if rank <= k else 0.0
                    all_hit[k].append(hit)
                    if rank <= k:
                        ndcg = 1.0 / np.log2(rank + 1)
                    else:
                        ndcg = 0.0
                    all_ndcg[k].append(ndcg)
    results = {
        'mrr': np.mean(all_mrr) if all_mrr else 0.0,
    }
    for k in k_list:
        results[f'hit@{k}'] = np.mean(all_hit[k]) if all_hit[k] else 0.0
        results[f'ndcg@{k}'] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
    return results

print("\n" + "="*70)
print("步骤 3: 初始化 BERT4Rec 模型")
print("="*70)

train_dataset = BERTRecDataset(train_inputs, train_targets, NUM_ITEMS, MAXLEN, MASK_PROB)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BERT4Rec(NUM_ITEMS, D_MODEL, MAXLEN, n_layers=2, n_heads=2, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=REG)
criterion = nn.CrossEntropyLoss(ignore_index=0)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params}")
print(f"d_model: {D_MODEL}, maxlen: {MAXLEN}, num_items: {NUM_ITEMS}")

print("\n" + "="*70)
print(f"步骤 4: 训练 {EPOCHS} 轮")
print("="*70)

start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | 用时: {elapsed:.1f}s")

print(f"\n训练总用时: {time.time() - start_time:.1f}s")

print("\n" + "="*70)
print("步骤 5: 评估模型")
print("="*70)

all_test_data = test_warm_data + test_cold_data
overall_results = evaluate(model, all_test_data, NUM_ITEMS, MAXLEN)
print("\n整体结果:")
print(f"  MRR     = {overall_results['mrr']:.6f}")
print(f"  Hit@1   = {overall_results['hit@1']:.6f}")
print(f"  Hit@10  = {overall_results['hit@10']:.6f}")
print(f"  NDCG@10 = {overall_results['ndcg@10']:.6f}")

if test_warm_data:
    warm_results = evaluate(model, test_warm_data, NUM_ITEMS, MAXLEN)
    print("\ntest_warm 分组结果:")
    print(f"  MRR     = {warm_results['mrr']:.6f}")
    print(f"  Hit@1   = {warm_results['hit@1']:.6f}")
    print(f"  Hit@10  = {warm_results['hit@10']:.6f}")
    print(f"  NDCG@10 = {warm_results['ndcg@10']:.6f}")

if test_cold_data:
    cold_results = evaluate(model, test_cold_data, NUM_ITEMS, MAXLEN)
    print("\ntest_cold 分组结果:")
    print(f"  MRR     = {cold_results['mrr']:.6f}")
    print(f"  Hit@1   = {cold_results['hit@1']:.6f}")
    print(f"  Hit@10  = {cold_results['hit@10']:.6f}")
    print(f"  NDCG@10 = {cold_results['ndcg@10']:.6f}")

print("\n" + "="*70)
print("完成!")
print("="*70)
