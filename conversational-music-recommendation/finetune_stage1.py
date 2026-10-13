"""
Stage 1 对比学习微调：在 TalkPlay 数据上微调 MiniLM。

原理:
  - 正例对: (context_text, thought) — 同一条 music turn 的上下文和推荐理由
  - 负例: 同 batch 内其他样本的 thought (in-batch negatives)
  - Loss: InfoNCE — 拉近正例对，推开负例

训练后导出 fine-tuned 模型，再重新运行 preprocess_thoughts.py 重建索引。
预期 MRR: 0.06 → 0.10~0.15
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

from config.default_config import (
    PROXY_ENV, DEVICE, TEXT_MODEL_NAME, TEXT_DIM,
)

os.environ.update(PROXY_ENV)


# ── 数据集 ──

class ContrastivePairsDataset(Dataset):
    """从训练集提取 (context_text, thought) 正例对。"""

    def __init__(self, split='train', max_pairs=None):
        print(f"加载 {split} 集，提取 context-thought 对...")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
        data = ds[split]

        self.contexts = []
        self.thoughts = []

        for row in tqdm(data, desc=f"扫描 {split}"):
            context_text = ""
            for turn in row['conversations']:
                role = turn.get('role', '')
                content = turn.get('content', '')

                if role == 'user':
                    context_text += f" User: {content}"
                elif role == 'assistant':
                    context_text += f" System: {content}"
                elif role == 'music':
                    thought = turn.get('thought', '')
                    if thought and context_text.strip():
                        self.contexts.append(context_text.strip())
                        self.thoughts.append(thought)
                    # 不更新 context_text（music turn 的 content 是 track_id）

        # 去重（同一 context + thought 可能重复出现）
        pairs = list(set(zip(self.contexts, self.thoughts)))
        self.contexts = [p[0] for p in pairs]
        self.thoughts = [p[1] for p in pairs]

        if max_pairs and len(self.contexts) > max_pairs:
            indices = np.random.choice(len(self.contexts), max_pairs, replace=False)
            self.contexts = [self.contexts[i] for i in indices]
            self.thoughts = [self.thoughts[i] for i in indices]

        print(f"  正例对: {len(self.contexts)}")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.thoughts[idx]


# ── 模型 ──

class ContrastiveMiniLM(nn.Module):
    """MiniLM 直接在 CLS 输出（384d）上做对比学习，与 text_utils 编码路径一致。"""

    def __init__(self, model_name=TEXT_MODEL_NAME, temperature=0.07):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature

    def forward(self, context_texts, thought_texts):
        """
        InfoNCE 对比学习。将 context + thought 合并为一个 batch 编码，省一半时间。
        """
        device = next(self.encoder.parameters()).device
        B = len(context_texts)

        # 合并编码：2B 个文本一次过
        all_texts = list(context_texts) + list(thought_texts)
        tokens = self.tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        ).to(device)

        outputs = self.encoder(**tokens)
        all_emb = outputs.last_hidden_state[:, 0, :]  # (2B, 384)
        all_emb = nn.functional.normalize(all_emb, dim=-1)

        ctx_emb = all_emb[:B]   # (B, 384)
        th_emb = all_emb[B:]    # (B, 384)

        # InfoNCE
        logits = ctx_emb @ th_emb.T / self.temperature
        labels = torch.arange(B, device=device)
        loss = (nn.functional.cross_entropy(logits, labels)
                + nn.functional.cross_entropy(logits.T, labels)) / 2
        acc = ((logits.argmax(dim=1) == labels).float().mean()
               + (logits.T.argmax(dim=1) == labels).float().mean()) / 2

        return loss, acc


# ── 训练 ──

def train():
    # 超参
    BATCH_SIZE = 128
    EPOCHS = 3
    LR = 2e-5
    TEMPERATURE = 0.07
    MAX_PAIRS = 80000  # 采样以加速训练

    print("=" * 60)
    print("Stage 1 对比学习微调（直接在 CLS 384d 上训练）")
    print(f"模型: {TEXT_MODEL_NAME}")
    print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 60)

    # 数据
    dataset = ContrastivePairsDataset(split='train', max_pairs=MAX_PAIRS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Batches/epoch: {len(loader)}\n")

    # 模型（无投影头，直接在 CLS 384d 上训练）
    model = ContrastiveMiniLM(
        model_name=TEXT_MODEL_NAME,
        temperature=TEMPERATURE,
    ).to(DEVICE)
    print(f"Encoder 参数量: {sum(p.numel() for p in model.encoder.parameters()):,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (contexts, thoughts) in enumerate(pbar):
            optimizer.zero_grad()
            loss, acc = model(contexts, thoughts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.3f}'})

        avg_loss = epoch_loss / len(loader)
        avg_acc = epoch_acc / len(loader)
        print(f"[Epoch {epoch+1}] 平均 Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            model.encoder.save_pretrained('models/minilm_finetuned')
            model.tokenizer.save_pretrained('models/minilm_finetuned')
            print(f"  → 保存最佳模型至 models/minilm_finetuned/")

    print(f"\n[OK] 微调完成！最佳 Loss: {best_loss:.4f}")
    print("下一步: 用 fine-tuned 模型重新运行 preprocess_thoughts.py 重建索引")
    print("       然后重新运行 stage1_thought_retrieval.py 查看 MRR")


if __name__ == '__main__':
    train()
