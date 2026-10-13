"""
GRU4Rec Evaluation on TalkPlayData Challenge Dataset
=====================================================
Evaluates the GRU4Rec session-based recommendation model on the
TalkPlayData-Challenge conversational music recommendation dataset.

Metrics: MRR@K, Hit@1, Hit@10

Datasets used:
  1. talkpl-ai/TalkPlayData-Challenge-Dataset (conversations with sessions)
  2. talkpl-ai/TalkPlayData-Challenge-User-Embeddings (user cf-bpr embeddings)
  3. talkpl-ai/TalkPlayData-Challenge-Track-Embeddings (track multimodal embeddings)

Approach:
  - Each conversation = a session
  - Music turns = recommended tracks
  - goal_progress_assessment: MOVES_TOWARD_GOAL → liked (1),
    DOES_NOT_MOVE_TOWARD_GOAL → not liked (0)
  - GRU4Rec predicts the next liked track given the liked-track history
"""

import os

# MUST set before any huggingface imports to skip network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # fallback mirror

import sys
import random
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    hf_endpoint = "https://hf-mirror.com"  # Set to None if you can access HF directly

    # Model
    embedding_dim = 128       # Item embedding dimension (matches cf-bpr dim)
    hidden_dim = 128          # GRU hidden dimension
    num_layers = 1            # Number of GRU layers
    dropout = 0.3             # Dropout rate

    # Training
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    negative_samples = 20     # Number of negative samples per positive
    loss_type = "bpr_max"     # "bpr_max", "top1_max", "cross_entropy"

    # Evaluation
    k_values = [1, 10]        # K for Hit@K

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    data_dir = "./talkplay_data"

    # Min liked items needed for a session to be usable
    min_liked_items = 2

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.seed)

if Config.hf_endpoint:
    os.environ["HF_ENDPOINT"] = Config.hf_endpoint

# Skip network checks — use cached data directly
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_and_preprocess_data():
    """
    Load all three datasets and build session data.

    Returns:
        train_sessions: list of lists, each inner list = [track_id_1, track_id_2, ...]
                        tracks that the user liked (goal_progress = MOVES_TOWARD_GOAL)
        test_sessions: same structure for test set
        track_id_to_idx: mapping from track_id string to integer index
        idx_to_track_id: reverse mapping
        track_features: tensor of shape (num_tracks, embedding_dim) — pre-trained cf-bpr
    """
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # 1. Load conversation dataset
    ds_main = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    train_conv = ds_main["train"]
    test_conv = ds_main["test"]
    print(f"  Conversations: train={len(train_conv)}, test={len(test_conv)}")

    # 2. Load track embeddings
    ds_track = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    print(f"  Tracks: all={len(ds_track['all_tracks'])}")

    # 3. Load user embeddings
    ds_user = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    print(f"  Users: train={len(ds_user['train'])}, test_warm={len(ds_user.get('test_warm', []))}, test_cold={len(ds_user.get('test_cold', []))}")

    # Build track features dictionary (track_id → cf-bpr embedding)
    print("\nBuilding track feature dictionary...")
    track_features_dict = {}
    for row in ds_track["all_tracks"]:
        track_id = row["track_id"]
        cf_bpr = torch.tensor(row["cf-bpr"], dtype=torch.float32)
        track_features_dict[track_id] = cf_bpr

    # Also add test_tracks
    if "test_tracks" in ds_track:
        for row in ds_track["test_tracks"]:
            track_id = row["track_id"]
            if track_id not in track_features_dict:
                cf_bpr = torch.tensor(row["cf-bpr"], dtype=torch.float32)
                track_features_dict[track_id] = cf_bpr

    print(f"  Total unique tracks with embeddings: {len(track_features_dict)}")

    # 4. Extract sessions from conversations
    def extract_sessions(conversation_dataset, split_name):
        """
        For each conversation, extract:
          - all tracks from music turns with liked labels
          - session of liked-only tracks for GRU4Rec

        Returns:
            all_sessions: list of dicts with 'session_id', 'user_id', 'track_sequence', 'liked_sequence', 'labels'
            track_id_set: set of all track IDs seen
        """
        sessions = []
        track_id_set = set()
        total_music_turns = 0
        total_liked = 0
        skipped_no_assessment = 0

        for row_idx, row in enumerate(conversation_dataset):
            session_id = row["session_id"]
            user_id = row["user_id"]
            conversations = row["conversations"]
            assessments = row["goal_progress_assessments"]

            # Extract music turns (track_ids)
            music_turns = []
            for turn in conversations:
                if turn["role"] == "music":
                    music_turns.append({
                        "track_id": turn["content"],  # content IS the track UUID
                        "turn_number": turn["turn_number"],
                    })

            # Map assessments by turn_number (1-indexed)
            assess_by_turn = {}
            for a in assessments:
                assess_by_turn[a["turn_number"]] = a["goal_progress_assessment"]

            # Assign labels to music turns
            # Music turn at turn_number T → assessment at turn_number T+1
            track_sequence = []
            liked_sequence = []
            labels = []

            for mt in music_turns:
                track_id = mt["track_id"]
                turn_num = mt["turn_number"]

                # Assessment for this music turn is at turn_num+1
                next_turn = turn_num + 1
                assessment = assess_by_turn.get(next_turn, None)

                track_sequence.append(track_id)
                track_id_set.add(track_id)
                total_music_turns += 1

                if assessment == "MOVES_TOWARD_GOAL":
                    liked = 1
                    total_liked += 1
                elif assessment == "DOES_NOT_MOVE_TOWARD_GOAL":
                    liked = 0
                else:
                    # No assessment (usually last turn): skip from liked-only analysis
                    liked = 0
                    skipped_no_assessment += 1

                labels.append(liked)
                if liked == 1:
                    liked_sequence.append(track_id)

            if len(liked_sequence) >= Config.min_liked_items or len(track_sequence) >= 2:
                sessions.append({
                    "session_id": session_id,
                    "user_id": user_id,
                    "track_sequence": track_sequence,      # All recommended tracks
                    "liked_sequence": liked_sequence,       # Only liked tracks
                    "labels": labels,                       # 0/1 for each track
                })

        print(f"\n  [{split_name}] Statistics:")
        print(f"    Sessions: {len(sessions)}")
        print(f"    Total music turns: {total_music_turns}")
        print(f"    Liked turns: {total_liked} ({100*total_liked/max(1,total_music_turns):.1f}%)")
        print(f"    Skipped (no assessment): {skipped_no_assessment}")
        avg_liked = np.mean([len(s["liked_sequence"]) for s in sessions])
        print(f"    Avg liked tracks per session: {avg_liked:.2f}")

        return sessions, track_id_set

    print("\n" + "=" * 60)
    print("Extracting sessions...")
    print("=" * 60)

    train_sessions, train_tracks = extract_sessions(train_conv, "train")
    test_sessions, test_tracks = extract_sessions(test_conv, "test")

    # Build unified track ID mapping
    all_track_ids = train_tracks | test_tracks
    # Also include tracks from embeddings dataset for full catalog
    all_track_ids |= set(track_features_dict.keys())

    track_id_list = sorted(all_track_ids)
    track_id_to_idx = {tid: idx for idx, tid in enumerate(track_id_list)}
    idx_to_track_id = {idx: tid for tid, idx in track_id_to_idx.items()}
    num_tracks = len(track_id_list)
    print(f"\n  Total unique tracks in catalog: {num_tracks}")

    # Build track feature matrix
    track_features = torch.zeros(num_tracks, Config.embedding_dim)
    missing_embeddings = 0
    empty_embeddings = 0
    for tid, idx in track_id_to_idx.items():
        if tid in track_features_dict:
            emb = track_features_dict[tid]
            if emb.shape[0] == Config.embedding_dim:
                track_features[idx] = emb
            else:
                # Empty or wrong-size embedding
                track_features[idx] = torch.randn(Config.embedding_dim) * 0.01
                empty_embeddings += 1
        else:
            # Random initialization for tracks without pre-trained embeddings
            track_features[idx] = torch.randn(Config.embedding_dim) * 0.01
            missing_embeddings += 1
    if missing_embeddings > 0:
        print(f"  Note: {missing_embeddings} tracks without pre-trained embeddings (random init)")
    if empty_embeddings > 0:
        print(f"  Note: {empty_embeddings} tracks with empty embeddings (random init)")

    # Convert session track IDs to indices
    train_data = []
    for s in train_sessions:
        liked_idxs = [track_id_to_idx[tid] for tid in s["liked_sequence"]]
        track_idxs = [track_id_to_idx[tid] for tid in s["track_sequence"]]
        if len(liked_idxs) >= Config.min_liked_items:
            train_data.append({
                "session_id": s["session_id"],
                "user_id": s["user_id"],
                "liked_sequence": liked_idxs,
                "track_sequence": track_idxs,
                "labels": s["labels"],
            })

    test_data = []
    for s in test_sessions:
        liked_idxs = [track_id_to_idx[tid] for tid in s["liked_sequence"]]
        track_idxs = [track_id_to_idx[tid] for tid in s["track_sequence"]]
        test_data.append({
            "session_id": s["session_id"],
            "user_id": s["user_id"],
            "liked_sequence": liked_idxs,
            "track_sequence": track_idxs,
            "labels": s["labels"],
        })

    print(f"\n  Final: train sessions={len(train_data)}, test sessions={len(test_data)}")

    return train_data, test_data, num_tracks, track_id_to_idx, idx_to_track_id, track_features


# ============================================================================
# GRU4Rec Model
# ============================================================================

class GRU4Rec(nn.Module):
    """
    GRU4Rec: Session-based recommendation with GRU.

    Architecture:
      - Item embedding layer (optionally initialized with pre-trained embeddings)
      - GRU layer(s) for session encoding
      - Output projection for scoring
    """

    def __init__(self, num_items, embedding_dim, hidden_dim, num_layers=1,
                 dropout=0.3, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Item embedding
        if pretrained_embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings
            )
        else:
            self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        # GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer: project hidden state to scores over all items
        # Score = item_embedding^T * (W * h_t)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq, seq_lengths=None):
        """
        Args:
            item_seq: (batch_size, max_seq_len) — sequence of item indices
            seq_lengths: (batch_size,) — actual sequence lengths

        Returns:
            session_embedding: (batch_size, hidden_dim) — final GRU hidden state
        """
        # Embed items: (batch_size, seq_len, embedding_dim)
        emb = self.item_embedding(item_seq)
        emb = self.dropout(emb)

        # Pack padded sequence if lengths provided
        if seq_lengths is not None:
            emb_packed = nn.utils.rnn.pack_padded_sequence(
                emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(emb_packed)
        else:
            _, hidden = self.gru(emb)

        # hidden: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        session_embedding = hidden[-1]  # (batch_size, hidden_dim)

        return session_embedding

    def compute_scores(self, session_embedding, item_indices=None):
        """
        Compute scores for items given session embedding.

        Args:
            session_embedding: (batch_size, hidden_dim)
            item_indices: (batch_size, num_candidates) or None (all items)

        Returns:
            scores: (batch_size, num_candidates)
        """
        # Project session to embedding space
        # (batch_size, embedding_dim)
        projected = self.output_proj(session_embedding)

        if item_indices is not None:
            # Score specific items
            # item_emb: (batch_size, num_candidates, embedding_dim)
            item_emb = self.item_embedding(item_indices)
            # Dot product: (batch_size, num_candidates)
            scores = (projected.unsqueeze(1) * item_emb).sum(-1)
        else:
            # Score all items
            # (batch_size, num_items)
            scores = torch.matmul(projected, self.item_embedding.weight.T)

        return scores


# ============================================================================
# Training
# ============================================================================

class SessionDataset(Dataset):
    """Dataset for training GRU4Rec."""

    def __init__(self, sessions, num_items):
        """
        Args:
            sessions: list of dicts with 'liked_sequence' (list of item indices)
            num_items: total number of items in catalog
        """
        self.num_items = num_items

        # Build training examples: for each session with liked sequence [i1, i2, ..., in],
        # create examples: (input=[], target=i1), (input=[i1], target=i2), ...
        self.examples = []

        for session in sessions:
            seq = session["liked_sequence"]
            # Need at least 2 items: 1 for input, 1 for target
            for pos in range(1, len(seq)):
                input_seq = seq[:pos]  # items before position pos
                target = seq[pos]       # item at position pos
                self.examples.append({
                    "input_seq": input_seq,
                    "target": target,
                    "session_id": session["session_id"],
                })

        print(f"  SessionDataset: {len(self.examples)} training examples from {len(sessions)} sessions")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, num_items, max_seq_len=50):
    """
    Collate function for DataLoader.
    Pads input sequences and samples negative items.
    """
    batch_size = len(batch)

    # Pad input sequences
    seq_lengths = [min(len(ex["input_seq"]), max_seq_len) for ex in batch]
    max_len = max(seq_lengths) if seq_lengths else 1

    padded_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, ex in enumerate(batch):
        seq = ex["input_seq"][-max_seq_len:]  # Take last max_seq_len items
        padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        targets[i] = ex["target"]

    # Sample negative items for each target
    # Ensure negatives don't include the positive target
    negatives = torch.zeros(batch_size, Config.negative_samples, dtype=torch.long)
    for i in range(batch_size):
        neg_set = set()
        # Also exclude items already in the input sequence
        exclude = set(batch[i]["input_seq"]) | {batch[i]["target"]}
        while len(neg_set) < Config.negative_samples:
            neg = random.randrange(0, num_items)
            if neg not in exclude:
                neg_set.add(neg)
        negatives[i] = torch.tensor(list(neg_set), dtype=torch.long)

    return padded_seqs, torch.tensor(seq_lengths), targets, negatives


def bpr_max_loss(positive_scores, negative_scores):
    """
    BPR-max loss from "Recurrent Neural Networks with Top-k Gains
    for Session-based Recommendations" (CIKM 2018).

    Instead of averaging over negatives, uses softmax to focus
    on the hardest (highest-scoring) negative.

    Args:
        positive_scores: (batch_size,) - scores for positive items
        negative_scores: (batch_size, num_negatives) - scores for negative items

    Returns:
        loss: scalar
    """
    # Softmax over negative scores
    # (batch_size, num_negatives)
    neg_softmax = F.softmax(negative_scores, dim=-1)

    # Weighted sum of sigmoid(-(pos - neg))
    # (batch_size, num_negatives)
    diff = positive_scores.unsqueeze(1) - negative_scores
    loss_per_neg = -F.logsigmoid(diff)

    # Apply softmax weights
    loss = (neg_softmax * loss_per_neg).sum(dim=-1).mean()

    return loss


def top1_max_loss(positive_scores, negative_scores):
    """
    TOP1-max loss from CIKM 2018 paper.
    """
    # Softmax over negative scores
    neg_softmax = F.softmax(negative_scores, dim=-1)

    # TOP1 loss: sigmoid(neg - pos) + sigmoid(neg^2)
    diff = negative_scores - positive_scores.unsqueeze(1)
    loss_per_neg = F.sigmoid(diff) + F.sigmoid(negative_scores ** 2)

    # Apply softmax weights
    loss = (neg_softmax * loss_per_neg).sum(dim=-1).mean()

    return loss


def cross_entropy_loss(positive_scores, negative_scores):
    """
    Cross-entropy loss: treat positive vs negatives as multi-class.
    """
    # Concatenate positive and negatives
    # (batch_size, 1 + num_negatives)
    scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    # Softmax over all candidates
    probs = F.softmax(scores, dim=-1)
    # Loss is -log(prob of positive at position 0)
    loss = -torch.log(probs[:, 0] + 1e-10).mean()

    return loss


def train_epoch(model, dataloader, optimizer, num_items, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        padded_seqs, seq_lengths, targets, negatives = batch
        padded_seqs = padded_seqs.to(device)
        seq_lengths = seq_lengths.to(device)
        targets = targets.to(device)
        negatives = negatives.to(device)

        # Get session embeddings
        session_emb = model(padded_seqs, seq_lengths)

        # Compute scores for positive and negative items
        # Positive scores
        pos_scores = model.compute_scores(session_emb, targets.unsqueeze(1)).squeeze(1)

        # Negative scores
        neg_scores = model.compute_scores(session_emb, negatives)

        # Compute loss
        if Config.loss_type == "bpr_max":
            loss = bpr_max_loss(pos_scores, neg_scores)
        elif Config.loss_type == "top1_max":
            loss = top1_max_loss(pos_scores, neg_scores)
        elif Config.loss_type == "cross_entropy":
            loss = cross_entropy_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown loss type: {Config.loss_type}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_sampled(model, test_data, num_items, device, k_values=[1, 10], num_samples=100):
    """
    Evaluate GRU4Rec using SAMPLED metrics (standard approach).

    For each test session, rank the true target against `num_samples` random
    negative items. This matches the standard evaluation in GRU4Rec papers.

    Args:
        model: trained GRU4Rec model
        test_data: list of session dicts
        num_items: total number of items
        device: torch device
        k_values: list of K values for evaluation
        num_samples: number of random negative samples

    Returns:
        metrics: dict with MRR@K and Hit@K for each K
    """
    model.eval()

    hits = {k: 0 for k in k_values}
    reciprocal_ranks = {k: 0.0 for k in k_values}
    total_sessions = 0
    ranks = []

    for session in test_data:
        liked_seq = session["liked_sequence"]

        # Need at least 2 liked items for prediction
        if len(liked_seq) < 2:
            continue

        input_seq = liked_seq[:-1]
        target = liked_seq[-1]

        # Truncate if too long
        if len(input_seq) > 50:
            input_seq = input_seq[-50:]

        # Create input tensor
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        seq_len = torch.tensor([len(input_seq)], dtype=torch.long).to(device)

        # Get session embedding
        session_emb = model(input_tensor, seq_len)

        # Sample random negative items (exclude input items and target)
        exclude_set = set(input_seq) | {target}
        neg_samples = []
        while len(neg_samples) < num_samples:
            neg = random.randrange(0, num_items)
            if neg not in exclude_set:
                neg_samples.append(neg)
                exclude_set.add(neg)

        # Candidates: 1 positive + num_samples negatives
        candidates = torch.tensor([target] + neg_samples, dtype=torch.long).to(device)

        # Compute scores for candidates only
        candidate_scores = model.compute_scores(session_emb, candidates.unsqueeze(0)).squeeze(0)

        # Rank: sort descending, find position of target (which is at index 0)
        sorted_indices = torch.argsort(candidate_scores, descending=True)
        rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

        ranks.append(rank)
        total_sessions += 1

        for k in k_values:
            if rank <= k:
                hits[k] += 1
                reciprocal_ranks[k] += 1.0 / rank

    if total_sessions == 0:
        print("WARNING: No valid test sessions found!")
        return {}, []

    metrics = {}
    for k in k_values:
        metrics[f"Hit@{k}"] = hits[k] / total_sessions
        metrics[f"MRR@{k}"] = reciprocal_ranks[k] / total_sessions

    metrics["sessions_evaluated"] = total_sessions
    metrics["mean_rank"] = np.mean(ranks)
    metrics["median_rank"] = np.median(ranks)
    metrics["num_samples"] = num_samples

    return metrics, ranks


@torch.no_grad()
def evaluate_full_ranking(model, test_data, num_items, device, k_values=[1, 10]):
    """
    全量排序评估 — 在所有曲目中对目标曲目进行排序。

    指标:
      - MRR (Mean Reciprocal Rank): mean(1/rank)，无截断
      - NDCG@10: Normalized Discounted Cumulative Gain @ 10
      - Hit@1, Hit@10
    """
    model.eval()

    ranks = []
    total_sessions = 0
    reciprocal_ranks_sum = 0.0   # 用于计算 MRR (无截断)
    ndcg_10_sum = 0.0            # 用于计算 NDCG@10
    hit_1_count = 0
    hit_10_count = 0

    all_item_embeddings = model.item_embedding.weight

    for session in test_data:
        liked_seq = session["liked_sequence"]

        if len(liked_seq) < 2:
            continue

        input_seq = liked_seq[:-1]
        target = liked_seq[-1]

        if len(input_seq) > 50:
            input_seq = input_seq[-50:]

        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        seq_len = torch.tensor([len(input_seq)], dtype=torch.long).to(device)
        session_emb = model(input_tensor, seq_len)
        projected = model.output_proj(session_emb)
        scores = torch.matmul(projected, all_item_embeddings.T).squeeze(0)

        # 排除输入序列中已出现的曲目
        for item_idx in set(input_seq):
            scores[item_idx] = -float("inf")

        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1

        ranks.append(rank)
        total_sessions += 1

        # MRR (无截断, 整条reciprocal rank取平均)
        reciprocal_ranks_sum += 1.0 / rank

        # Hit@1, Hit@10
        if rank <= 1:
            hit_1_count += 1
        if rank <= 10:
            hit_10_count += 1

        # NDCG@10: DCG@10 / IDCG@10
        # 假设只有目标曲目是相关的 (relevance=1), 其余都是0
        # DCG@10 = 1 / log2(rank+1)  (if rank <= 10 else 0)
        # IDCG@10 = 1 / log2(1+1) = 1  (最优排序: rank=1)
        # NDCG@10 = DCG@10 / IDCG@10 = 1 / log2(rank+1)  (if rank <= 10 else 0)
        if rank <= 10:
            ndcg_10_sum += 1.0 / np.log2(rank + 1)

    if total_sessions == 0:
        print("WARNING: No valid test sessions found!")
        return {}, []

    metrics = {
        "MRR": reciprocal_ranks_sum / total_sessions,
        "NDCG@10": ndcg_10_sum / total_sessions,
        "Hit@1": hit_1_count / total_sessions,
        "Hit@10": hit_10_count / total_sessions,
        "sessions_evaluated": total_sessions,
        "mean_rank": np.mean(ranks),
        "median_rank": np.median(ranks),
    }

    return metrics, ranks


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("  GRU4Rec Evaluation on TalkPlayData Challenge")
    print("=" * 60)
    print(f"\nDevice: {Config.device}")
    print(f"Loss: {Config.loss_type}")
    print(f"Embedding dim: {Config.embedding_dim}")
    print(f"Hidden dim: {Config.hidden_dim}")
    print(f"Batch size: {Config.batch_size}")
    print(f"Epochs: {Config.num_epochs}")
    print(f"Negative samples: {Config.negative_samples}")

    # 1. Load data
    train_data, test_data, num_items, track_id_to_idx, idx_to_track_id, track_features = load_and_preprocess_data()

    # 2. Create dataset and dataloader
    train_dataset = SessionDataset(train_data, num_items)

    # Custom collate function
    def custom_collate(batch):
        return collate_fn(batch, num_items)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,
    )

    # 3. Create model
    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
        pretrained_embeddings=track_features,
        freeze_embeddings=False,
    ).to(Config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # 4. Train
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print("\n" + "=" * 60)
    print("  Training")
    print("=" * 60)

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, Config.num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, num_items, Config.device)
        scheduler.step(loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{Config.num_epochs} | Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

    print(f"\n  Best loss: {best_loss:.4f} at epoch {best_epoch}")

    # 5. Evaluate
    print("\n" + "=" * 60)
    print("  Evaluation (Full Ranking)")
    print("=" * 60)

    metrics, ranks = evaluate_full_ranking(
        model, test_data, num_items, Config.device
    )

    print(f"\n  Sessions evaluated: {metrics.get('sessions_evaluated', 0)}")
    print(f"  Mean Rank: {metrics.get('mean_rank', 0):.2f} / {num_items}")
    print(f"  Median Rank: {metrics.get('median_rank', 0):.2f}")
    print()
    print(f"  {'Metric':<14} {'Value':<12}")
    print(f"  {'-'*14} {'-'*12}")
    print(f"  {'MRR':<14} {metrics.get('MRR', 0):.4f}")
    print(f"  {'NDCG@10':<14} {metrics.get('NDCG@10', 0):.4f}")
    print(f"  {'Hit@1':<14} {metrics.get('Hit@1', 0):.4f}")
    print(f"  {'Hit@10':<14} {metrics.get('Hit@10', 0):.4f}")

    return metrics, ranks


if __name__ == "__main__":
    metrics, ranks = main()
