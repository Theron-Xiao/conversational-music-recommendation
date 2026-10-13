import hashlib
import json
import math
import pickle
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(".tmp/pydeps").resolve()))

import autograd.numpy as anp
import numpy as np
import pandas as pd
from autograd import grad


CACHE_DIR = Path(".cache_sasrec")
PICKLE_DIR = CACHE_DIR / "pickles"
FILE_DIR = CACHE_DIR / "files"
MAXLEN = 10
D_MODEL = 32
USER_META_DIM = 16
REG = 1e-5
EPOCHS = 30
BATCH_SIZE = 128
MAX_TRAIN_SAMPLES = 40000
SEED = 42

TRACK_EMBEDDINGS_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
USER_EMBEDDINGS_DATASET = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
TRACK_METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
USER_METADATA_DATASET = "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
CHALLENGE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def ensure_dirs():
    CACHE_DIR.mkdir(exist_ok=True)
    PICKLE_DIR.mkdir(exist_ok=True)
    FILE_DIR.mkdir(exist_ok=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def pickle_path(name: str) -> Path:
    return PICKLE_DIR / name


def load_or_build(name: str, builder):
    ensure_dirs()
    path = pickle_path(name)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = builder()
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def get_parquet_urls(dataset: str, split: str):
    manifest_url = f"https://huggingface.co/api/datasets/{dataset}/parquet"
    with urllib.request.urlopen(manifest_url, timeout=60) as resp:
        data = json.load(resp)
    return data["default"][split]


def cached_file_path(url: str) -> Path:
    name = hashlib.md5(url.encode("utf-8")).hexdigest() + ".parquet"
    return FILE_DIR / name


def download_file(url: str) -> Path:
    ensure_dirs()
    path = cached_file_path(url)
    if path.exists() and path.stat().st_size > 0:
        return path
    last_error = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if path.stat().st_size > 0:
                return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if path.exists():
                path.unlink()
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def read_parquet_split(dataset: str, split: str, columns=None):
    urls = get_parquet_urls(dataset, split)
    local_paths = [download_file(url) for url in urls]
    frames = [pd.read_parquet(path, columns=columns) for path in local_paths]
    return pd.concat(frames, ignore_index=True)


def normalized(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_item_embeddings():
    item_vecs = {}
    train_df = read_parquet_split(TRACK_EMBEDDINGS_DATASET, "all_tracks", columns=["track_id", "cf-bpr"])
    test_df = read_parquet_split(TRACK_EMBEDDINGS_DATASET, "test_tracks", columns=["track_id", "cf-bpr"])
    train_item_ids = train_df["track_id"].tolist()
    candidate_item_ids = test_df["track_id"].tolist()
    for frame in (train_df, test_df):
        for track_id, vec in zip(frame["track_id"], frame["cf-bpr"]):
            vec = np.asarray(vec, dtype=np.float32)
            if vec.size == 0:
                vec = np.zeros(128, dtype=np.float32)
            item_vecs[track_id] = normalized(vec)
    return {"item_vecs": item_vecs, "train_item_ids": train_item_ids, "candidate_item_ids": candidate_item_ids}


def build_user_embeddings():
    user_vecs = {}
    for split in ("train", "test_warm", "test_cold"):
        df = read_parquet_split(USER_EMBEDDINGS_DATASET, split, columns=["user_id", "cf-bpr"])
        for user_id, vec in zip(df["user_id"], df["cf-bpr"]):
            vec = np.asarray(vec, dtype=np.float32)
            if vec.size == 0:
                vec = np.zeros(128, dtype=np.float32)
            user_vecs[user_id] = normalized(vec)
    return user_vecs


def build_track_metadata():
    df = read_parquet_split(
        TRACK_METADATA_DATASET,
        "test_tracks",
        columns=["track_id", "popularity", "release_date", "duration"],
    )
    return {
        row["track_id"]: {
            "popularity": float(row["popularity"]) if pd.notna(row["popularity"]) else 0.0,
            "release_date": row["release_date"] if pd.notna(row["release_date"]) else "2000-01-01",
            "duration": float(row["duration"]) if pd.notna(row["duration"]) else 0.0,
        }
        for _, row in df.iterrows()
    }


def build_user_metadata():
    df = read_parquet_split(
        USER_METADATA_DATASET,
        "all_users",
        columns=["user_id", "age_group", "gender", "country_code"],
    )
    return {
        row["user_id"]: {
            "age_group": row["age_group"] if pd.notna(row["age_group"]) else "",
            "gender": row["gender"] if pd.notna(row["gender"]) else "",
            "country_code": row["country_code"] if pd.notna(row["country_code"]) else "",
        }
        for _, row in df.iterrows()
    }


def build_sessions(split: str):
    return read_parquet_split(CHALLENGE_DATASET, split).to_dict("records")


def user_meta_vector(user_id: str, challenge_profile, user_metadata):
    vec = np.zeros(USER_META_DIM, dtype=np.float32)
    meta = user_metadata.get(user_id, {})
    values = {
        "age_group": meta.get("age_group") or (challenge_profile or {}).get("age_group") or "",
        "gender": meta.get("gender") or (challenge_profile or {}).get("gender") or "",
        "country_code": meta.get("country_code") or (challenge_profile or {}).get("country_code") or "",
        "culture": (challenge_profile or {}).get("preferred_musical_culture") or "",
        "language": (challenge_profile or {}).get("preferred_language") or "",
    }
    for key, value in values.items():
        if not value:
            continue
        bucket = stable_hash(f"{key}:{value}") % USER_META_DIM
        sign = 1.0 if (stable_hash(f"sign:{key}:{value}") % 2 == 0) else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def candidate_bias_features(candidate_item_ids, track_metadata):
    pop = []
    dur = []
    year = []
    for track_id in candidate_item_ids:
        row = track_metadata.get(track_id, {})
        pop.append(float(row.get("popularity") or 0.0))
        dur.append(float(row.get("duration") or 0.0))
        release = row.get("release_date") or "2000-01-01"
        try:
            year.append(float(str(release)[:4]))
        except Exception:  # noqa: BLE001
            year.append(2000.0)
    def scale(x):
        x = np.asarray(x, dtype=np.float32)
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        return np.zeros_like(x)
    return {"pop": scale(pop), "dur": scale(dur), "year": scale(year)}


def extract_music_sequence(session):
    return [entry["content"] for entry in session["conversations"] if entry["role"] == "music"]


def build_datasets(item_info, user_vecs, user_metadata):
    train_sessions = load_or_build("challenge_train.pkl", lambda: build_sessions("train"))
    test_sessions = load_or_build("challenge_test.pkl", lambda: build_sessions("test"))
    train_ids_set = set(item_info["train_item_ids"])
    candidate_ids_set = set(item_info["candidate_item_ids"])
    all_known = set(item_info["item_vecs"].keys())

    item_to_idx = {"<PAD>": 0}
    idx_to_item = ["<PAD>"]
    for track_id in item_info["train_item_ids"] + item_info["candidate_item_ids"]:
        if track_id not in item_to_idx:
            item_to_idx[track_id] = len(idx_to_item)
            idx_to_item.append(track_id)

    item_matrix = np.zeros((len(idx_to_item), 128), dtype=np.float32)
    for track_id, idx in item_to_idx.items():
        if idx > 0:
            item_matrix[idx] = item_info["item_vecs"][track_id]

    user_features = {}
    def get_user_features(user_id, challenge_profile):
        if user_id not in user_features:
            emb = user_vecs.get(user_id, np.zeros(128, dtype=np.float32))
            meta_vec = user_meta_vector(user_id, challenge_profile, user_metadata)
            user_features[user_id] = np.concatenate([emb, meta_vec]).astype(np.float32)
        return user_features[user_id]

    train_samples = []
    rng = np.random.default_rng(SEED)
    for session in train_sessions:
        user_feat = get_user_features(session["user_id"], session.get("user_profile") or {})
        seq = [tid for tid in extract_music_sequence(session) if tid in train_ids_set]
        if not seq:
            continue
        history = []
        for target_tid in seq:
            train_samples.append(([item_to_idx[x] for x in history[-MAXLEN:]], item_to_idx[target_tid], user_feat))
            history.append(target_tid)
    if len(train_samples) > MAX_TRAIN_SAMPLES:
        chosen = rng.choice(len(train_samples), size=MAX_TRAIN_SAMPLES, replace=False)
        train_samples = [train_samples[int(i)] for i in chosen]

    test_events = []
    for session in test_sessions:
        split = (session.get("user_profile") or {}).get("user_split", "test_warm")
        user_feat = get_user_features(session["user_id"], session.get("user_profile") or {})
        history = []
        for target_tid in extract_music_sequence(session):
            if target_tid not in candidate_ids_set:
                if target_tid in all_known:
                    history.append(target_tid)
                continue
            test_events.append(([item_to_idx[x] for x in history[-MAXLEN:] if x in item_to_idx], item_to_idx[target_tid], user_feat, split))
            history.append(target_tid)

    candidate_indices = np.array([item_to_idx[x] for x in item_info["candidate_item_ids"]], dtype=np.int32)
    return {
        "item_matrix": item_matrix,
        "train_samples": train_samples,
        "test_events": test_events,
        "candidate_indices": candidate_indices,
        "candidate_item_ids": item_info["candidate_item_ids"],
        "user_feat_dim": 128 + USER_META_DIM,
    }


def init_params(item_feat_dim, user_feat_dim):
    rng = np.random.default_rng(SEED)
    scale = 0.05
    return {
        "W_item": rng.normal(0, scale, size=(item_feat_dim, D_MODEL)).astype(np.float32),
        "W_user": rng.normal(0, scale, size=(user_feat_dim, D_MODEL)).astype(np.float32),
        "W_q": rng.normal(0, scale, size=(D_MODEL, D_MODEL)).astype(np.float32),
        "W_k": rng.normal(0, scale, size=(D_MODEL, D_MODEL)).astype(np.float32),
        "W_v": rng.normal(0, scale, size=(D_MODEL, D_MODEL)).astype(np.float32),
        "W_o": rng.normal(0, scale, size=(D_MODEL, D_MODEL)).astype(np.float32),
        "pos_emb": rng.normal(0, scale, size=(MAXLEN, D_MODEL)).astype(np.float32),
        "start": rng.normal(0, scale, size=(D_MODEL,)).astype(np.float32),
    }


def pad_sequences(seq_list):
    arr = np.zeros((len(seq_list), MAXLEN), dtype=np.int32)
    lens = np.zeros(len(seq_list), dtype=np.int32)
    for i, seq in enumerate(seq_list):
        seq = seq[-MAXLEN:]
        lens[i] = len(seq)
        if seq:
            arr[i, -len(seq):] = np.asarray(seq, dtype=np.int32)
    return arr, lens


def encode_state(params, item_matrix, seq_idx, seq_len, user_feat):
    item_feats = item_matrix[seq_idx]
    H = anp.einsum("blf,fd->bld", item_feats, params["W_item"])
    mask = (seq_idx != 0).astype(anp.float32)
    H = H + params["pos_emb"][None, :, :] * mask[:, :, None]
    Q = anp.einsum("bld,df->blf", H, params["W_q"])
    K = anp.einsum("bld,df->blf", H, params["W_k"])
    V = anp.einsum("bld,df->blf", H, params["W_v"])
    batch = seq_idx.shape[0]
    last_pos = anp.clip(MAXLEN - seq_len, 0, MAXLEN - 1)
    q = Q[anp.arange(batch), last_pos]
    logits = anp.einsum("bd,bld->bl", q, K) / math.sqrt(D_MODEL)
    logits = anp.where(mask > 0, logits, -1e9)
    logits = logits - anp.max(logits, axis=1, keepdims=True)
    exp_logits = anp.exp(logits) * mask
    attn = exp_logits / (anp.sum(exp_logits, axis=1, keepdims=True) + 1e-8)
    context = anp.einsum("bl,bld->bd", attn, V)
    has_hist = (seq_len > 0).astype(anp.float32)[:, None]
    last_emb = H[anp.arange(batch), last_pos] * has_hist
    user_proj = anp.dot(user_feat, params["W_user"])
    return anp.tanh(anp.dot(context, params["W_o"]) + last_emb + user_proj + params["start"])


def batch_loss(params, item_matrix, seq_idx, seq_len, user_feat, pos_idx, neg_idx):
    state = encode_state(params, item_matrix, seq_idx, seq_len, user_feat)
    proj_items = anp.dot(item_matrix, params["W_item"])
    pos_vec = proj_items[pos_idx]
    neg_vec = proj_items[neg_idx]
    diff = anp.sum(state * pos_vec, axis=1) - anp.sum(state * neg_vec, axis=1)
    reg = REG * sum(anp.sum(v * v) for v in params.values())
    return -anp.mean(-anp.logaddexp(0.0, -diff)) + reg


def train_sasrec(dataset):
    item_matrix = anp.asarray(dataset["item_matrix"])
    params = {k: anp.asarray(v) for k, v in init_params(item_matrix.shape[1], dataset["user_feat_dim"]).items()}
    loss_grad = grad(batch_loss)
    rng = np.random.default_rng(SEED)
    train_samples = dataset["train_samples"]
    train_item_indices = np.unique(np.asarray([x[1] for x in train_samples], dtype=np.int32))
    m = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
    v = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
    step = 0
    for _ in range(EPOCHS):
        rng.shuffle(train_samples)
        for start in range(0, len(train_samples), BATCH_SIZE):
            batch = train_samples[start : start + BATCH_SIZE]
            seq_idx, seq_len = pad_sequences([x[0] for x in batch])
            user_feat = np.stack([x[2] for x in batch]).astype(np.float32)
            pos_idx = np.asarray([x[1] for x in batch], dtype=np.int32)
            neg_idx = rng.choice(train_item_indices, size=len(batch), replace=True)
            same = neg_idx == pos_idx
            while np.any(same):
                neg_idx[same] = rng.choice(train_item_indices, size=int(np.sum(same)), replace=True)
                same = neg_idx == pos_idx
            grads = loss_grad(params, item_matrix, anp.asarray(seq_idx), anp.asarray(seq_len), anp.asarray(user_feat), anp.asarray(pos_idx), anp.asarray(neg_idx))
            step += 1
            lr = 0.003
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            updated = {}
            for key in params:
                g = np.asarray(grads[key], dtype=np.float32)
                m[key] = beta1 * m[key] + (1 - beta1) * g
                v[key] = beta2 * v[key] + (1 - beta2) * (g * g)
                m_hat = m[key] / (1 - beta1 ** step)
                v_hat = v[key] / (1 - beta2 ** step)
                updated[key] = np.asarray(params[key]) - lr * m_hat / (np.sqrt(v_hat) + eps)
            params = {k: anp.asarray(val) for k, val in updated.items()}
    return {k: np.asarray(v) for k, v in params.items()}


def evaluate_sasrec(params, dataset, candidate_bias):
    item_matrix = anp.asarray(dataset["item_matrix"])
    candidate_indices = dataset["candidate_indices"]
    proj_items = np.asarray(item_matrix @ params["W_item"])
    candidate_proj = proj_items[candidate_indices]
    bias = 0.05 * candidate_bias["pop"] + 0.01 * candidate_bias["year"] + 0.01 * candidate_bias["dur"]
    candidate_pos = {int(idx): pos for pos, idx in enumerate(candidate_indices)}
    metrics = {
        "overall": {"count": 0, "mrr": 0.0, "hit1": 0, "hit10": 0, "ndcg10": 0.0},
        "test_warm": {"count": 0, "mrr": 0.0, "hit1": 0, "hit10": 0, "ndcg10": 0.0},
        "test_cold": {"count": 0, "mrr": 0.0, "hit1": 0, "hit10": 0, "ndcg10": 0.0},
    }
    for start in range(0, len(dataset["test_events"]), 256):
        batch = dataset["test_events"][start : start + 256]
        seq_idx, seq_len = pad_sequences([x[0] for x in batch])
        user_feat = np.stack([x[2] for x in batch]).astype(np.float32)
        state = np.asarray(encode_state({k: anp.asarray(v) for k, v in params.items()}, item_matrix, anp.asarray(seq_idx), anp.asarray(seq_len), anp.asarray(user_feat)))
        scores = state @ candidate_proj.T + bias[None, :]
        for row_i, (_, target_idx, _, split) in enumerate(batch):
            pos = candidate_pos[int(target_idx)]
            target_score = float(scores[row_i, pos])
            rank = 1 + int(np.count_nonzero(scores[row_i] > target_score))
            for key in ("overall", split):
                if key in metrics:
                    metrics[key]["count"] += 1
                    metrics[key]["mrr"] += 1.0 / rank
                    metrics[key]["hit1"] += int(rank <= 1)
                    metrics[key]["hit10"] += int(rank <= 10)
                    metrics[key]["ndcg10"] += (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0
    for val in metrics.values():
        if val["count"]:
            val["mrr"] /= val["count"]
            val["hit1"] /= val["count"]
            val["hit10"] /= val["count"]
            val["ndcg10"] /= val["count"]
    return metrics


def main():
    set_global_seed(SEED)
    item_info = load_or_build("item_info.pkl", build_item_embeddings)
    user_vecs = load_or_build("user_embeddings.pkl", build_user_embeddings)
    track_metadata = load_or_build("track_metadata.pkl", build_track_metadata)
    user_metadata = load_or_build("user_metadata.pkl", build_user_metadata)
    dataset = load_or_build("dataset.pkl", lambda: build_datasets(item_info, user_vecs, user_metadata))
    params = train_sasrec(dataset)
    metrics = evaluate_sasrec(params, dataset, candidate_bias_features(dataset["candidate_item_ids"], track_metadata))
    output = {
        "config": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "maxlen": MAXLEN,
            "d_model": D_MODEL,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "reg": REG,
        },
        "metrics": metrics,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
