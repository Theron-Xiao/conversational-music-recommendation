import os
import re
from transformers import AutoTokenizer, AutoModel
import torch
from config.default_config import PROXY_ENV, TEXT_MODEL_NAME, STAGE1_MODEL_PATH, DEVICE

# 激活代理
os.environ.update(PROXY_ENV)

# 延迟初始化加载（避免导包时卡顿）
tokenizer = None
text_encoder = None


def _get_model_name():
    """返回实际使用的模型路径：微调版优先，否则用预训练版。"""
    if STAGE1_MODEL_PATH and os.path.isdir(STAGE1_MODEL_PATH):
        return STAGE1_MODEL_PATH
    return TEXT_MODEL_NAME


def get_text_embedding(text):
    global tokenizer, text_encoder
    if tokenizer is None or text_encoder is None:
        model_name = _get_model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).to(DEVICE)

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        # 搬回 CPU 转为 numpy 供后续处理
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]


def clear_model_cache():
    """清除缓存的模型，强制下次调用重新加载（切换模型时使用）。"""
    global tokenizer, text_encoder
    tokenizer = None
    text_encoder = None


def get_negative_artists_safe(context_string, track_to_artist_dict):
    negative_artists = set()
    trigger_words = ['new artist', 'different artist', 'not heard', 'not mentioned', 'not alesana',
                     'completely different']
    if any(word in context_string.lower() for word in trigger_words):
        historical_tracks = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                                       context_string.lower())
        for tid in historical_tracks:
            if tid in track_to_artist_dict:
                banned_artist = track_to_artist_dict[tid]
                if isinstance(banned_artist, str):
                    negative_artists.add(banned_artist)
    return negative_artists