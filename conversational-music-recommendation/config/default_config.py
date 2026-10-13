import torch

# ── 网络代理 ──
PROXY_ADDRESS = ""
PROXY_ENV = {
    "HTTP_PROXY": PROXY_ADDRESS,
    "HTTPS_PROXY": PROXY_ADDRESS,
    "http_proxy": PROXY_ADDRESS,
    "https_proxy": PROXY_ADDRESS,
}

# ── 嵌入维度 ──
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# 微调后设为 'models/minilm_finetuned' 使用对比学习微调版
STAGE1_MODEL_PATH = 'models/minilm_finetuned'  # None = 用原始预训练模型；设置路径 = 用微调版
TEXT_DIM = 384
AUDIO_DIM = 512
USER_EMB_DIM = 128

# ── 训练超参数 ──
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ── 数据 & 候选池 ──
MAX_SESSION_SAMPLES = None  # 使用全部 15199 个 session
CANDIDATE_POOL_SIZE = 1000
EVAL_SAMPLE_SIZE = 20

# ── Stage 2: Cross-Attention Re-Ranker ──
STAGE2_D_MODEL = 256
STAGE2_NUM_HEADS = 4
STAGE2_NUM_LAYERS = 1            # 减到 1 层防过拟合
STAGE2_DROPOUT = 0.3             # 加大 dropout 防过拟合
STAGE2_MLP_HIDDEN = 128
STAGE2_K_RETRIEVAL = 200         # Stage 1 召回的 Top-K 候选数（训练/评估一致）
STAGE2_SIGMA = 1.0               # LambdaRank sigmoid 缩放因子
STAGE2_EPOCHS = 5                # Stage 2 训练轮数
STAGE2_LR = 1e-3
STAGE2_BATCH_SIZE = 4            # K=200 时降低 batch size

# ── 设备 ──
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
