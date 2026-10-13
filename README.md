# 对话式音乐推荐：融合对比学习与交叉注意力的两阶段多模态架构

基于 TalkPlay RecSys Challenge 2026 数据的两阶段对话式音乐推荐系统，面向「AI / 数据产品 / 推荐算法」方向。

## 问题定义

对话式音乐推荐需要理解动态上下文语义，并在稀疏的长尾曲目 catalog 中捕捉用户瞬时偏好。传统序列模型（GRU4Rec、SASRec）在长尾冷启动与会话稀疏场景下表现不佳，跨模态（文本语义、用户协同偏好、音频特征）难以对齐。

## 数据来源

HuggingFace TalkPlay RecSys Challenge 2026，共 5 个数据集：

| 数据集 | 内容 | 规模 |
| --- | --- | --- |
| Challenge-Dataset | 多轮对话 session，含 GPA 标签 | 15,199 train + 1,000 test |
| Track-Embeddings | LAION-CLAP 音频嵌入 | 47,071 × 512 |
| Track-Metadata | 曲目 → 艺术家映射 | 47,071 条 |
| User-Embeddings | CF-BPR 协同过滤用户嵌入 | 8,720 × 128 |
| User-Metadata | 用户画像 | 8,772 用户 |

## 方法

- **Stage 1 召回**：MiniLM 对比学习微调 → context 编码 → 与 46K thought 向量余弦检索，绕过 Item ID 冷启动，返回 Top-200。
- **Stage 2 精排**：Cross-Attention 重排网络，融合上下文语义、用户 CF 偏好、显式负反馈与候选音频特征，使用 LambdaRank 排序损失端到端优化。
- **基线**：LR、GRU4Rec、SASRec、BERT4Rec。

## 结果（全量 Test 6,184 条）

| 模型 | MRR | NDCG@10 | Hit@1 | Hit@10 |
| --- | :--: | :--: | :--: | :--: |
| GRU4Rec | 0.0140 | 0.0170 | 0.41% | 3.37% |
| SASRec | 0.0190 | 0.0200 | 0.42% | 4.59% |
| BERT4Rec | 0.0100 | 0.0110 | 0.30% | 2.20% |
| LR 基线 | 0.0610 | 0.0662 | 1.78% | 13.66% |
| Stage 1 | 0.1360 | 0.1641 | 6.14% | 29.17% |
| **Stage 1+2** | **0.2745** | **0.2940** | **20.10%** | **41.30%** |

两阶段架构相对 Stage 1 的 MRR 提升约 102%。

## 运行

```bash
pip install -r requirements.txt
python preprocess_thoughts.py            # 数据预处理（所有模型共用）

# 两阶段
python stage1_thought_retrieval.py       # Stage 1 语义检索
python finetune_stage1.py                # 可选：对比学习微调
python train_stage2.py                   # Stage 2 精排训练
python pipeline.py                       # 两阶段联合评估

# 序列基线
python sasrec.py
python gru4rec.py
python bert4rec_train.py

# 线性基线
python baseline.py
```

## 项目结构

```
├── config/default_config.py
├── data/
├── models/
├── utils/
├── baseline.py
├── stage1_thought_retrieval.py
├── finetune_stage1.py
├── train_stage2.py
├── pipeline.py
├── sasrec.py
├── gru4rec.py
├── bert4rec_train.py
├── preprocess_thoughts.py
├── reports/paper.pdf
└── requirements.txt
```

## 局限性

- 仅在单一挑战赛数据集上验证。
- Stage 1 固定 Top-200 召回，未做召回数量的敏感性实验。
- 使用数据集自带的 thought 文本，未接入真实 LLM 在线生成。
