# MARS: Manifold Alignment for Language Model Security

## 项目概述
- 本项目是论文“Manifold Alignment for Language Model Security: A Machine Learning Approach to Jailbreak Defense”的代码。
- 目标：实现意图抽取（LoRA 微调）+ 语义对齐（Sinkhorn 最优传输）+ 下游安全分类器的完整训练与评估管线。
- 特点：模块化 PyTorch 代码、混合精度（AMP）与梯度检查点、省显存训练、t-SNE 可视化、输出安全指标（TPR/FPR/ASR）。

## 目录结构
- `data_loader.py`：数据预处理与对抗式变换，构造 `(x, y)` 对，其中 `x=T(y)`。
- `models/intent_extractor.py`：LoRA 微调的意图抽取器，训练 CLM 损失 `-log P(y|x)`，推理生成简洁意图 `y′`。
- `losses/sinkhorn_alignment.py`：使用 `geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1)` 计算对齐损失 `Lalign`。
- `models/classifier.py`：轻量安全分类器（MLP），对 `y′` 的句向量进行二分类。
- `train.py`：训练循环，组合总损失 `Ltotal = Lalign + λ·LLM`，`λ=0.5`，`AdamW`+Warmup+Cosine 调度，AMP 混合精度。
- `eval.py`：评估与 t-SNE 可视化，训练并推断安全分类器，输出 `TPR/FPR/ASR` 与 `tsne.png`。

## 环境与依赖
- Python ≥ 3.10；可选 CUDA 环境。
- 安装：
  - `pip install torch transformers peft sentence-transformers geomloss scikit-learn matplotlib`
  - 可选：`pip install wandb`

## 数据准备
- 数据集：
  - Benign intents：Natural Questions（约 20K）
  - Malicious intents：JailBreakV-28K（约 20K）
- 文件格式：JSONL，每行一个对象，文本键默认 `text`。示例：
  - `{"text": "What is the capital of France?"}`
- 在脚本中通过 `--benign_path --malicious_path --text_key` 指定路径与字段名。

## 快速开始
- 训练（示例）：
  - `python3 train.py --model_name meta-llama/Llama-3.1-8B-Instruct --benign_path data/nq_20k.jsonl --malicious_path data/jb_20k.jsonl --text_key text --epochs 1 --batch_size 16 --lr 5e-5 --lambda_align 0.5 --accum_steps 4`
  - 训练完成后，LoRA 权重与 tokenizer 保存在 `checkpoints/intent_extractor`。
- 评估与可视化：
  - `python3 eval.py --model_name meta-llama/Llama-3.1-8B-Instruct --benign_path data/nq_20k.jsonl --malicious_path data/jb_20k.jsonl --text_key text`
  - 输出 `TPR/FPR/ASR` 指标，并在根目录生成 `tsne.png`。

## 方法与实现
- 意图抽取器（`models/intent_extractor.py`）：
  - 基座模型：`AutoModelForCausalLM` + `AutoTokenizer`。
  - LoRA 配置：`r=16, lora_alpha=32, target_modules=[q_proj, v_proj, k_proj, o_proj], lora_dropout=0.1`。
  - 训练：拼接输入 `x` 与目标 `y` 的 token 序列，仅对 `y` 部分打标签，其余置 `-100` 计算 CLM 损失。
  - 推理：`generate(**x_inputs)` 生成简洁意图 `y′`。
- 语义对齐（`losses/sinkhorn_alignment.py`）：
  - 句向量编码器：`SentenceTransformer("all-MiniLM-L6-v2")`，返回归一化嵌入 `ϕ(·)`。
  - Sinkhorn 损失：`SamplesLoss(loss="sinkhorn", p=2, blur=0.1)`，计算 `Lalign(δ_{ϕ(y′)}, ν_c)`。
  - 经验分布：在训练脚本中对类分布进行嵌入累积与对齐（当前示例以全集近似）。
- 安全分类器（`models/classifier.py`）：
  - 结构：两层 MLP（`Linear → GELU → Dropout → Linear`），输入维度随句向量模型变化（默认 384）。
  - 训练：在评估脚本中对 `y′` 的嵌入与标签进行快速拟合，然后推断安全/恶意决策。

## 训练细节（`train.py`）
- 总损失：`Ltotal = Lalign + λ·LLM`，默认 `λ=0.5`。
- 优化：`AdamW(lr=5e-5)`；总步数 10% Warmup；Cosine decay。
- 批大小：`16`；梯度累积：`4`，等效批 `64`。
- AMP：`torch.cuda.amp` 与 `GradScaler`；启用 gradient checkpointing 降低显存占用。

## 评估与可视化（`eval.py`）
- t-SNE：对 `x, y′, y` 的嵌入进行降维与散点可视化，超参数示例：`perplexity=30, n_iter=1000`。
- 指标：输出 `TPR, FPR, ASR`；可替换数据为 AdvBench/HarmBench 进行基准对比。
- 分类器：默认使用 MLP 轻量分类器；如环境允许可替换为 Llama-Guard3-1B。

## 扩展与替代
- 基座模型：可替换为 `Qwen3-8B`、`mistral-7b-instruct` 等更小模型以快速验证。
- 安全分类器：可替换为 TextCNN 或集成 Llama-Guard；训练时冻结意图抽取器。
- 实验追踪：可集成 W&B 记录训练与评估过程（loss 曲线、指标、可视化）。

## 注意事项
- 模型与编码器首次加载需要网络访问与合适的权限（HF Hub）；`sentence-transformers` 与 `geomloss` 下载可能较慢。
- 显存不足时建议：减小 `batch_size`、增大 `accum_steps`、缩短 `max_length`，并保持 AMP 开启。
- 生产使用需增加审计与过滤策略，避免模型输出敏感或有害内容。

- 生态：Transformers, PEFT, Sentence-Transformers, GeomLoss, scikit-learn, PyTorch
