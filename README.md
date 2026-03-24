# Mini-LLM

A **50M-parameter language model built entirely from scratch** — implementing the same architecture as Llama 3, Qwen, and Mistral — trained end-to-end on a MacBook M1.

No HuggingFace Trainer. No wrappers. Every layer of the stack written by hand.

<p align="center">
  <img src="results/loss_curves.png" alt="Training loss curves" width="600"/>
</p>

---

## Highlights

- 🏗️ **Modern architecture** — RoPE, RMSNorm, SwiGLU, Grouped Query Attention, KV Cache
- 🔁 **Full training pipeline** — Pre-training → Supervised Fine-Tuning → DPO Alignment
- 🧪 **Synthetic data pipeline** — Wikipedia-grounded instruction pairs generated locally via Ollama
- 📊 **Rigorous evaluation** — Perplexity tracking, sample generations at each stage, LLM-as-judge scoring
- 💻 **Runs on commodity hardware** — Everything fits on 16GB Apple Silicon, $0 cloud budget

---

## Why This Project Exists

Most "build an LLM" tutorials either wrap HuggingFace (you learn nothing about the architecture) or clone GPT-2 (a 2019 design). This project implements **2024–2025 techniques** — the same components found in Llama 3, Mistral, and Qwen — and trains the full stack: pre-training, supervised fine-tuning, and preference alignment.

---

## Architecture

**Llama-style decoder-only transformer (~50M parameters)**

The [TinyStories paper](https://arxiv.org/abs/2305.07759) showed that models under 10M parameters can generate coherent English on curated data. At 50M, the model learns non-trivial patterns while fitting comfortably in M1 memory and training in hours.

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 10,000 | Top-10k tokens from GPT-Neo tokenizer |
| `hidden_dim` | 512 | Embedding dimension |
| `n_layers` | 12 | Transformer blocks |
| `n_heads` | 8 | Query attention heads |
| `n_kv_heads` | 2 | Key/Value heads (GQA ratio 4:1) |
| `ffn_hidden` | 1,536 | SwiGLU intermediate (~3× hidden) |
| `max_seq_len` | 512 | Context length |
| `dropout` | 0.1 | Regularization during training |
| `rope_theta` | 10,000 | RoPE base frequency |

### Components

Every component is implemented from scratch using [Apple MLX](https://github.com/ml-explore/mlx):

| Component | What It Replaces | Why It's Used |
|-----------|-----------------|---------------|
| **RoPE** (Rotary Position Embeddings) | Learned absolute position embeddings | Generalizes to unseen sequence lengths; used in Llama, Qwen, Mistral |
| **RMSNorm** | LayerNorm | Simpler, faster (no mean subtraction), empirically equivalent |
| **SwiGLU** | GELU/ReLU in FFN | Consistently outperforms alternatives on language tasks; uses 3 weight matrices |
| **Grouped Query Attention** | Multi-Head Attention | Reduces KV cache size at inference; 8 Q heads share 2 KV heads |
| **KV Cache** | Full recomputation | Reduces generation from O(N²) to O(N) compute |

### Why MLX?

Apple's [MLX](https://github.com/ml-explore/mlx) is designed for Apple Silicon's unified memory — zero-copy between CPU and GPU. PyTorch on Mac uses the MPS backend which copies tensors between memory pools. MLX's API mirrors NumPy/PyTorch, so the code stays familiar.

---

## Training Pipeline

### Phase 1: Pre-training

**Objective:** Causal language modeling (next token prediction)

**Data:**
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — ~470M tokens of synthetic short stories
- Wikipedia subset — ~30–50M tokens for factual grounding

**Hyperparameters:**
- Optimizer: AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)
- Learning rate: 3e-4 peak, cosine decay, 1000-step warmup
- Batch size: 32 × 512 tokens (~16k tokens/step)
- Steps: ~30k (~1 epoch over TinyStories)
- Time: ~4–6 hours on M1

**Expected result:** Loss drops from ~9.2 (random, ln(10000)) to ~2.5–3.0. Model generates grammatical English.

### Phase 2: Supervised Fine-Tuning (SFT)

Turns the text completion model into an instruction-following model.

**Data (~10k–20k samples from 3 sources):**

| Source | Samples | Description |
|--------|---------|-------------|
| Wiki-Synth (custom pipeline) | 2k–5k | Wikipedia-grounded instruction pairs generated locally via Ollama |
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 5k | Instruction-response pairs (subsampled from 52k) |
| [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 5k | Human-written instruction pairs (subsampled from 15k) |

**Chat template:**
```
<|system|>You are a helpful assistant.<|end|>
<|user|>{instruction}<|end|>
<|assistant|>{response}<|end|>
```

Loss is computed on assistant tokens only.

**Hyperparameters:**
- Learning rate: 2e-5
- Epochs: 3
- Time: ~1–2 hours on M1

### Phase 3: DPO Alignment

**Why DPO over RLHF?** [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) achieves the same goal as RLHF without training a separate reward model. One training loop, much more stable. Used by most modern open-source alignment efforts.

**Data (2k–5k preference pairs):**
- Generate multiple responses per prompt from the SFT model
- Use a local Ollama judge (qwen2.5:7b) to rank them
- Best = chosen, worst = rejected

**Hyperparameters:**
- Learning rate: 5e-7
- β (temperature): 0.1
- Epochs: 1
- Time: ~30 minutes on M1

---

## Evaluation

| Metric | Method | Measured At |
|--------|--------|-------------|
| **Perplexity** | Held-out test set | After each phase |
| **Generation quality** | Side-by-side samples | Pre-trained → SFT → DPO |
| **LLM-as-judge** | Ollama scores (coherence, instruction following, factual grounding, 1–5 scale) | Final model |

---

## Project Structure

```
mini-llm/
├── README.md
├── config.py                  # All hyperparameters
├── model/
│   ├── architecture.py        # Transformer implementation
│   ├── rope.py                # Rotary position embeddings
│   ├── rmsnorm.py             # RMS normalization
│   ├── attention.py           # Grouped query attention + KV cache
│   ├── ffn.py                 # SwiGLU feedforward
│   └── generate.py            # Inference with KV cache
├── data/
│   ├── prepare_pretrain.py    # Download + tokenize TinyStories
│   ├── prepare_sft.py         # Merge synthetic + Alpaca + Dolly
│   └── prepare_dpo.py         # Generate preference pairs
├── train/
│   ├── pretrain.py            # Pre-training loop
│   ├── sft.py                 # Supervised fine-tuning loop
│   └── dpo.py                 # DPO alignment loop
├── eval/
│   ├── perplexity.py          # Perplexity measurement
│   ├── generate_samples.py    # Sample generations at each stage
│   └── judge.py               # LLM-as-judge evaluation
├── wiki-synth/                # Synthetic data generation pipeline
│   ├── step1_download.py      # Wikipedia article download
│   ├── step2_chunk.py         # Text chunking
│   ├── step3_generate.py      # Instruction pair generation (Ollama)
│   ├── step4_judge.py         # Quality filtering
│   ├── step5_dedupe.py        # Deduplication
│   └── step6_export.py        # Export to training format
└── results/
    ├── loss_curves.png
    ├── sample_generations.md
    └── eval_scores.json
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/mini-llm.git
cd mini-llm

# Install dependencies
pip install mlx datasets transformers

# Pre-train
python train/pretrain.py

# Supervised fine-tuning
python train/sft.py

# DPO alignment
python train/dpo.py

# Generate text
python model/generate.py --prompt "Once upon a time"
```

---

## References

| Paper | Topic |
|-------|-------|
| [TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759) | Small-scale language model training on synthetic stories |
| [RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) | Rotary position embeddings |
| [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) | GQA, SwiGLU, RMSNorm architecture |
| [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization |
| [Source2Synth (Gupta et al., 2024)](https://arxiv.org/abs/2409.08239) | Synthetic data pipeline design |

---

## License

MIT
