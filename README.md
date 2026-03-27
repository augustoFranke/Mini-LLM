# Mini-LLM

A 50M-parameter language model built from scratch using Apple MLX, trained end-to-end on a MacBook M1.

The goal is to implement the same architecture used in modern models like Llama 3, Qwen, and Mistral — not by wrapping libraries, but by writing every component by hand. No HuggingFace Trainer. No shortcuts.

---

## What I'm building

A full training pipeline in three phases:

1. **Pre-training** — next token prediction on TinyStories + a Wikipedia subset
2. **Supervised Fine-Tuning (SFT)** — turning the base model into an instruction follower
3. **DPO Alignment** — preference optimization without a reward model

---

## What I'm learning

- How transformers actually work under the hood — not conceptually, but in code
- Modern architecture components: RoPE, RMSNorm, SwiGLU, Grouped Query Attention, KV Cache
- How pre-training, SFT, and alignment fit together and depend on each other
- How to build a synthetic data pipeline (Wikipedia → instruction pairs via Ollama)
- Training dynamics: loss curves, learning rate schedules, gradient behavior
- How to run real ML experiments on consumer hardware with zero cloud budget

---

## Architecture

Llama-style decoder-only transformer, ~50M parameters.

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 10,000 |
| `hidden_dim` | 512 |
| `n_layers` | 12 |
| `n_heads` | 8 |
| `n_kv_heads` | 2 (GQA 4:1) |
| `ffn_hidden` | 1,536 |
| `max_seq_len` | 512 |

Built on [Apple MLX](https://github.com/ml-explore/mlx) — designed for Apple Silicon's unified memory, zero-copy between CPU and GPU.

---

## Project Structure

```
mini-llm/
├── config.py
├── model/
│   ├── architecture.py
│   ├── rope.py
│   ├── rmsnorm.py
│   ├── attention.py
│   ├── ffn.py
│   └── generate.py
├── data/
│   ├── prepare_pretrain.py
│   ├── prepare_sft.py
│   └── prepare_dpo.py
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   └── dpo.py
├── eval/
│   ├── perplexity.py
│   ├── generate_samples.py
│   └── judge.py
└── wiki-synth/
    ├── step1_download.py
    ├── step2_chunk.py
    ├── step3_generate.py
    ├── step4_judge.py
    ├── step5_dedupe.py
    └── step6_export.py
```

---

## Quick Start

```bash
pip install mlx datasets transformers

python train/pretrain.py
python train/sft.py
python train/dpo.py

python model/generate.py --prompt "Once upon a time"
```