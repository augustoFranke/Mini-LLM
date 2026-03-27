from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10_000       # Top-10k tokens from GPT-Neo tokenizer
    hidden_dim: int = 512          # Embedding dimension
    n_layers: int = 12             # Transformer blocks
    n_heads: int = 8               # Query attention heads
    n_kv_heads: int = 2            # Key/Value heads (GQA ratio 4:1)
    ffn_hidden: int = 1_536        # SwiGLU intermediate (~3× hidden_dim)
    max_seq_len: int = 512         # Context length
    dropout: float = 0.1           # Regularization during training
    rope_theta: float = 10_000.0   # RoPE base frequency
    rms_norm_eps: float = 1e-6     # RMSNorm epsilon


@dataclass
class PreTrainConfig:
    # Optimizer
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    # Schedule
    warmup_steps: int = 1_000
    max_steps: int = 30_000
    # Batching
    batch_size: int = 32           # sequences per batch
    # ~16k tokens per step (batch_size × max_seq_len)
    # Checkpointing
    save_every: int = 1_000        # steps between checkpoints
    eval_every: int = 500


@dataclass
class SFTConfig:
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    save_every: int = 500
    eval_every: int = 250


@dataclass
class DPOConfig:
    learning_rate: float = 5e-7
    beta: float = 0.1              # KL penalty coefficient
    epochs: int = 1
    batch_size: int = 8
    save_every: int = 200


@dataclass
class DataConfig:
    tinystories_path: str = "data/tinystories"
    wiki_path: str = "data/wiki"
    sft_path: str = "data/sft"
    dpo_path: str = "data/dpo"
    tokenizer_name: str = "EleutherAI/gpt-neo-125M"


# Convenience instances
model_config = ModelConfig()
pretrain_config = PreTrainConfig()
sft_config = SFTConfig()
dpo_config = DPOConfig()
data_config = DataConfig()