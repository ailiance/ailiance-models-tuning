"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""

    # Model
    base_model: str = "Qwen/Qwen2.5-32B-Instruct"
    model_revision: str = "main"

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    packing: bool = True

    # Output
    output_dir: str = "outputs"
    hub_model_id: str | None = None
    push_to_hub: bool = False

    # Data
    dataset_path: str = "datasets/processed/train.jsonl"
    eval_dataset_path: str | None = None


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    model_path: str = "outputs/checkpoint-latest"
    eval_dataset: str = "datasets/processed/eval.jsonl"
    num_samples: int = 100
    providers: list[str] = field(default_factory=lambda: ["local"])
    output_file: str = "artifacts/eval_results.json"
