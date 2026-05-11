"""Tests for training configuration."""

from src.ailiance_tuning.config import TrainingConfig


def test_default_config():
    """Default config has sensible values."""
    cfg = TrainingConfig()
    assert cfg.base_model == "Qwen/Qwen3-8B"
    assert cfg.lora_r == 16
    assert cfg.load_in_4bit is True
    assert cfg.max_seq_length == 2048
    assert cfg.num_train_epochs == 3


def test_config_override():
    """Config fields can be overridden."""
    cfg = TrainingConfig(
        base_model="meta-llama/Llama-3-8B",
        lora_r=32,
        learning_rate=1e-5,
        num_train_epochs=1,
    )
    assert cfg.base_model == "meta-llama/Llama-3-8B"
    assert cfg.lora_r == 32
    assert cfg.learning_rate == 1e-5
    assert cfg.num_train_epochs == 1
    # Unchanged defaults
    assert cfg.lora_alpha == 32
    assert cfg.packing is True


def test_config_lora_target_modules_default():
    """Default LoRA target modules cover all attention + MLP projections."""
    cfg = TrainingConfig()
    assert "q_proj" in cfg.lora_target_modules
    assert "v_proj" in cfg.lora_target_modules
    assert "gate_proj" in cfg.lora_target_modules
    assert len(cfg.lora_target_modules) == 7


def test_config_hub_push_disabled_by_default():
    """Push to hub is disabled by default."""
    cfg = TrainingConfig()
    assert cfg.push_to_hub is False
    assert cfg.hub_model_id is None


def test_config_quantization_defaults():
    """4-bit quantization uses NF4 with bfloat16 compute."""
    cfg = TrainingConfig()
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype == "bfloat16"
