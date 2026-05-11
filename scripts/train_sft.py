#!/usr/bin/env python3
"""SFT Training script using trl.SFTTrainer with QLoRA.

Target: KXKM-AI RTX 4090 (24GB VRAM)
Usage: python scripts/train_sft.py --config configs/sft_default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ailiance_tuning.train_sft")


def main():
    parser = argparse.ArgumentParser(description="SFT Training with QLoRA")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset", default="datasets/processed/train.jsonl", help="Training dataset path")
    parser.add_argument("--output-dir", default="outputs/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    args = parser.parse_args()

    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output_dir}")

    # Lazy imports (heavy)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch transformers peft trl datasets bitsandbytes accelerate")
        return

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        logger.warning("No GPU available — training will be very slow")

    # Quantization config (4-bit NF4) with CPU offload for large models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Load model with max_memory to prevent OOM during quantization
    logger.info(f"Loading model: {args.base_model}")
    max_memory = {0: "22GiB", "cpu": "50GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    logger.info(f"Dataset: {len(dataset)} examples")

    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        trainer.push_to_hub()
        logger.info(f"Pushed to HuggingFace Hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
