"""LoRA Fine-tuning for WigtnOCR using ms-swift.

Wraps ms-swift CLI to fine-tune Qwen3-VL-2B-Instruct with LoRA
on document parsing data (PDF page images → structured Markdown).

Defaults reproduce the v1 run (2026-03-16, ms-swift 4.0.1) whose
checkpoint-600 was merged and released as Wigtn/Qwen3-VL-2B-WigtnOCR.

Usage:
    # Train with the v1 config
    python -m training.lora_trainer --config configs/lora_qwen3vl2b_v1.yaml

    # Print the swift command without running it
    python -m training.lora_trainer --config configs/lora_qwen3vl2b_v1.yaml --dry-run

    # Resume from checkpoint
    python -m training.lora_trainer --resume output/wigtnocr-2b-lora/vx-xxx/checkpoint-xxx
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "output_dir": "output/wigtnocr-2b-lora",
    "train_dataset": "datasets/training/train.jsonl",
    "val_dataset": "datasets/training/val.jsonl",
    # LoRA
    "tuner_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    # Freeze vision encoder
    "freeze_vit": True,
    "freeze_aligner": True,
    # Training
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "max_length": 4096,
    "seed": 42,
    "data_seed": 42,
    # Performance
    "torch_dtype": "bfloat16",
    "attn_impl": "sdpa",
    "gradient_checkpointing": True,
    "packing": False,
    "padding_free": False,
    "deepspeed": "zero2",
    "nproc_per_node": 2,
    # Eval & Save
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 3,
    "logging_steps": 5,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 4,
}

DEFAULT_ENV = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "IMAGE_MAX_TOKEN_NUM": "1024",
}


def load_config(config_path: Path) -> tuple[dict, dict]:
    """Load training config from YAML, merge with defaults. Returns (swift_args, env)."""
    config = DEFAULT_CONFIG.copy()
    env = DEFAULT_ENV.copy()
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            # Flatten nested YAML structure
            if "model" in yaml_config and isinstance(yaml_config["model"], dict):
                m = yaml_config["model"]
                config["model"] = m.get("base", config["model"])
                config["output_dir"] = m.get("output", config["output_dir"])
                for key in ["revision", "use_hf"]:
                    if key in m:
                        config["model_revision" if key == "revision" else key] = m[key]
            if "lora" in yaml_config:
                lora = yaml_config["lora"]
                config["lora_rank"] = lora.get("r", config["lora_rank"])
                config["lora_alpha"] = lora.get("alpha", config["lora_alpha"])
                config["lora_dropout"] = lora.get("dropout", config["lora_dropout"])
                for key in ["target_modules", "freeze_vit", "freeze_aligner"]:
                    if key in lora:
                        config[key] = lora[key]
            if "training" in yaml_config:
                t = yaml_config["training"]
                mapped = {
                    "num_epochs": "num_train_epochs",
                    "batch_size": "per_device_train_batch_size",
                    "max_seq_length": "max_length",
                }
                for key in ["learning_rate", "num_epochs", "batch_size",
                            "gradient_accumulation_steps", "warmup_ratio",
                            "lr_scheduler_type", "weight_decay", "max_seq_length",
                            "seed", "data_seed", "eval_steps", "save_steps",
                            "save_total_limit", "logging_steps"]:
                    if key in t:
                        config[mapped.get(key, key)] = t[key]
            if "data" in yaml_config:
                d = yaml_config["data"]
                if "train" in d:
                    config["train_dataset"] = d["train"]
                if "val" in d:
                    config["val_dataset"] = d["val"]
                if "train_dir" in d:
                    config["train_dataset"] = str(Path(d["train_dir"]) / "training" / "train.jsonl")
                    config["val_dataset"] = str(Path(d["train_dir"]) / "training" / "val.jsonl")
            if "hardware" in yaml_config:
                h = yaml_config["hardware"]
                for key in ["nproc_per_node", "deepspeed"]:
                    if key in h:
                        config[key] = h[key]
                if h.get("bf16") is False:
                    config["torch_dtype"] = "float16"
                if "cuda_visible_devices" in h:
                    env["CUDA_VISIBLE_DEVICES"] = str(h["cuda_visible_devices"])
                if "image_max_token_num" in h:
                    env["IMAGE_MAX_TOKEN_NUM"] = str(h["image_max_token_num"])
    env["NPROC_PER_NODE"] = str(config.get("nproc_per_node", 2))
    return config, env


def build_swift_command(config: dict, resume_from: str = None) -> list[str]:
    """Build ms-swift CLI command from config dict."""
    cmd = ["swift", "sft"]

    # Map config keys to swift CLI args
    arg_map = {
        "model": "--model",
        "model_revision": "--model_revision",
        "output_dir": "--output_dir",
        "train_dataset": "--dataset",
        "val_dataset": "--val_dataset",
        "tuner_type": "--tuner_type",
        "lora_rank": "--lora_rank",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "target_modules": "--target_modules",
        "torch_dtype": "--torch_dtype",
        "attn_impl": "--attn_impl",
        "num_train_epochs": "--num_train_epochs",
        "learning_rate": "--learning_rate",
        "lr_scheduler_type": "--lr_scheduler_type",
        "weight_decay": "--weight_decay",
        "per_device_train_batch_size": "--per_device_train_batch_size",
        "per_device_eval_batch_size": "--per_device_eval_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "warmup_ratio": "--warmup_ratio",
        "max_length": "--max_length",
        "seed": "--seed",
        "data_seed": "--data_seed",
        "eval_steps": "--eval_steps",
        "save_steps": "--save_steps",
        "save_total_limit": "--save_total_limit",
        "logging_steps": "--logging_steps",
        "deepspeed": "--deepspeed",
        "dataset_num_proc": "--dataset_num_proc",
        "dataloader_num_workers": "--dataloader_num_workers",
    }

    bool_args = {
        "use_hf": "--use_hf",
        "freeze_vit": "--freeze_vit",
        "freeze_aligner": "--freeze_aligner",
        "gradient_checkpointing": "--gradient_checkpointing",
        "packing": "--packing",
        "padding_free": "--padding_free",
    }

    for key, flag in arg_map.items():
        if key in config:
            val = config[key]
            if isinstance(val, list):
                cmd.extend([flag, " ".join(val)])
            else:
                cmd.extend([flag, str(val)])

    for key, flag in bool_args.items():
        if key in config:
            cmd.extend([flag, "true" if config[key] else "false"])

    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])

    return cmd


def train(config_path: Path = None, resume_from: str = None, dry_run: bool = False):
    """Run LoRA fine-tuning via ms-swift.

    Args:
        config_path: Path to the training YAML config
        resume_from: Checkpoint path to resume from
        dry_run: Print command without executing
    """
    config, env_vars = load_config(config_path)

    # Validate data files exist
    train_path = Path(config["train_dataset"])
    if not train_path.exists() and not dry_run:
        print(f"Training data not found: {train_path}")
        print("Run data preparation first:")
        print("  python scripts/training/prepare_training_data.py --dataset all")
        sys.exit(1)

    cmd = build_swift_command(config, resume_from)

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    cmd_str = " ".join(cmd)
    full_cmd = f"{env_str} {cmd_str}"

    print(f"\n{'=' * 60}")
    print("WigtnOCR LoRA Training")
    print(f"  Model: {config['model']}")
    print(f"  Train data: {config['train_dataset']}")
    print(f"  Output: {config['output_dir']}")
    print(f"  LoRA rank: {config['lora_rank']}, alpha: {config['lora_alpha']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch: {config['per_device_train_batch_size']} x {config['gradient_accumulation_steps']} accum")
    print(f"  GPUs: {env_vars['CUDA_VISIBLE_DEVICES']} ({env_vars['NPROC_PER_NODE']} procs)")
    print(f"  Seed: {config['seed']}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print(f"[DRY RUN] Command:\n{full_cmd}")
        return

    # Execute
    env = os.environ.copy()
    env.update(env_vars)

    print(f"Executing: {cmd_str}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining complete! Checkpoints saved to: {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WigtnOCR LoRA Fine-tuning (ms-swift)")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_qwen3vl2b_v1.yaml"),
                        help="Training config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume, dry_run=args.dry_run)
