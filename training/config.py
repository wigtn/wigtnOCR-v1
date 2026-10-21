"""Training configuration for ms-swift LoRA fine-tuning."""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA hyperparameters for ms-swift."""
    rank: int = 8
    alpha: int = 32
    target_modules: str = "all-linear"
    freeze_vit: bool = True
    freeze_aligner: bool = True


@dataclass
class TrainingConfig:
    """Full training configuration matching ms-swift CLI options."""
    # Model
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "output/wigtnocr-2b-lora"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 4096

    # Data
    train_dataset: str = "datasets/training/train.jsonl"
    val_dataset: str = "datasets/training/val.jsonl"

    # Hardware
    torch_dtype: str = "bfloat16"
    deepspeed: str = "zero2"
    nproc_per_node: int = 2

    # Data prep
    min_score: int = 3
    max_doc_ratio: float = 0.25
    val_ratio: float = 0.1
    image_dpi: int = 200
