from dataclasses import dataclass, field
from typing import List




























@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Purpose: encapsulates all PEFT LoRA hyperparameters.
    Inputs:  none (all have sensible defaults)
    Outputs: dataclass used by SFTTrainingJob._apply_lora
    Complexity: O(1)
    Side effects: none
    """
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization configuration.

    Purpose: encapsulates NF4 double-quant settings for bitsandbytes.
    Inputs:  none (all have sensible defaults)
    Outputs: dataclass consumed by SFTTrainingJob._load_model_and_tokenizer
    Complexity: O(1)
    Side effects: none
    """
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class SFTTrainingConfig:
    """Unified SFT training configuration.

    Purpose: holds LoRA, quantization, and HuggingFace TrainingArguments
             parameters in a single validated dataclass.
    Inputs:  none (all fields have sensible defaults)
    Outputs: used by SFTTrainingJob and MockSFTJob
    Complexity: O(1)
    Side effects: none
    """
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optimizer: str = "paged_adamw_8bit"
    early_stopping_patience: int = 3
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True

    def to_training_arguments_kwargs(self) -> dict:
        """Convert to transformers TrainingArguments kwargs.

        Purpose: produce a dict of kwargs that can be passed directly to
                 transformers.TrainingArguments(**kwargs, output_dir=...).
        Inputs:  self
        Outputs: dict with all required TrainingArguments keys
        Complexity: O(1)
        Side effects: none
        """
        return {
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optim": self.optimizer,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "logging_steps": self.logging_steps,
            "evaluation_strategy": "steps",
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "load_best_model_at_end": True,
        }
