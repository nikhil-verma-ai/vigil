"""SFT (Supervised Fine-Tuning) training pipeline with QLoRA.

This module implements:
  - SFTTrainingJob: full production training using HuggingFace Transformers +
    PEFT LoRA + BitsAndBytes 4-bit quantization + trl SFTTrainer
  - MockSFTJob: lightweight simulator for testing without GPU/model download

Invariants:
  - adapter_path always points to a directory containing adapter_config.json
  - SFTResult.final_train_loss and final_eval_loss are always positive floats
  - progress_callback, when provided, is called exactly once per logging_steps
    interval during training
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from services.trainer.qlora_config import SFTTrainingConfig


@dataclass
class SFTResult:
    """Result returned by a completed SFT training run.

    Fields:
      adapter_path       — path to directory containing the saved LoRA adapter
      final_train_loss   — training loss at the last logged step
      final_eval_loss    — eval loss at the last evaluation step
      training_steps     — total number of optimizer steps completed
      epochs_completed   — number of full epochs run
      early_stopped      — True if training halted via early stopping
      duration_seconds   — wall-clock time for the training run
    """
    adapter_path: str
    final_train_loss: float
    final_eval_loss: float
    training_steps: int
    epochs_completed: int
    early_stopped: bool
    duration_seconds: float


class SFTTrainingJob:
    """Full SFT training job using QLoRA (BitsAndBytes + PEFT + trl).

    Purpose: orchestrate model loading, LoRA application, and SFTTrainer
             execution for a single fine-tuning run.
    Inputs:
      config            — SFTTrainingConfig with all hyperparameters
      output_dir        — directory where the adapter and checkpoints are written
      progress_callback — optional callable(step: int, loss: float) called at
                          each logging_steps interval; useful for live UIs
    """

    def __init__(
        self,
        config: SFTTrainingConfig,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback

    def run(self, model_name_or_path: str, train_dataset, eval_dataset) -> SFTResult:
        """Execute the full SFT training run.

        Purpose: load base model with 4-bit quantization, apply LoRA, run
                 trl.SFTTrainer, save adapter, return result.
        Inputs:
          model_name_or_path — HuggingFace model ID or local path; if it
                               starts with "mock://" a MockSFTJob is used
          train_dataset      — HuggingFace Dataset for training
          eval_dataset       — HuggingFace Dataset for evaluation
        Outputs: SFTResult with adapter location and training metrics
        Complexity: O(steps * model_size) for real; O(steps) for mock
        Side effects: writes files to self.output_dir; may download model weights
        """
        # Dispatch to MockSFTJob when running in test/mock mode so that tests
        # never require a GPU or network access to model repositories.
        if model_name_or_path.startswith("mock://"):
            mock = MockSFTJob(
                config=self.config,
                output_dir=self.output_dir,
                progress_callback=self.progress_callback,
            )
            return mock.run(model_name_or_path, train_dataset, eval_dataset)

        # --- Real training path (requires GPU + installed heavy deps) ---
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer

        start_time = time.time()
        adapter_output_dir = os.path.join(self.output_dir, "sft_adapter")
        os.makedirs(adapter_output_dir, exist_ok=True)

        model, tokenizer = self._load_model_and_tokenizer(model_name_or_path)
        model = self._apply_lora(model)

        training_args_kwargs = self.config.to_training_arguments_kwargs()
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            **training_args_kwargs,
        )

        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            )

        # Wrap progress_callback into a HuggingFace TrainerCallback so we get
        # per-logging-step notifications without monkey-patching the trainer.
        if self.progress_callback is not None:
            from transformers import TrainerCallback, TrainerControl, TrainerState

            progress_cb = self.progress_callback

            class _ProgressBridge(TrainerCallback):
                def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        progress_cb(state.global_step, logs["loss"])

            callbacks.append(_ProgressBridge())

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            callbacks=callbacks,
        )

        train_result = trainer.train()

        # Evaluate once at end to capture final eval loss
        eval_result = trainer.evaluate()
        final_eval_loss = eval_result.get("eval_loss", float("nan"))

        # Save the LoRA adapter — not the full model — to keep artifacts small
        trainer.model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)

        duration = time.time() - start_time
        early_stopped = (
            train_result.training_loss
            != train_result.training_loss  # NaN guard; real check below
        )
        # Detect early stopping via epoch count falling short of configured total
        epochs_completed = int(train_result.metrics.get("epoch", self.config.num_epochs))
        early_stopped = epochs_completed < self.config.num_epochs

        return SFTResult(
            adapter_path=adapter_output_dir,
            final_train_loss=train_result.training_loss,
            final_eval_loss=final_eval_loss,
            training_steps=train_result.global_step,
            epochs_completed=epochs_completed,
            early_stopped=early_stopped,
            duration_seconds=duration,
        )

    def _load_model_and_tokenizer(self, model_name_or_path: str):
        """Load base model with BitsAndBytes 4-bit quantization.

        Purpose: materialise BitsAndBytesConfig from self.config.quantization
                 and load model + tokenizer in 4-bit NF4 double-quant mode.
        Inputs:  model_name_or_path — HuggingFace model ID or local path
        Outputs: (model, tokenizer) tuple
        Complexity: O(model_params) — dominated by weight loading
        Side effects: allocates GPU memory; may download weights
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        q = self.config.quantization
        compute_dtype = getattr(torch, q.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q.load_in_4bit,
            bnb_4bit_quant_type=q.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
        # Disable KV cache — incompatible with gradient checkpointing
        model.config.use_cache = False
        # Suppress extraneous token warnings from trl
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
        )
        # Ensure a pad token exists; use EOS if absent to avoid tokenizer errors
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _apply_lora(self, model):
        """Apply LoRA configuration and prepare model for k-bit training.

        Purpose: wrap the quantized base model with PEFT LoRA adapters and
                 prepare gradient computation paths for 4-bit training.
        Inputs:  model — quantized AutoModelForCausalLM
        Outputs: PEFT model with LoRA adapters attached
        Complexity: O(target_module_count)
        Side effects: modifies model in-place; adds trainable LoRA parameters
        """
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        lc = self.config.lora
        lora_config = LoraConfig(
            r=lc.r,
            lora_alpha=lc.lora_alpha,
            target_modules=lc.target_modules,
            lora_dropout=lc.lora_dropout,
            bias=lc.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # prepare_model_for_kbit_training enables gradient checkpointing and
        # casts non-quantized layers (norms, embeddings) to float32 for stable
        # gradient flow — essential for correct NF4 fine-tuning.
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        return model


class MockSFTJob:
    """Simulates SFT training for tests without GPU or model download.

    Purpose: deterministic, fast stand-in for SFTTrainingJob that produces
             the same output contract (SFTResult + adapter_config.json) without
             any external dependencies.
    Inputs:
      config            — SFTTrainingConfig (logging_steps and num_epochs used)
      output_dir        — directory where the fake adapter is written
      progress_callback — optional callable(step, loss) called during simulation
    """

    def __init__(
        self,
        config: SFTTrainingConfig,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback

    def run(self, model_name: str, train_dataset, eval_dataset) -> SFTResult:
        """Simulate a training run and write a fake adapter artifact.

        Purpose: exercise all code paths that consume SFTResult without
                 requiring a GPU, internet connection, or heavy ML libs.
        Inputs:
          model_name    — ignored (may start with "mock://")
          train_dataset — used only for len() to determine step count
          eval_dataset  — ignored
        Outputs: SFTResult with deterministic but realistic-looking values
        Complexity: O(steps)
        Side effects: creates output_dir/sft_adapter/adapter_config.json
        """
        adapter_dir = os.path.join(self.output_dir, "sft_adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        # Cap steps to dataset size so we don't overshoot on tiny datasets
        steps = min(100, len(train_dataset))
        start_time = time.time()

        # Simulate training loop with exponentially decaying loss — the decay
        # base (0.95) ensures loss strictly decreases across logged intervals.
        for step in range(0, steps, self.config.logging_steps):
            loss = 2.0 * (0.95 ** step)
            if self.progress_callback:
                self.progress_callback(step, loss)
            time.sleep(0.001)  # simulate compute work

        duration = time.time() - start_time

        # Write minimal adapter_config.json so downstream consumers can verify
        # the adapter directory is well-formed without loading real weights.
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"r": self.config.lora.r, "mock": True}, f)

        return SFTResult(
            adapter_path=adapter_dir,
            final_train_loss=0.15,
            final_eval_loss=0.18,
            training_steps=steps,
            epochs_completed=self.config.num_epochs,
            early_stopped=False,
            duration_seconds=duration,
        )
