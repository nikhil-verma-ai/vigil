"""
DPO (Direct Preference Optimisation) training pipeline.

Provides:
  - DPOConfig:             all DPO hyperparameters
  - DPOResult:             structured output from a completed DPO run
  - DPOTrainingJob:        real training via trl.DPOTrainer (GPU required)
  - MockDPOJob:            CPU-only simulator for tests; same output contract
  - load_dpo_dataset_from_jsonl: load {prompt, chosen, rejected} JSONL to Dataset

Invariants:
  - adapter_path always points to a directory containing adapter_config.json
  - DPOResult.reward_accuracy is always in [0.0, 1.0]
  - MockDPOJob.reward_accuracy monotonically improves over training steps
  - If model_name_or_path starts with "mock://", DPOTrainingJob delegates to
    MockDPOJob — no GPU or internet access required
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """
    All hyperparameters for a DPO training run.

    Fields:
      beta                          — KL-divergence penalty weight (DPO β).
                                      Larger values keep the policy closer to
                                      the reference model.
      loss_type                     — "sigmoid" (standard DPO) | "simpo"
      learning_rate                 — AdamW learning rate
      num_train_epochs              — number of full passes over training data
      per_device_train_batch_size   — batch size per GPU
      gradient_accumulation_steps   — simulate larger batch via grad accumulation
      max_length                    — max tokens for chosen/rejected sequences
      max_prompt_length             — max tokens for the prompt prefix
      logging_steps                 — frequency of loss/metrics logging
    """
    beta: float = 0.1
    loss_type: str = "sigmoid"          # "sigmoid" | "simpo"
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    max_prompt_length: int = 1024
    logging_steps: int = 10


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DPOResult:
    """
    Structured output from a completed DPO training run.

    Fields:
      adapter_path      — directory containing saved LoRA adapter weights
      final_train_loss  — DPO loss at the last logged training step
      final_eval_loss   — DPO loss on the evaluation split at run end
      training_steps    — total number of gradient update steps
      duration_seconds  — wall-clock training time
      reward_accuracy   — fraction of steps where chosen reward > rejected reward;
                          a value > 0.5 indicates the policy prefers chosen outputs
    """
    adapter_path: str
    final_train_loss: float
    final_eval_loss: float
    training_steps: int
    duration_seconds: float
    reward_accuracy: float


# ---------------------------------------------------------------------------
# Real job
# ---------------------------------------------------------------------------

class DPOTrainingJob:
    """
    Full DPO training using trl.DPOTrainer with a LoRA adapter.

    Purpose:  fine-tune a policy adapter using direct preference optimisation
              against a frozen reference model.
    Inputs:
      config            — DPOConfig with all hyperparameters
      output_dir        — directory where the adapter is written
      progress_callback — optional callable(step, loss, reward_accuracy)
    Outputs: DPOResult
    Side effects: writes adapter weights + config to output_dir/dpo_adapter/
    """

    def __init__(
        self,
        config: DPOConfig,
        output_dir: str,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback

    def run(
        self,
        model_name_or_path: str,
        reference_model_path: str,
        train_dataset,
        eval_dataset,
    ) -> DPOResult:
        """
        Execute a DPO training run.

        Purpose:  load SFT-warm-started policy and reference model, run
                  trl.DPOTrainer, save adapter, return result.
        Inputs:
          model_name_or_path    — SFT adapter path (warm start); if starts with
                                  "mock://" the run is delegated to MockDPOJob
          reference_model_path  — production adapter used as KL reference model
          train_dataset         — HuggingFace Dataset with {prompt, chosen, rejected}
          eval_dataset          — HuggingFace Dataset with {prompt, chosen, rejected}
        Outputs: DPOResult
        Complexity: O(steps * model_size) real; O(steps) mock
        Side effects: writes to self.output_dir/dpo_adapter/
        """
        # Fast path: mock mode avoids GPU / heavy imports for tests
        if model_name_or_path.startswith("mock://"):
            mock = MockDPOJob(
                config=self.config,
                output_dir=self.output_dir,
                progress_callback=self.progress_callback,
            )
            return mock.run(model_name_or_path, reference_model_path, train_dataset, eval_dataset)

        # --- Real training path (GPU required) ---
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import PeftModel
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig

        t0 = time.monotonic()
        adapter_dir = os.path.join(self.output_dir, "dpo_adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        log.info("dpo_loading_policy", path=model_name_or_path)

        tokeniser = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token

        # Policy model: SFT adapter warm-start
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Reference model: frozen production adapter; loaded separately so that
        # trl can hold both in memory simultaneously for KL divergence computation.
        ref_model = AutoModelForCausalLM.from_pretrained(
            reference_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Disable cache for gradient-checkpoint-compatible training
        policy_model.config.use_cache = False
        ref_model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir=adapter_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            bf16=True,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            remove_unused_columns=False,
        )

        reward_acc_history: list[float] = []

        # Wrap progress_callback into a HuggingFace TrainerCallback
        callbacks = []
        if self.progress_callback is not None:
            from transformers import TrainerCallback, TrainerState, TrainerControl

            progress_cb = self.progress_callback
            acc_ref = reward_acc_history  # captured by closure

            class _DPOProgressBridge(TrainerCallback):
                def on_log(
                    self,
                    args,
                    state: TrainerState,
                    control: TrainerControl,
                    logs=None,
                    **kwargs,
                ):
                    if logs:
                        loss = logs.get("loss", 0.0)
                        # trl logs rewards/accuracies as
                        # rewards/accuracies or reward_accuracies depending on version
                        acc = logs.get(
                            "rewards/accuracies",
                            logs.get("reward_accuracies", 0.5),
                        )
                        acc_ref.append(acc)
                        progress_cb(state.global_step, loss, acc)

            callbacks.append(_DPOProgressBridge())

        dpo_trainer = DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            beta=self.config.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokeniser,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            loss_type=self.config.loss_type,
            callbacks=callbacks,
        )

        train_output = dpo_trainer.train()
        eval_metrics = dpo_trainer.evaluate()

        dpo_trainer.model.save_pretrained(adapter_dir)
        tokeniser.save_pretrained(adapter_dir)

        duration = time.monotonic() - t0

        final_reward_acc = (
            reward_acc_history[-1] if reward_acc_history else
            eval_metrics.get("eval_rewards/accuracies", 0.5)
        )

        log.info(
            "dpo_complete",
            steps=train_output.global_step,
            reward_accuracy=final_reward_acc,
            duration_s=duration,
        )

        return DPOResult(
            adapter_path=adapter_dir,
            final_train_loss=train_output.training_loss,
            final_eval_loss=eval_metrics.get("eval_loss", 0.0),
            training_steps=train_output.global_step,
            duration_seconds=duration,
            reward_accuracy=min(max(final_reward_acc, 0.0), 1.0),
        )


# ---------------------------------------------------------------------------
# Mock job (CPU / test environments)
# ---------------------------------------------------------------------------

class MockDPOJob:
    """
    Simulates DPO training without GPU for unit and integration tests.

    Purpose:  deterministic stand-in for DPOTrainingJob that:
              - writes adapter_config.json with beta and loss_type
              - simulates reward_accuracy improving from 0.5 toward ~0.8
              - calls progress_callback at every logging_steps interval
    Inputs:   same constructor signature as DPOTrainingJob
    Invariant: reward_accuracy at end > reward_accuracy at start (always > 0.5)
    Side effects: creates output_dir/dpo_adapter/adapter_config.json
    """

    def __init__(
        self,
        config: DPOConfig,
        output_dir: str,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback

    def run(
        self,
        model_path: str,
        reference_path: str,
        train_dataset,
        eval_dataset,
    ) -> DPOResult:
        """
        Simulate a DPO training run.

        Purpose:  exercise orchestrator + test-suite code paths without real GPU.
        Inputs:
          model_path      — ignored (may start with "mock://")
          reference_path  — ignored
          train_dataset   — used for len() to determine step count
          eval_dataset    — ignored
        Outputs: DPOResult with plausible monotonically-improving reward_accuracy
        Complexity: O(steps) — bounded at min(50, len(train_dataset))
        Side effects: creates output_dir/dpo_adapter/adapter_config.json
        """
        adapter_dir = os.path.join(self.output_dir, "dpo_adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        # Cap step count to dataset size so tiny test datasets don't over-count
        steps = min(50, len(train_dataset))
        logging_steps = max(1, self.config.logging_steps)
        reward_acc_history: list[float] = []

        t0 = time.monotonic()

        for step in range(0, steps, logging_steps):
            # DPO loss decays; reward accuracy improves from 0.5 toward ~0.8
            loss = 1.5 * (0.92 ** step)
            # Linear ramp: 0.5 at step 0 → ~0.8 at steps-1
            reward_acc = 0.5 + (step / max(steps - 1, 1)) * 0.3
            reward_acc_history.append(reward_acc)
            if self.progress_callback:
                self.progress_callback(step, loss, reward_acc)
            time.sleep(0.001)

        duration = time.monotonic() - t0

        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as fh:
            json.dump(
                {
                    "beta": self.config.beta,
                    "mock": True,
                    "loss_type": self.config.loss_type,
                },
                fh,
            )

        final_acc = reward_acc_history[-1] if reward_acc_history else 0.5

        log.info(
            "mock_dpo_complete",
            steps=steps,
            reward_accuracy=final_acc,
            duration_s=duration,
        )

        return DPOResult(
            adapter_path=adapter_dir,
            final_train_loss=0.08,
            final_eval_loss=0.10,
            training_steps=steps,
            duration_seconds=duration,
            reward_accuracy=final_acc,
        )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dpo_dataset_from_jsonl(path: str):
    """
    Load a JSONL file of preference pairs into a HuggingFace Dataset.

    Purpose:  parse preference-pair JSONL for DPO training; validate schema.
    Inputs:   path — absolute or relative path to JSONL file
              Each line must be: {"prompt": str, "chosen": str, "rejected": str}
    Outputs:  HuggingFace Dataset with columns [prompt, chosen, rejected]
    Complexity: O(N) where N = number of lines
    Side effects: opens and reads the file at `path`
    Raises:   AssertionError if any record is missing required keys
    """
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert "prompt" in record and "chosen" in record and "rejected" in record, (
                f"DPO record missing required key — got keys: {list(record.keys())}"
            )
            records.append(record)

    from datasets import Dataset
    return Dataset.from_list(records)
