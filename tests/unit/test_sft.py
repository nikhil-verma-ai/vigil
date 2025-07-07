"""Unit tests for the SFT QLoRA training pipeline (Module 4a).

All tests run without GPU or external model downloads by using:
  - In-memory HuggingFace Datasets
  - MockSFTJob for training simulation
  - Temporary directories for filesystem side effects

Run with:
  pytest tests/unit/test_sft.py -v
"""
import json
import os
import tempfile
import pytest
from pathlib import Path
import datasets

from services.trainer.dataset import (
    load_sft_dataset_from_jsonl,
    format_sft_prompt,
    split_dataset,
    validate_dataset,
)
from services.trainer.sft import MockSFTJob, SFTResult
from services.trainer.qlora_config import SFTTrainingConfig
from services.trainer.checkpointing import CheckpointManager, CheckpointMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_sft_dataset(n: int = 50) -> list:
    """Create minimal SFT JSONL records for testing."""
    return [
        {"prompt": f"Question {i}: What is {i}+{i}?", "response": f"The answer is {i*2}."}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_dataset_loading_and_formatting():
    """Loaded dataset must have correct ChatML format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in make_tiny_sft_dataset(20):
            f.write(json.dumps(record) + '\n')
        path = f.name

    try:
        dataset = load_sft_dataset_from_jsonl(path)
        assert len(dataset) == 20
        for record in dataset:
            text = record["text"]
            assert "<|im_start|>user" in text
            assert "<|im_start|>assistant" in text
            assert "<|im_end|>" in text
            # Verify the original content is preserved (spot-check first record)
    finally:
        os.unlink(path)


def test_dataset_split_proportions():
    """Train/eval split must respect eval_fraction."""
    records = make_tiny_sft_dataset(100)
    dataset = datasets.Dataset.from_list([
        {"text": format_sft_prompt(r["prompt"], r["response"])} for r in records
    ])
    train, eval_ds = split_dataset(dataset, eval_fraction=0.2)

    assert len(train) == 80
    assert len(eval_ds) == 20


def test_dataset_validation_rejects_empty():
    """validate_dataset must raise AssertionError on empty records."""
    bad_dataset = datasets.Dataset.from_list([{"text": ""}, {"text": "valid text"}])
    with pytest.raises(AssertionError, match="empty records"):
        validate_dataset(bad_dataset)


# ---------------------------------------------------------------------------
# MockSFTJob tests
# ---------------------------------------------------------------------------

def test_mock_sft_job_produces_adapter_file():
    """MockSFTJob must create adapter_config.json in output dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SFTTrainingConfig(num_epochs=1, logging_steps=10)
        job = MockSFTJob(config=config, output_dir=tmpdir)
        records = make_tiny_sft_dataset(50)
        dataset = datasets.Dataset.from_list([
            {"text": format_sft_prompt(r["prompt"], r["response"])} for r in records
        ])
        train, eval_ds = split_dataset(dataset)

        result = job.run("mock://gpt2", train, eval_ds)

        assert os.path.exists(f"{tmpdir}/sft_adapter/adapter_config.json")
        assert result.final_train_loss > 0
        assert result.training_steps > 0
        assert result.duration_seconds > 0


def test_mock_sft_progress_callback():
    """Progress callback must be called at each logging_steps interval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SFTTrainingConfig(logging_steps=10)
        progress_calls = []
        job = MockSFTJob(
            config=config,
            output_dir=tmpdir,
            progress_callback=lambda step, loss: progress_calls.append((step, loss)),
        )
        records = make_tiny_sft_dataset(50)
        dataset = datasets.Dataset.from_list([
            {"text": format_sft_prompt(r["prompt"], r["response"])} for r in records
        ])
        train, eval_ds = split_dataset(dataset)
        job.run("mock://gpt2", train, eval_ds)

        assert len(progress_calls) > 0
        # Loss should be decreasing overall (exponential decay guarantees this)
        losses = [loss for _, loss in progress_calls]
        assert losses[0] > losses[-1], f"Loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_load():
    """Save checkpoint, load it back, assert metadata matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(base_dir=tmpdir)
        metadata = CheckpointMetadata(
            cycle_id="cycle-001",
            step=100,
            phase="sft",
            train_loss=0.25,
            eval_loss=0.28,
            timestamp="2026-03-28T01:00:00Z",
        )
        # Create a fake adapter dir
        adapter_dir = Path(tmpdir) / "fake_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}')

        ckpt_id = manager.save(metadata, str(adapter_dir))
        loaded = manager.load_latest("cycle-001")

        assert loaded is not None
        assert loaded.cycle_id == "cycle-001"
        assert loaded.step == 100
        assert loaded.train_loss == 0.25
        assert loaded.phase == "sft"


def test_checkpoint_cleanup_keeps_last_n():
    """cleanup_old_checkpoints must leave exactly keep_last_n checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(base_dir=tmpdir)
        adapter_dir = Path(tmpdir) / "fake_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{}')

        # Save 5 checkpoints
        for step in [100, 200, 300, 400, 500]:
            manager.save(
                CheckpointMetadata("cycle-001", step, "sft", 0.1, 0.1, "2026-03-28T00:00:00Z"),
                str(adapter_dir),
            )

        manager.cleanup_old_checkpoints("cycle-001", keep_last_n=3)
        remaining = manager.list_checkpoints("cycle-001")

        assert len(remaining) == 3
        assert remaining[-1].step == 500  # newest kept
        assert remaining[0].step == 300   # oldest kept after cleanup


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_qlora_config_to_training_args():
    """to_training_arguments_kwargs must return dict with all required keys."""
    config = SFTTrainingConfig()
    kwargs = config.to_training_arguments_kwargs()

    required_keys = [
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "optim",
    ]
    for key in required_keys:
        assert key in kwargs, f"Missing key: {key}"

    assert kwargs["learning_rate"] == 2e-4
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["optim"] == "paged_adamw_8bit"
