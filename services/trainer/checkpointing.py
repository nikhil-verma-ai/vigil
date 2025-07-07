"""Checkpoint management for SFT and DPO training runs.

Checkpoints are stored on disk as:
  <base_dir>/ckpt-<cycle_id>-step<step>/
    metadata.json   — JSON-serialised CheckpointMetadata
    adapter/        — symlink to the adapter directory at save time

Invariants:
  - Every checkpoint directory contains exactly one metadata.json
  - list_checkpoints returns checkpoints sorted ascending by step
  - load_latest returns the CheckpointMetadata with the highest step for the
    given cycle_id, or None if no checkpoints exist
  - cleanup_old_checkpoints leaves exactly keep_last_n checkpoints,
    removing those with the lowest step values first
"""
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each training checkpoint.

    Fields:
      cycle_id   — training cycle identifier (e.g. "cycle-001")
      step       — global optimizer step at checkpoint save time
      phase      — training phase: "sft" | "dpo"
      train_loss — training loss at this step
      eval_loss  — evaluation loss at this step
      timestamp  — ISO-8601 UTC timestamp string
    """
    cycle_id: str
    step: int
    phase: str
    train_loss: float
    eval_loss: float
    timestamp: str


class CheckpointManager:
    """Manages checkpoint lifecycle: save, load, list, and cleanup.

    Purpose: provide a consistent interface for persisting training state so
             that runs can be resumed and old checkpoints pruned to save disk.
    Inputs:  base_dir — root directory where checkpoint subdirectories are created
    Side effects: reads/writes/deletes files under base_dir
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def save(self, metadata: CheckpointMetadata, adapter_path: str) -> str:
        """Save checkpoint metadata and create a symlink to the adapter.

        Purpose: persist training state for a given cycle_id and step so it
                 can be restored later via load_latest.
        Inputs:
          metadata     — CheckpointMetadata describing this checkpoint
          adapter_path — absolute or resolvable path to the adapter directory
        Outputs: checkpoint_id string (e.g. "ckpt-cycle-001-step100")
        Complexity: O(1) filesystem ops
        Side effects: creates directory + metadata.json + adapter symlink under base_dir
        """
        checkpoint_id = f"ckpt-{metadata.cycle_id}-step{metadata.step}"
        ckpt_dir = self.base_dir / checkpoint_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f)

        # Use an absolute path for the symlink target so it remains valid even
        # if the working directory changes at load time.
        adapter_symlink = ckpt_dir / "adapter"
        abs_adapter = os.path.abspath(adapter_path)

        # Re-create symlink if a stale one exists (e.g. from a previous run at
        # the same step that was interrupted before finalisation).
        if adapter_symlink.exists() or adapter_symlink.is_symlink():
            adapter_symlink.unlink()
        adapter_symlink.symlink_to(abs_adapter)

        return checkpoint_id

    def load_latest(self, cycle_id: str) -> Optional[CheckpointMetadata]:
        """Find the latest checkpoint for a cycle_id by step number.

        Purpose: resume training from the last saved state.
        Inputs:  cycle_id — training cycle identifier
        Outputs: CheckpointMetadata for the highest step, or None if absent
        Complexity: O(N) where N = number of checkpoints for this cycle
        Side effects: reads metadata.json files from disk
        """
        checkpoints = self.list_checkpoints(cycle_id)
        if not checkpoints:
            return None
        # list_checkpoints is sorted ascending; last element is the latest
        return checkpoints[-1]

    def list_checkpoints(self, cycle_id: str) -> List[CheckpointMetadata]:
        """Return all checkpoints for a cycle_id, sorted ascending by step.

        Purpose: enumerate available checkpoints for display or cleanup logic.
        Inputs:  cycle_id — training cycle identifier
        Outputs: list of CheckpointMetadata sorted by step (lowest first)
        Complexity: O(N log N) where N = matching checkpoint directories
        Side effects: reads metadata.json files from disk
        """
        if not self.base_dir.exists():
            return []

        prefix = f"ckpt-{cycle_id}-step"
        results: List[CheckpointMetadata] = []

        for entry in self.base_dir.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix):
                continue
            meta_file = entry / "metadata.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                data = json.load(f)
            results.append(CheckpointMetadata(**data))

        # Sort ascending by step so callers can rely on positional ordering
        results.sort(key=lambda m: m.step)
        return results

    def cleanup_old_checkpoints(self, cycle_id: str, keep_last_n: int = 3) -> None:
        """Delete all but the last keep_last_n checkpoints for a cycle.

        Purpose: prevent unbounded disk growth during long training runs by
                 pruning low-step checkpoints that are no longer needed.
        Inputs:
          cycle_id    — training cycle identifier
          keep_last_n — number of most recent (highest step) checkpoints to retain
        Outputs: None
        Complexity: O(N) where N = number of checkpoints for this cycle
        Side effects: deletes checkpoint directories (metadata + symlink) from disk
        """
        checkpoints = self.list_checkpoints(cycle_id)
        if len(checkpoints) <= keep_last_n:
            return

        # checkpoints is sorted ascending by step; drop the oldest (front)
        to_delete = checkpoints[: len(checkpoints) - keep_last_n]
        for meta in to_delete:
            checkpoint_id = f"ckpt-{meta.cycle_id}-step{meta.step}"
            ckpt_dir = self.base_dir / checkpoint_id
            if ckpt_dir.exists():
                # Remove the symlink manually before rmtree to avoid following it
                adapter_symlink = ckpt_dir / "adapter"
                if adapter_symlink.is_symlink():
                    adapter_symlink.unlink()
                shutil.rmtree(ckpt_dir)
