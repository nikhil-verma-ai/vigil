"""SFT QLoRA training pipeline for the autonomous fine-tuning platform.

Public API:
  - qlora_config.SFTTrainingConfig
  - qlora_config.LoRAConfig
  - qlora_config.QuantizationConfig
  - dataset.load_sft_dataset_from_jsonl
  - dataset.format_sft_prompt
  - dataset.split_dataset
  - dataset.validate_dataset
  - sft.SFTTrainingJob
  - sft.MockSFTJob
  - sft.SFTResult
  - checkpointing.CheckpointManager
  - checkpointing.CheckpointMetadata
"""
