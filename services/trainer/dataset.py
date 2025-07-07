"""Dataset loading and preprocessing for SFT training.

All public functions maintain the invariant that the returned Dataset has
a single 'text' column formatted as ChatML-style instruction sequences.
"""
import json
from datasets import Dataset


def load_sft_dataset_from_jsonl(path: str) -> Dataset:
    """Load {prompt, response} JSONL into a HuggingFace Dataset.

    Purpose: parse a JSONL file where each line is {"prompt": str, "response": str}
             and convert to ChatML-formatted text records.
    Inputs:  path — absolute or relative path to the JSONL file
    Outputs: Dataset with a single 'text' column
    Complexity: O(N) where N = number of lines
    Side effects: opens and reads the file at `path`
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert "prompt" in record and "response" in record, (
                f"Record missing 'prompt' or 'response': {record}"
            )
            records.append({
                "text": format_sft_prompt(record["prompt"], record["response"])
            })
    return Dataset.from_list(records)


def format_sft_prompt(prompt: str, response: str) -> str:
    """Format a prompt/response pair as ChatML-style instruction.

    Purpose: produce a canonical ChatML string for SFT training.
    Inputs:  prompt — user turn text; response — assistant turn text
    Outputs: formatted string with <|im_start|>/<|im_end|> markers
    Complexity: O(len(prompt) + len(response))
    Side effects: none
    """
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def split_dataset(dataset: Dataset, eval_fraction: float = 0.1) -> tuple:
    """Split a Dataset into train and eval subsets.

    Purpose: reproducible train/eval split for SFT.
    Inputs:  dataset — HuggingFace Dataset; eval_fraction — fraction for eval (0,1)
    Outputs: (train_dataset, eval_dataset) tuple
    Complexity: O(N)
    Side effects: none
    """
    split = dataset.train_test_split(test_size=eval_fraction, seed=42)
    return split["train"], split["test"]


def validate_dataset(dataset: Dataset) -> dict:
    """Validate a Dataset and return descriptive statistics.

    Purpose: surface data quality issues early; guard against empty records
             that would corrupt training loss.
    Inputs:  dataset — HuggingFace Dataset with a 'text' column
    Outputs: dict with total_records, avg/min/max text_length, empty_records count
    Complexity: O(N)
    Side effects: raises AssertionError on empty records or empty dataset
    """
    assert len(dataset) > 0, "Dataset is empty"
    lengths = [len(r["text"]) for r in dataset]
    stats = {
        "total_records": len(dataset),
        "avg_text_length": sum(lengths) / len(lengths),
        "min_text_length": min(lengths),
        "max_text_length": max(lengths),
        "empty_records": sum(1 for length in lengths if length == 0),
    }
    assert stats["empty_records"] == 0, (
        f"Dataset has {stats['empty_records']} empty records"
    )
    return stats
