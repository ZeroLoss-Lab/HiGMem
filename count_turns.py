# count_turns.py
import sys
from pathlib import Path
from load_dataset import load_locomo_dataset, LoCoMoSample
from typing import List
import io


class SuppressPrint:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def count_total_turns(samples: List[LoCoMoSample]) -> int:
    total_turns = 0
    for sample in samples:
        for session in sample.conversation.sessions.values():
            total_turns += len(session.turns)

    return total_turns


if __name__ == "__main__":
    dataset_path = Path("data") / "locomo10.json"

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at the following path: {dataset_path}")
        print("Please ensure the 'locomo10.json' file exists in the 'data' folder.")
    else:
        print(f"Loading dataset from '{dataset_path}' and counting turns...")

        with SuppressPrint():
            loaded_samples = load_locomo_dataset(dataset_path)

        total_turns_count = count_total_turns(loaded_samples)

        print("\n" + "=" * 40)
        print(f"Counting complete.")
        print(f"Total number of conversation turns in locomo10.json: {total_turns_count}")
        print("=" * 40)