"""Dataset serialization and loading utilities for evaluation."""

import json
from pathlib import Path

from structured_ie.schemas.models import LabeledSample


def load_dataset_from_json(file_path: str | Path) -> list[LabeledSample]:
    """Load evaluation test cases from a JSON file.

    Args:
        file_path: Path to the JSON dataset file.

    Returns:
        List of validated LabeledSample objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}, got {type(data).__name__}")

    return [LabeledSample.model_validate(item) for item in data]


def save_dataset_to_json(samples: list[LabeledSample], file_path: str | Path) -> None:
    """Save a list of LabeledSample instances to a JSON file.

    Args:
        samples: List of LabeledSample objects.
        file_path: Destination path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for s in samples:
        serializable.append(
            {
                "text": s.text,
                "expected_entities": [e.to_dict() for e in s.expected_entities],
                "expected_relations": [r.to_dict() for r in s.expected_relations],
                "metadata": s.metadata,
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
