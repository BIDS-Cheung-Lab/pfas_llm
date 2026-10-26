"""Load data/all.jsonl (written by prep_data.py) and split it into train / test.

The split comes from data/split.csv (50 train / 50 test abstracts, no overlap).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def load_docs():
    with open(ROOT / "data" / "all.jsonl") as fh:
        return [json.loads(l) for l in fh]


def load_split():
    """(train, test) docs."""
    docs = load_docs()
    return ([d for d in docs if d["split"] == "train"],
            [d for d in docs if d["split"] == "test"])
