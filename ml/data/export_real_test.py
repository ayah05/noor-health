"""
Export the real-world test set in the format expected by
ml/evaluation/run_lora_modal.py.

The current LoRA model was trained with language code "ar_msa"
only. Egyptian Arabic records are therefore passed as "ar_msa",
so the measurement isolates the dialect effect. The original
code is kept in "language_original" for the per-dialect analysis.

Usage (from the repository root):
    python ml/data/validate_real_test.py
    python ml/data/export_real_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402


INPUT_FILE = DATA_DIR / "real_test" / "real_test_v1.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "real_test_sft.jsonl"

LANGUAGE_MAP_FOR_CURRENT_MODEL = {
    "ar_eg": "ar_msa",
}


def main():
    records = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            sample = {
                "case_id": r["case_id"],
                "split": "real_test",
                "language": LANGUAGE_MAP_FOR_CURRENT_MODEL.get(r["language"], r["language"]),
                "language_original": r["language"],
                "input_mode": r["input_mode"],
                "utterance": r["utterance"],
                "target": r["target"],
                "source": r["author"],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"exported {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
