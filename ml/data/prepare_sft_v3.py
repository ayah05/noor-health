"""
Export multilingual_cases_v3.jsonl into train / validation / test SFT files.

Output: data/processed/v3/{train,validation,test}_sft.jsonl
(same record format as v2, so train.py can read it unchanged).

Usage (from the repository root):
    python ml/data/prepare_sft_v3.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR  # noqa: E402


INPUT_FILE = SYNTHETIC_DATA_DIR / "multilingual_cases_v3.jsonl"
OUTPUT_DIR = PROCESSED_DATA_DIR / "v3"
SPLITS = ("train", "validation", "test")


def main():
    records = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Guard against cross-split leakage: one case_id must have one split.
    split_of = defaultdict(set)
    for r in records:
        split_of[r["case_id"]].add(r["split"])
    leaking = [cid for cid, splits in split_of.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"case_ids in more than one split: {leaking[:5]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    per_lang = defaultdict(Counter)

    for split in SPLITS:
        path = OUTPUT_DIR / f"{split}_sft.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                if r["split"] != split:
                    continue
                sample = {
                    "case_id": r["case_id"],
                    "split": r["split"],
                    "language": r["language"],
                    "utterance": r["utterance"],
                    "target": r["target"],
                    "input_mode": r["plan"]["input_mode"],
                    "source": r["source"],
                    "schema_version": r["schema_version"],
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[split] += 1
                per_lang[split][r["language"]] += 1
        print(f"{split:<11} {counts[split]:>5}  {dict(per_lang[split])}  -> {path}")

    plans = [r["plan"] for r in records]
    print("\nCOVERAGE OF NEW PHENOMENA")
    print(f"  chief complaint not first: {sum(bool(p['chief_cue']) for p in plans)}")
    print(f"  coordinated durations:     {sum(bool(p['coordinated']) for p in plans)}")
    print(f"  speech-like (asr_style):   {sum(p['input_mode'] == 'asr_style' for p in plans)}")
    print(f"  code-switching:            {sum(bool(p['code_switch_symptom']) for p in plans)}")
    modes = Counter(s["time"]["mode"] for p in plans for s in p["symptoms"])
    print(f"  time renderings:           {dict(modes)}")


if __name__ == "__main__":
    main()