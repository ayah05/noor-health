import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

MULTILINGUAL_FILE = (
    ROOT / "data" / "synthetic" / "multilingual_cases_v2_full.jsonl"
)

SEMANTIC_FILE = (
    ROOT / "results" / "semantic_validation.jsonl"
)

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "train_sft.jsonl",
    "validation": OUTPUT_DIR / "validation_sft.jsonl",
    "test": OUTPUT_DIR / "test_sft.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    print("=" * 60)
    print("NOOR HEALTH - PREPARE FINAL TRAINING DATA")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    samples = load_jsonl(MULTILINGUAL_FILE)
    qc_results = load_jsonl(SEMANTIC_FILE)

    print(f"\nDataset samples: {len(samples)}")
    print(f"QC results:      {len(qc_results)}")

    if len(samples) != 7000:
        raise ValueError(
            f"Expected 7000 samples, found {len(samples)}"
        )

    if len(qc_results) != 7000:
        raise ValueError(
            f"Expected 7000 QC results, found {len(qc_results)}"
        )

    # ---------------------------------------------------------
    # BUILD QC LOOKUP
    # ---------------------------------------------------------

    qc_lookup = {}

    for result in qc_results:
        key = (
            result["case_id"],
            result["language"],
        )

        if key in qc_lookup:
            raise ValueError(
                f"Duplicate QC key: {key}"
            )

        qc_lookup[key] = result

    # ---------------------------------------------------------
    # FILTER
    # ---------------------------------------------------------

    accepted = []
    rejected = []

    for sample in samples:
        key = (
            sample["case_id"],
            sample["language"],
        )

        if key not in qc_lookup:
            raise ValueError(
                f"No QC result found for {key}"
            )

        qc = qc_lookup[key]

        if (
            qc["semantic_passed"]
            and qc["language_passed"]
        ):
            accepted.append(sample)
        else:
            rejected.append(
                {
                    "case_id": sample["case_id"],
                    "language": sample["language"],
                    "split": sample["split"],
                    "semantic_passed":
                        qc["semantic_passed"],
                    "language_passed":
                        qc["language_passed"],
                    "semantic_errors":
                        qc.get("semantic_errors", []),
                    "language_errors":
                        qc.get("language_errors", []),
                }
            )

    print(f"\nAccepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")

    if len(accepted) != 6988:
        raise ValueError(
            f"Expected 6988 accepted samples, "
            f"found {len(accepted)}"
        )

    if len(rejected) != 12:
        raise ValueError(
            f"Expected 12 rejected samples, "
            f"found {len(rejected)}"
        )

    # ---------------------------------------------------------
    # SPLIT WITHOUT RESHUFFLING
    # ---------------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for sample in accepted:
        split = sample["split"]

        if split not in splits:
            raise ValueError(
                f"Unknown split: {split}"
            )

        splits[split].append(sample)

    # ---------------------------------------------------------
    # CASE LEAKAGE CHECK
    # ---------------------------------------------------------

    case_sets = {
        split: {
            sample["case_id"]
            for sample in records
        }
        for split, records in splits.items()
    }

    if case_sets["train"] & case_sets["validation"]:
        raise ValueError(
            "Case leakage between train and validation."
        )

    if case_sets["train"] & case_sets["test"]:
        raise ValueError(
            "Case leakage between train and test."
        )

    if case_sets["validation"] & case_sets["test"]:
        raise ValueError(
            "Case leakage between validation and test."
        )

    # ---------------------------------------------------------
    # WRITE FILES
    # ---------------------------------------------------------

    for split, records in splits.items():
        path = OUTPUT_FILES[split]

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"{split:<12} "
            f"{len(records):>5} samples "
            f"({len(case_sets[split])} cases)"
        )

    # ---------------------------------------------------------
    # REPORT REJECTED SAMPLES
    # ---------------------------------------------------------

    rejection_file = (
        ROOT
        / "results"
        / "training_samples_rejected.jsonl"
    )

    with rejection_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in rejected:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # LANGUAGE DISTRIBUTION
    # ---------------------------------------------------------

    print("\nAccepted samples by language:")

    language_counts = Counter(
        sample["language"]
        for sample in accepted
    )

    for language in sorted(language_counts):
        print(
            f"  {language:<8} "
            f"{language_counts[language]}"
        )

    print("\nRejected samples:")

    for sample in rejected:
        print(
            f"  {sample['case_id']}/"
            f"{sample['language']} "
            f"[{sample['split']}]"
        )

    print("\nFinal training dataset ready.")
    print(f"Rejected sample report: {rejection_file}")


if __name__ == "__main__":
    main()