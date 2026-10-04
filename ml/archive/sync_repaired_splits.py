import json
from pathlib import Path

from config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# PATHS
# ============================================================

SOURCE_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

SPLIT_FILES = {
    "train":
        PROCESSED_DATA_DIR
        / "train_cases.jsonl",

    "validation":
        PROCESSED_DATA_DIR
        / "validation_cases.jsonl",

    "test":
        PROCESSED_DATA_DIR
        / "test_cases.jsonl",
}


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                records.append(
                    json.loads(line)
                )

    return records


def save_jsonl(
    path: Path,
    records: list[dict],
):

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "SYNC REPAIRED SPLITS"
    )
    print("=" * 60)

    repaired_cases = load_jsonl(
        SOURCE_FILE
    )

    repaired_by_id = {
        case["case_id"]: case
        for case in repaired_cases
    }

    print(
        f"\nCanonical cases loaded: "
        f"{len(repaired_cases)}"
    )

    all_split_ids = set()

    total_replaced = 0

    for split_name, split_file in (
        SPLIT_FILES.items()
    ):

        old_cases = load_jsonl(
            split_file
        )

        old_ids = [
            case["case_id"]
            for case in old_cases
        ]

        # Important:
        # preserve exact split membership AND order.
        new_cases = []

        replaced = 0

        for old_case in old_cases:

            case_id = old_case[
                "case_id"
            ]

            if case_id not in repaired_by_id:

                raise ValueError(
                    f"{case_id} from "
                    f"{split_name} not found "
                    "in canonical dataset"
                )

            new_case = repaired_by_id[
                case_id
            ]

            if new_case != old_case:
                replaced += 1

            new_cases.append(
                new_case
            )

        new_ids = [
            case["case_id"]
            for case in new_cases
        ]

        # Split membership/order must not change.
        if old_ids != new_ids:

            raise ValueError(
                f"{split_name}: "
                "case IDs or ordering changed"
            )

        overlap = (
            all_split_ids
            & set(new_ids)
        )

        if overlap:

            raise ValueError(
                f"Split leakage detected: "
                f"{sorted(overlap)[:5]}"
            )

        all_split_ids.update(
            new_ids
        )

        save_jsonl(
            split_file,
            new_cases,
        )

        total_replaced += replaced

        print(
            f"\n{split_name:<12}"
            f"{len(new_cases):>4} cases"
        )

        print(
            f"  updated: {replaced}"
        )

    # --------------------------------------------------------
    # FINAL CHECKS
    # --------------------------------------------------------

    if len(all_split_ids) != 1000:

        raise ValueError(
            f"Expected 1000 unique cases "
            f"across splits, found "
            f"{len(all_split_ids)}"
        )

    canonical_ids = set(
        repaired_by_id.keys()
    )

    if all_split_ids != canonical_ids:

        raise ValueError(
            "Split case IDs do not exactly "
            "match canonical dataset"
        )

    print()
    print("=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal cases updated "
        f"across splits: "
        f"{total_replaced}"
    )

    print(
        "\nUnique cases across splits: "
        f"{len(all_split_ids)}"
    )

    print(
        "\nSplit membership preserved."
    )

    print(
        "No case leakage detected."
    )


if __name__ == "__main__":
    main()