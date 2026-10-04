import json
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"
SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"


SEMANTIC_FILE = (
    RESULTS_DIR
    / "semantic_validation.jsonl"
)

BACKUP_FILE = (
    RESULTS_DIR
    / "semantic_validation_before_multilingual_repair.jsonl"
)

CLINICAL_REPAIR_REPORT = (
    SYNTHETIC_DIR
    / "clinical_cases_v2_repair_report.json"
)

LANGUAGES = {
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def main():
    print("=" * 60)
    print(
        "NOOR HEALTH - INVALIDATE REPAIRED SEMANTIC QC"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD EXISTING SEMANTIC RESULTS
    # --------------------------------------------------------

    semantic_results = load_jsonl(
        SEMANTIC_FILE
    )

    print(
        f"\nExisting semantic results: "
        f"{len(semantic_results)}"
    )

    if len(semantic_results) != 7000:
        raise ValueError(
            "Expected 7000 semantic validation results, "
            f"found {len(semantic_results)}"
        )

    # --------------------------------------------------------
    # LOAD REPAIRED CLINICAL CASE IDS
    # --------------------------------------------------------

    with CLINICAL_REPAIR_REPORT.open(
        "r",
        encoding="utf-8",
    ) as file:
        repair_report = json.load(file)

    repaired_case_ids = set(
        repair_report["changed_case_ids"]
    )

    print(
        f"Repaired clinical cases: "
        f"{len(repaired_case_ids)}"
    )

    if len(repaired_case_ids) != 200:
        raise ValueError(
            "Expected 200 repaired clinical cases."
        )

    # --------------------------------------------------------
    # BUILD KEYS FROM CLINICAL REPAIRS
    # --------------------------------------------------------

    invalid_keys = set()

    for case_id in repaired_case_ids:
        for language in LANGUAGES:
            invalid_keys.add(
                (
                    case_id,
                    language,
                )
            )

    print(
        f"Keys from clinical repairs: "
        f"{len(invalid_keys)}"
    )

    # --------------------------------------------------------
    # ALSO INVALIDATE PREVIOUS QC FAILURES
    # --------------------------------------------------------

    old_failures = 0

    for result in semantic_results:

        semantic_pass = result.get(
            "semantic_passed",
            False,
        )

        language_pass = result.get(
            "language_passed",
            False,
        )

        if (
            not semantic_pass
            or not language_pass
        ):
            old_failures += 1

            invalid_keys.add(
                (
                    result["case_id"],
                    result["language"],
                )
            )

    print(
        f"Previous QC failures: "
        f"{old_failures}"
    )

    print(
        f"Unique results to invalidate: "
        f"{len(invalid_keys)}"
    )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    shutil.copy2(
        SEMANTIC_FILE,
        BACKUP_FILE,
    )

    print(
        f"\nBackup created:\n"
        f"{BACKUP_FILE}"
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    retained = []

    removed = []

    for result in semantic_results:

        key = (
            result["case_id"],
            result["language"],
        )

        if key in invalid_keys:
            removed.append(result)
        else:
            retained.append(result)

    print(
        f"\nResults removed: "
        f"{len(removed)}"
    )

    print(
        f"Results retained: "
        f"{len(retained)}"
    )

    if len(removed) != 1438:
        raise ValueError(
            "Expected exactly 1438 results "
            f"to invalidate, found {len(removed)}"
        )

    if len(retained) != 5562:
        raise ValueError(
            "Expected exactly 5562 retained results, "
            f"found {len(retained)}"
        )

    # --------------------------------------------------------
    # WRITE FILTERED RESULTS
    # --------------------------------------------------------

    with SEMANTIC_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in retained:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nSemantic validation cache updated.")
    print(
        f"Remaining cached results: "
        f"{len(retained)}"
    )

    print("\nReady to rerun validate_semantics.py")


if __name__ == "__main__":
    main()