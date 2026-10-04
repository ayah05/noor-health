import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from config import SYNTHETIC_DATA_DIR


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

BACKUP_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2_before_repair.jsonl"
)

REPORT_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2_repair_report.json"
)


# ============================================================
# LOAD / SAVE
# ============================================================

def load_jsonl(path: Path) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

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
# DETERMINISTIC DURATION
# ============================================================

def deterministic_duration(
    case_id: str,
) -> dict:
    """
    Generate a deterministic duration from case_id.

    The same case_id will always receive the same duration.
    """

    digest = hashlib.sha256(
        case_id.encode("utf-8")
    ).digest()

    unit_index = (
        digest[0] % 3
    )

    units = [
        "hours",
        "days",
        "weeks",
    ]

    unit = units[
        unit_index
    ]

    if unit == "hours":

        value = (
            digest[1] % 12
        ) + 1

    elif unit == "days":

        value = (
            digest[1] % 7
        ) + 1

    else:

        value = (
            digest[1] % 3
        ) + 1

    return {
        "value": value,
        "unit": unit,
    }


# ============================================================
# MISSING INFORMATION
# ============================================================

def determine_missing_information(
    case: dict,
) -> list[str]:

    missing = []

    chief_complaint = (
        case["chief_complaint"]
    )

    chief = next(
        symptom
        for symptom in case["symptoms"]
        if symptom["name"]
        == chief_complaint
    )

    if chief["duration"] is None:

        missing.append(
            "duration"
        )

    if (
        case["medications"]["status"]
        == "unknown"
    ):

        missing.append(
            "medications"
        )

    if (
        case["allergies"]["status"]
        == "unknown"
    ):

        missing.append(
            "allergies"
        )

    return missing


# ============================================================
# REPAIR
# ============================================================

def repair_case(
    case: dict,
) -> bool:
    """
    Repair the accidental correlation:

    missing_medications / missing_allergies
        must NOT imply missing duration.

    Returns True if the case was modified.
    """

    pattern = case[
        "case_pattern"
    ]

    if pattern not in {
        "missing_medications",
        "missing_allergies",
    }:
        return False

    chief_complaint = (
        case["chief_complaint"]
    )

    chief = next(
        symptom
        for symptom in case["symptoms"]
        if symptom["name"]
        == chief_complaint
    )

    # Only repair the actual confounding bug.
    if chief["duration"] is None:

        chief["duration"] = (
            deterministic_duration(
                case["case_id"]
            )
        )

        case[
            "missing_information"
        ] = (
            determine_missing_information(
                case
            )
        )

        return True

    return False


# ============================================================
# VALIDATION
# ============================================================

def validate_repair(
    cases: list[dict],
):

    if len(cases) != 1000:

        raise ValueError(
            f"Expected 1000 cases, "
            f"found {len(cases)}"
        )

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    if (
        len(case_ids)
        != len(set(case_ids))
    ):

        raise ValueError(
            "Duplicate case IDs detected"
        )

    pattern_counts = Counter(
        case["case_pattern"]
        for case in cases
    )

    if (
        pattern_counts[
            "missing_medications"
        ]
        != 100
    ):
        raise ValueError(
            "Expected exactly 100 "
            "missing_medications cases"
        )

    if (
        pattern_counts[
            "missing_allergies"
        ]
        != 100
    ):
        raise ValueError(
            "Expected exactly 100 "
            "missing_allergies cases"
        )

    for case in cases:

        pattern = case[
            "case_pattern"
        ]

        if pattern not in {
            "missing_medications",
            "missing_allergies",
        }:
            continue

        chief = next(
            symptom
            for symptom in case[
                "symptoms"
            ]
            if symptom["name"]
            == case["chief_complaint"]
        )

        if chief["duration"] is None:

            raise ValueError(
                f"{case['case_id']}: "
                f"{pattern} still has "
                "missing chief duration"
            )

        expected_missing = (
            determine_missing_information(
                case
            )
        )

        if (
            case["missing_information"]
            != expected_missing
        ):

            raise ValueError(
                f"{case['case_id']}: "
                "incorrect "
                "missing_information "
                "after repair"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "CLINICAL CASE REPAIR"
    )
    print("=" * 60)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{INPUT_FILE}"
        )

    cases = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nCases loaded: "
        f"{len(cases)}"
    )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    if not BACKUP_FILE.exists():

        shutil.copy2(
            INPUT_FILE,
            BACKUP_FILE,
        )

        print(
            f"Backup created:\n"
            f"{BACKUP_FILE}"
        )

    else:

        print(
            "\nBackup already exists."
        )

    # --------------------------------------------------------
    # REPAIR
    # --------------------------------------------------------

    changed_cases = []

    changed_by_pattern = Counter()

    for case in cases:

        changed = repair_case(
            case
        )

        if changed:

            changed_cases.append(
                case["case_id"]
            )

            changed_by_pattern[
                case["case_pattern"]
            ] += 1

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_repair(
        cases
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_jsonl(
        INPUT_FILE,
        cases,
    )

    report = {
        "total_cases": len(cases),
        "changed_cases": len(
            changed_cases
        ),
        "changed_by_pattern": dict(
            changed_by_pattern
        ),
        "changed_case_ids":
            changed_cases,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)

    print(
        f"\nChanged cases: "
        f"{len(changed_cases)}"
    )

    print(
        "\nChanged by pattern:"
    )

    for pattern in [
        "missing_medications",
        "missing_allergies",
    ]:

        print(
            f"  {pattern:<22} "
            f"{changed_by_pattern[pattern]}"
        )

    print(
        "\nValidation: PASSED"
    )

    print(
        f"\nUpdated dataset:\n"
        f"{INPUT_FILE}"
    )

    print(
        f"\nRepair report:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()