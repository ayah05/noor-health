import json
import random
from collections import Counter, defaultdict

from config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

TRAIN_CASES_FILE = (
    PROCESSED_DATA_DIR
    / "train_cases.jsonl"
)

VALIDATION_CASES_FILE = (
    PROCESSED_DATA_DIR
    / "validation_cases.jsonl"
)

TEST_CASES_FILE = (
    PROCESSED_DATA_DIR
    / "test_cases.jsonl"
)

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path) -> list[dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON in "
                    f"{path.name}, "
                    f"line {line_number}: "
                    f"{error}"
                ) from error

    return rows


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    rows: list[dict],
    path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    cases: list[dict],
):

    """
    Split cases by case_pattern.

    All cases remain intact.

    Later, all seven language variants of a case_id
    will inherit this same split.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    cases_by_pattern = defaultdict(
        list
    )

    for case in cases:

        pattern = case.get(
            "case_pattern"
        )

        if pattern is None:
            raise ValueError(
                f"{case.get('case_id')}: "
                "missing case_pattern"
            )

        cases_by_pattern[
            pattern
        ].append(case)

    train_cases = []
    validation_cases = []
    test_cases = []

    for pattern in sorted(
        cases_by_pattern
    ):

        pattern_cases = list(
            cases_by_pattern[
                pattern
            ]
        )

        rng.shuffle(
            pattern_cases
        )

        total = len(
            pattern_cases
        )

        train_count = round(
            total * TRAIN_RATIO
        )

        validation_count = round(
            total * VALIDATION_RATIO
        )

        # Let test absorb rounding differences.
        test_count = (
            total
            - train_count
            - validation_count
        )

        if test_count < 0:
            raise ValueError(
                f"Invalid split counts "
                f"for {pattern}"
            )

        train_end = train_count

        validation_end = (
            train_count
            + validation_count
        )

        train_cases.extend(
            pattern_cases[
                :train_end
            ]
        )

        validation_cases.extend(
            pattern_cases[
                train_end:
                validation_end
            ]
        )

        test_cases.extend(
            pattern_cases[
                validation_end:
            ]
        )

    # Shuffle within each final split so patterns
    # are not grouped together in the output files.
    rng.shuffle(
        train_cases
    )

    rng.shuffle(
        validation_cases
    )

    rng.shuffle(
        test_cases
    )

    return (
        train_cases,
        validation_cases,
        test_cases,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_splits(
    original_cases,
    train_cases,
    validation_cases,
    test_cases,
):

    original_ids = {
        case["case_id"]
        for case in original_cases
    }

    train_ids = {
        case["case_id"]
        for case in train_cases
    }

    validation_ids = {
        case["case_id"]
        for case in validation_cases
    }

    test_ids = {
        case["case_id"]
        for case in test_cases
    }

    # --------------------------------------------------------
    # DUPLICATE IDs INSIDE SPLITS
    # --------------------------------------------------------

    if (
        len(train_ids)
        != len(train_cases)
    ):
        raise ValueError(
            "Duplicate case_id in train split"
        )

    if (
        len(validation_ids)
        != len(validation_cases)
    ):
        raise ValueError(
            "Duplicate case_id in "
            "validation split"
        )

    if (
        len(test_ids)
        != len(test_cases)
    ):
        raise ValueError(
            "Duplicate case_id in test split"
        )

    # --------------------------------------------------------
    # NO LEAKAGE
    # --------------------------------------------------------

    if train_ids & validation_ids:
        raise ValueError(
            "Data leakage between "
            "train and validation"
        )

    if train_ids & test_ids:
        raise ValueError(
            "Data leakage between "
            "train and test"
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Data leakage between "
            "validation and test"
        )

    # --------------------------------------------------------
    # NOTHING LOST
    # --------------------------------------------------------

    combined_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    if combined_ids != original_ids:

        missing = (
            original_ids
            - combined_ids
        )

        extra = (
            combined_ids
            - original_ids
        )

        raise ValueError(
            "Split mismatch. "
            f"Missing={sorted(missing)}, "
            f"Extra={sorted(extra)}"
        )


# ============================================================
# REPORT
# ============================================================

def print_pattern_distribution(
    name,
    cases,
):

    counts = Counter(
        case["case_pattern"]
        for case in cases
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Cases: {len(cases)}"
    )

    for pattern in sorted(
        counts
    ):

        print(
            f"  {pattern:<22} "
            f"{counts[pattern]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "NOOR HEALTH - "
        "CLINICAL CASE SPLIT"
    )

    print("=" * 60)

    cases = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nLoaded: {len(cases)} cases"
    )

    (
        train_cases,
        validation_cases,
        test_cases,
    ) = stratified_split(
        cases
    )

    validate_splits(
        original_cases=cases,
        train_cases=train_cases,
        validation_cases=(
            validation_cases
        ),
        test_cases=test_cases,
    )

    save_jsonl(
        train_cases,
        TRAIN_CASES_FILE,
    )

    save_jsonl(
        validation_cases,
        VALIDATION_CASES_FILE,
    )

    save_jsonl(
        test_cases,
        TEST_CASES_FILE,
    )

    print_pattern_distribution(
        "Train",
        train_cases,
    )

    print_pattern_distribution(
        "Validation",
        validation_cases,
    )

    print_pattern_distribution(
        "Test",
        test_cases,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "SPLIT VALIDATION: PASSED"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTrain:      "
        f"{len(train_cases)}"
    )

    print(
        f"Validation: "
        f"{len(validation_cases)}"
    )

    print(
        f"Test:       "
        f"{len(test_cases)}"
    )

    print(
        f"Total:      "
        f"{len(train_cases) + len(validation_cases) + len(test_cases)}"
    )

    print(
        "\nFiles:"
    )

    print(
        f"  {TRAIN_CASES_FILE}"
    )

    print(
        f"  {VALIDATION_CASES_FILE}"
    )

    print(
        f"  {TEST_CASES_FILE}"
    )


if __name__ == "__main__":
    main()