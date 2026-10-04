import json
from collections import Counter, defaultdict

from config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# CONFIG
# ============================================================

CLINICAL_CASES_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

MULTILINGUAL_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2_full.jsonl"
)

SPLIT_FILES = {
    "train": (
        PROCESSED_DATA_DIR
        / "train_cases.jsonl"
    ),
    "validation": (
        PROCESSED_DATA_DIR
        / "validation_cases.jsonl"
    ),
    "test": (
        PROCESSED_DATA_DIR
        / "test_cases.jsonl"
    ),
}


EXPECTED_LANGUAGES = {
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
}


EXPECTED_SPLIT_CASE_COUNTS = {
    "train": 800,
    "validation": 100,
    "test": 100,
}


EXPECTED_SPLIT_SAMPLE_COUNTS = {
    "train": 5600,
    "validation": 700,
    "test": 700,
}


EXPECTED_TOTAL_CASES = 1000
EXPECTED_TOTAL_SAMPLES = 7000


VALID_SYMPTOM_STATUSES = {
    "present",
    "absent",
    "uncertain",
}


VALID_INFO_STATUSES = {
    "unknown",
    "none",
    "reported",
}


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

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
                    f"Invalid JSON in {path.name}, "
                    f"line {line_number}: {error}"
                ) from error

    return rows


# ============================================================
# EXPECTED TARGET
# ============================================================

def build_expected_target(
    clinical_case: dict,
) -> dict:

    return {
        "chief_complaint":
            clinical_case["chief_complaint"],

        "symptoms":
            clinical_case["symptoms"],

        "medications":
            clinical_case["medications"],

        "allergies":
            clinical_case["allergies"],

        "missing_information":
            clinical_case["missing_information"],
    }


# ============================================================
# VALIDATE TARGET SCHEMA
# ============================================================

def validate_target(
    target: dict,
    sample_name: str,
) -> list[str]:

    errors = []

    required_fields = {
        "chief_complaint",
        "symptoms",
        "medications",
        "allergies",
        "missing_information",
    }

    missing_fields = (
        required_fields
        - set(target.keys())
    )

    if missing_fields:

        errors.append(
            f"{sample_name}: target missing fields: "
            f"{sorted(missing_fields)}"
        )

        return errors

    # --------------------------------------------------------
    # CHIEF COMPLAINT
    # --------------------------------------------------------

    if not isinstance(
        target["chief_complaint"],
        str,
    ):

        errors.append(
            f"{sample_name}: chief_complaint "
            f"must be a string"
        )

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    symptoms = target["symptoms"]

    if not isinstance(symptoms, list):

        errors.append(
            f"{sample_name}: symptoms must be a list"
        )

    else:

        symptom_names = []

        for index, symptom in enumerate(
            symptoms
        ):

            location = (
                f"{sample_name}: "
                f"symptoms[{index}]"
            )

            if not isinstance(
                symptom,
                dict,
            ):

                errors.append(
                    f"{location} must be an object"
                )

                continue

            name = symptom.get("name")
            status = symptom.get("status")
            duration = symptom.get("duration")

            if not isinstance(name, str):

                errors.append(
                    f"{location}: invalid name"
                )

            else:

                symptom_names.append(name)

            if status not in VALID_SYMPTOM_STATUSES:

                errors.append(
                    f"{location}: invalid status "
                    f"{status!r}"
                )

            if (
                status in {
                    "absent",
                    "uncertain",
                }
                and duration is not None
            ):

                errors.append(
                    f"{location}: {status} symptom "
                    f"must have duration=null"
                )

            if duration is not None:

                if not isinstance(
                    duration,
                    dict,
                ):

                    errors.append(
                        f"{location}: duration "
                        f"must be object or null"
                    )

                else:

                    value = duration.get("value")
                    unit = duration.get("unit")

                    if (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value <= 0
                    ):

                        errors.append(
                            f"{location}: invalid "
                            f"duration value "
                            f"{value!r}"
                        )

                    if unit not in {
                        "hours",
                        "days",
                        "weeks",
                    }:

                        errors.append(
                            f"{location}: invalid "
                            f"duration unit "
                            f"{unit!r}"
                        )

        if (
            len(symptom_names)
            != len(set(symptom_names))
        ):

            errors.append(
                f"{sample_name}: duplicate symptoms"
            )

        chief_complaint = (
            target["chief_complaint"]
        )

        chief_matches = [
            symptom
            for symptom in symptoms
            if isinstance(symptom, dict)
            and symptom.get("name")
            == chief_complaint
        ]

        if len(chief_matches) != 1:

            errors.append(
                f"{sample_name}: chief complaint "
                f"must occur exactly once "
                f"in symptoms"
            )

        elif (
            chief_matches[0].get("status")
            != "present"
        ):

            errors.append(
                f"{sample_name}: chief complaint "
                f"must have status='present'"
            )

    # --------------------------------------------------------
    # MEDICATIONS / ALLERGIES
    # --------------------------------------------------------

    for field in [
        "medications",
        "allergies",
    ]:

        info = target[field]

        if not isinstance(info, dict):

            errors.append(
                f"{sample_name}: {field} "
                f"must be an object"
            )

            continue

        status = info.get("status")
        items = info.get("items")

        if status not in VALID_INFO_STATUSES:

            errors.append(
                f"{sample_name}: {field} has "
                f"invalid status {status!r}"
            )

        if not isinstance(items, list):

            errors.append(
                f"{sample_name}: {field}.items "
                f"must be a list"
            )

            continue

        if (
            status in {
                "unknown",
                "none",
            }
            and items
        ):

            errors.append(
                f"{sample_name}: {field} with "
                f"status={status!r} must have "
                f"empty items"
            )

        if (
            status == "reported"
            and not items
        ):

            errors.append(
                f"{sample_name}: {field} with "
                f"status='reported' must contain "
                f"at least one item"
            )

    # --------------------------------------------------------
    # MISSING INFORMATION
    # --------------------------------------------------------

    missing = target[
        "missing_information"
    ]

    if not isinstance(missing, list):

        errors.append(
            f"{sample_name}: "
            f"missing_information must be a list"
        )

    else:

        allowed_missing = {
            "duration",
            "medications",
            "allergies",
        }

        invalid_missing = (
            set(missing)
            - allowed_missing
        )

        if invalid_missing:

            errors.append(
                f"{sample_name}: invalid "
                f"missing_information values: "
                f"{sorted(invalid_missing)}"
            )

        expected_missing = set()

        medications = target["medications"]
        allergies = target["allergies"]

        if (
            isinstance(medications, dict)
            and medications.get("status")
            == "unknown"
        ):

            expected_missing.add(
                "medications"
            )

        if (
            isinstance(allergies, dict)
            and allergies.get("status")
            == "unknown"
        ):

            expected_missing.add(
                "allergies"
            )

        if isinstance(symptoms, list):

            chief_complaint = (
                target["chief_complaint"]
            )

            for symptom in symptoms:

                if (
                    isinstance(symptom, dict)
                    and symptom.get("name")
                    == chief_complaint
                    and symptom.get("duration")
                    is None
                ):

                    expected_missing.add(
                        "duration"
                    )

        if set(missing) != expected_missing:

            errors.append(
                f"{sample_name}: "
                f"missing_information mismatch. "
                f"Expected "
                f"{sorted(expected_missing)}, "
                f"got {sorted(missing)}"
            )

    return errors


# ============================================================
# LOAD EXPECTED SPLITS
# ============================================================

def load_expected_splits():

    expected_split_by_case = {}

    errors = []

    for split, path in SPLIT_FILES.items():

        cases = load_jsonl(path)

        expected_count = (
            EXPECTED_SPLIT_CASE_COUNTS[
                split
            ]
        )

        if len(cases) != expected_count:

            errors.append(
                f"{split}: expected "
                f"{expected_count} cases, "
                f"found {len(cases)}"
            )

        for clinical_case in cases:

            case_id = clinical_case.get(
                "case_id"
            )

            if case_id in expected_split_by_case:

                errors.append(
                    f"{case_id}: appears in "
                    f"multiple split files"
                )

            expected_split_by_case[
                case_id
            ] = split

    return (
        expected_split_by_case,
        errors,
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - FINAL DATASET VALIDATION"
    )
    print("=" * 60)

    clinical_cases = load_jsonl(
        CLINICAL_CASES_FILE
    )

    samples = load_jsonl(
        MULTILINGUAL_FILE
    )

    (
        expected_split_by_case,
        errors,
    ) = load_expected_splits()

    clinical_by_id = {
        case["case_id"]: case
        for case in clinical_cases
    }

    # --------------------------------------------------------
    # BASIC GLOBAL COUNTS
    # --------------------------------------------------------

    if (
        len(clinical_cases)
        != EXPECTED_TOTAL_CASES
    ):

        errors.append(
            f"Expected "
            f"{EXPECTED_TOTAL_CASES} "
            f"clinical cases, found "
            f"{len(clinical_cases)}"
        )

    if (
        len(expected_split_by_case)
        != EXPECTED_TOTAL_CASES
    ):

        errors.append(
            f"Expected "
            f"{EXPECTED_TOTAL_CASES} "
            f"unique split cases, found "
            f"{len(expected_split_by_case)}"
        )

    if (
        len(samples)
        != EXPECTED_TOTAL_SAMPLES
    ):

        errors.append(
            f"Expected "
            f"{EXPECTED_TOTAL_SAMPLES} "
            f"multilingual samples, found "
            f"{len(samples)}"
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    seen_keys = set()

    languages_by_case = defaultdict(set)
    splits_by_case = defaultdict(set)

    language_counts = Counter()
    split_sample_counts = Counter()

    # --------------------------------------------------------
    # VALIDATE EACH SAMPLE
    # --------------------------------------------------------

    for sample in samples:

        case_id = sample.get("case_id")
        language = sample.get("language")
        split = sample.get("split")

        sample_name = (
            f"{case_id}/"
            f"{split}/"
            f"{language}"
        )

        # ----------------------------------------------------
        # REQUIRED FIELDS
        # ----------------------------------------------------

        required_sample_fields = {
            "case_id",
            "split",
            "language",
            "utterance",
            "target",
            "source",
            "schema_version",
        }

        missing_sample_fields = (
            required_sample_fields
            - set(sample.keys())
        )

        if missing_sample_fields:

            errors.append(
                f"{sample_name}: missing "
                f"sample fields "
                f"{sorted(missing_sample_fields)}"
            )

            continue

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        key = (
            case_id,
            split,
            language,
        )

        if key in seen_keys:

            errors.append(
                f"{sample_name}: "
                f"duplicate sample"
            )

        seen_keys.add(key)

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        languages_by_case[
            case_id
        ].add(language)

        splits_by_case[
            case_id
        ].add(split)

        language_counts[
            language
        ] += 1

        split_sample_counts[
            split
        ] += 1

        # ----------------------------------------------------
        # LANGUAGE
        # ----------------------------------------------------

        if language not in EXPECTED_LANGUAGES:

            errors.append(
                f"{sample_name}: "
                f"unexpected language"
            )

        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        if split not in SPLIT_FILES:

            errors.append(
                f"{sample_name}: "
                f"unexpected split"
            )

        expected_split = (
            expected_split_by_case.get(
                case_id
            )
        )

        if expected_split is None:

            errors.append(
                f"{sample_name}: case_id "
                f"does not exist in split files"
            )

        elif split != expected_split:

            errors.append(
                f"{sample_name}: wrong split. "
                f"Expected {expected_split!r}, "
                f"got {split!r}"
            )

        # ----------------------------------------------------
        # UTTERANCE
        # ----------------------------------------------------

        utterance = sample["utterance"]

        if (
            not isinstance(utterance, str)
            or not utterance.strip()
        ):

            errors.append(
                f"{sample_name}: "
                f"empty utterance"
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        if sample["source"] != "synthetic":

            errors.append(
                f"{sample_name}: unexpected "
                f"source "
                f"{sample['source']!r}"
            )

        if sample["schema_version"] != "v2":

            errors.append(
                f"{sample_name}: unexpected "
                f"schema version "
                f"{sample['schema_version']!r}"
            )

        # ----------------------------------------------------
        # SOURCE CASE
        # ----------------------------------------------------

        if case_id not in clinical_by_id:

            errors.append(
                f"{sample_name}: "
                f"unknown case_id"
            )

            continue

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = sample["target"]

        if not isinstance(target, dict):

            errors.append(
                f"{sample_name}: target "
                f"must be an object"
            )

            continue

        errors.extend(
            validate_target(
                target,
                sample_name,
            )
        )

        # ----------------------------------------------------
        # EXACT TARGET CONSISTENCY
        # ----------------------------------------------------

        expected_target = (
            build_expected_target(
                clinical_by_id[
                    case_id
                ]
            )
        )

        if target != expected_target:

            errors.append(
                f"{sample_name}: target "
                f"differs from source "
                f"clinical case"
            )

        # ----------------------------------------------------
        # ARABIC SCRIPT CHECK
        # ----------------------------------------------------

        if language == "ar_msa":

            western_digits = (
                set("0123456789")
                & set(utterance)
            )

            if western_digits:

                errors.append(
                    f"{sample_name}: Arabic "
                    f"contains Western digits "
                    f"{sorted(western_digits)}"
                )

    # ========================================================
    # CASE-LEVEL VALIDATION
    # ========================================================

    generated_case_ids = set(
        languages_by_case.keys()
    )

    if (
        len(generated_case_ids)
        != EXPECTED_TOTAL_CASES
    ):

        errors.append(
            f"Expected "
            f"{EXPECTED_TOTAL_CASES} "
            f"represented cases, found "
            f"{len(generated_case_ids)}"
        )

    # --------------------------------------------------------
    # EVERY CASE MUST HAVE EXACTLY 7 LANGUAGES
    # --------------------------------------------------------

    for case_id in sorted(
        clinical_by_id.keys()
    ):

        actual_languages = (
            languages_by_case.get(
                case_id,
                set(),
            )
        )

        missing_languages = (
            EXPECTED_LANGUAGES
            - actual_languages
        )

        extra_languages = (
            actual_languages
            - EXPECTED_LANGUAGES
        )

        if missing_languages:

            errors.append(
                f"{case_id}: "
                f"missing languages "
                f"{sorted(missing_languages)}"
            )

        if extra_languages:

            errors.append(
                f"{case_id}: "
                f"unexpected languages "
                f"{sorted(extra_languages)}"
            )

        if (
            len(actual_languages)
            != len(EXPECTED_LANGUAGES)
        ):

            errors.append(
                f"{case_id}: expected "
                f"{len(EXPECTED_LANGUAGES)} "
                f"languages, found "
                f"{len(actual_languages)}"
            )

    # --------------------------------------------------------
    # CASE MUST NEVER CROSS SPLITS
    # --------------------------------------------------------

    for (
        case_id,
        actual_splits,
    ) in splits_by_case.items():

        if len(actual_splits) != 1:

            errors.append(
                f"{case_id}: appears in "
                f"multiple generated splits: "
                f"{sorted(actual_splits)}"
            )

    # --------------------------------------------------------
    # SPLIT SAMPLE COUNTS
    # --------------------------------------------------------

    for (
        split,
        expected_count,
    ) in (
        EXPECTED_SPLIT_SAMPLE_COUNTS.items()
    ):

        actual_count = (
            split_sample_counts[
                split
            ]
        )

        if actual_count != expected_count:

            errors.append(
                f"{split}: expected "
                f"{expected_count} samples, "
                f"found {actual_count}"
            )

    # --------------------------------------------------------
    # LANGUAGE COUNTS
    # --------------------------------------------------------

    for language in sorted(
        EXPECTED_LANGUAGES
    ):

        actual_count = (
            language_counts[
                language
            ]
        )

        if (
            actual_count
            != EXPECTED_TOTAL_CASES
        ):

            errors.append(
                f"{language}: expected "
                f"{EXPECTED_TOTAL_CASES} "
                f"samples, found "
                f"{actual_count}"
            )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"\nClinical cases: "
        f"{len(clinical_cases)}"
    )

    print(
        f"Cases represented: "
        f"{len(generated_case_ids)}"
    )

    print(
        f"Multilingual samples: "
        f"{len(samples)}"
    )

    print(
        "\nSplit distribution:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:<12} "
            f"{split_sample_counts[split]}"
        )

    print(
        "\nLanguage distribution:"
    )

    for language in sorted(
        EXPECTED_LANGUAGES
    ):

        print(
            f"  {language:<8} "
            f"{language_counts[language]}"
        )

    print(
        "\nValidation:"
    )

    if errors:

        print(
            f"  FAILED - "
            f"{len(errors)} error(s)"
        )

        print(
            "\nErrors:"
        )

        for index, error in enumerate(
            errors,
            start=1,
        ):

            print(
                f"  {index}. {error}"
            )

        raise SystemExit(1)

    print(
        "  PASSED"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL STRUCTURAL DATASET CHECK PASSED"
    )

    print(
        "=" * 60
    )

    print(
        "\nExpected dataset confirmed:"
    )

    print(
        "  1000 clinical cases"
    )

    print(
        "  7000 multilingual samples"
    )

    print(
        "  5600 train samples"
    )

    print(
        "   700 validation samples"
    )

    print(
        "   700 test samples"
    )

    print(
        "  7 languages per case"
    )

    print(
        "  No duplicate samples"
    )

    print(
        "  No case leakage across splits"
    )

    print(
        "  Targets match source cases"
    )

    print(
        "\nNOTE:"
    )

    print(
        "This proves structural and target "
        "consistency."
    )

    print(
        "It does NOT prove that every generated "
        "utterance semantically matches its target."
    )


if __name__ == "__main__":
    main()