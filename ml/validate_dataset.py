import json
from collections import Counter, defaultdict

from config import SYNTHETIC_DATA_DIR


# ============================================================
# CONFIG
# ============================================================

CLINICAL_CASES_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

MULTILINGUAL_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2.jsonl"
)

EXPECTED_LANGUAGES = {
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
}

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

        # Further validation would be unreliable.
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

        for index, symptom in enumerate(symptoms):

            location = (
                f"{sample_name}: "
                f"symptoms[{index}]"
            )

            if not isinstance(symptom, dict):
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

            # Absent / uncertain symptoms should not
            # receive invented durations.
            if (
                status in {"absent", "uncertain"}
                and duration is not None
            ):
                errors.append(
                    f"{location}: {status} symptom "
                    f"must have duration=null"
                )

            if duration is not None:

                if not isinstance(duration, dict):
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
                            f"duration value {value!r}"
                        )

                    if unit not in {
                        "hours",
                        "days",
                        "weeks",
                    }:
                        errors.append(
                            f"{location}: invalid "
                            f"duration unit {unit!r}"
                        )

        if (
            len(symptom_names)
            != len(set(symptom_names))
        ):
            errors.append(
                f"{sample_name}: duplicate symptoms"
            )

        # Chief complaint must exist and be present.
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
                f"must occur exactly once in symptoms"
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
            status in {"unknown", "none"}
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
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - DATASET VALIDATION"
    )
    print("=" * 60)

    clinical_cases = load_jsonl(
        CLINICAL_CASES_FILE
    )

    samples = load_jsonl(
        MULTILINGUAL_FILE
    )

    clinical_by_id = {
        case["case_id"]: case
        for case in clinical_cases
    }

    errors = []

    # --------------------------------------------------------
    # DUPLICATES + LANGUAGE DISTRIBUTION
    # --------------------------------------------------------

    seen_keys = set()

    languages_by_case = defaultdict(set)

    language_counts = Counter()

    for sample in samples:

        case_id = sample.get("case_id")
        language = sample.get("language")

        sample_name = (
            f"{case_id}/{language}"
        )

        key = (
            case_id,
            language,
        )

        if key in seen_keys:
            errors.append(
                f"{sample_name}: duplicate sample"
            )

        seen_keys.add(key)

        languages_by_case[
            case_id
        ].add(language)

        language_counts[
            language
        ] += 1

        # ----------------------------------------------------
        # BASIC SAMPLE FIELDS
        # ----------------------------------------------------

        required_sample_fields = {
            "case_id",
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
                f"{sample_name}: missing sample "
                f"fields "
                f"{sorted(missing_sample_fields)}"
            )
            continue

        if language not in EXPECTED_LANGUAGES:
            errors.append(
                f"{sample_name}: unexpected "
                f"language"
            )

        utterance = sample["utterance"]

        if (
            not isinstance(utterance, str)
            or not utterance.strip()
        ):
            errors.append(
                f"{sample_name}: empty utterance"
            )

        if sample["source"] != "synthetic":
            errors.append(
                f"{sample_name}: unexpected "
                f"source {sample['source']!r}"
            )

        if sample["schema_version"] != "v2":
            errors.append(
                f"{sample_name}: unexpected "
                f"schema version "
                f"{sample['schema_version']!r}"
            )

        # ----------------------------------------------------
        # SOURCE CASE EXISTS
        # ----------------------------------------------------

        if case_id not in clinical_by_id:
            errors.append(
                f"{sample_name}: unknown case_id"
            )
            continue

        # ----------------------------------------------------
        # TARGET SCHEMA
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
                clinical_by_id[case_id]
            )
        )

        if target != expected_target:
            errors.append(
                f"{sample_name}: target differs "
                f"from source clinical case"
            )

        # ----------------------------------------------------
        # SCRIPT CHECKS
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

    # --------------------------------------------------------
    # LANGUAGE COMPLETENESS PER CASE
    # --------------------------------------------------------

    generated_case_ids = set(
        languages_by_case.keys()
    )

    for case_id in sorted(
        generated_case_ids
    ):

        actual_languages = (
            languages_by_case[case_id]
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
                f"{case_id}: missing languages "
                f"{sorted(missing_languages)}"
            )

        if extra_languages:
            errors.append(
                f"{case_id}: unexpected languages "
                f"{sorted(extra_languages)}"
            )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        f"\nClinical cases available: "
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
        "\nNo structural dataset "
        "problems detected."
    )

    print(
        "\nNOTE:"
    )

    print(
        "This validator checks structure and "
        "target consistency."
    )

    print(
        "It does not prove that every generated "
        "utterance semantically matches its target."
    )


if __name__ == "__main__":
    main()