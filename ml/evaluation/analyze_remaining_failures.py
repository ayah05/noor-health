import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_predictions import (
    parse_prediction,
    normalize_string,
    normalize_duration,
    normalize_information_group,
)


# ============================================================
# HELPERS
# ============================================================

def build_symptom_map(
    data: dict,
) -> dict:

    symptoms = data.get(
        "symptoms",
        []
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return {}

    result = {}

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            continue

        name = normalize_string(
            symptom.get("name")
        )

        if not isinstance(
            name,
            str,
        ):
            continue

        # Keep duplicates detectable.
        if name in result:

            result[name][
                "_duplicate"
            ] = True

            continue

        result[name] = {
            "status":
                normalize_string(
                    symptom.get(
                        "status"
                    )
                ),

            "duration":
                normalize_duration(
                    symptom.get(
                        "duration"
                    )
                ),

            "_duplicate":
                False,
        }

    return result


# ============================================================
# ERROR ANALYSIS
# ============================================================

def analyze_failure(
    target: dict,
    prediction_raw: Any,
) -> dict:

    prediction = parse_prediction(
        prediction_raw
    )

    errors = []

    details = {}

    # --------------------------------------------------------
    # INVALID JSON
    # --------------------------------------------------------

    if prediction is None:

        return {
            "errors": [
                "invalid_json"
            ],
            "details": {},
        }

    # --------------------------------------------------------
    # CHIEF COMPLAINT
    # --------------------------------------------------------

    target_chief = normalize_string(
        target.get(
            "chief_complaint"
        )
    )

    predicted_chief = normalize_string(
        prediction.get(
            "chief_complaint"
        )
    )

    if (
        target_chief
        != predicted_chief
    ):

        errors.append(
            "wrong_chief_complaint"
        )

        details[
            "chief_complaint"
        ] = {
            "expected":
                target_chief,

            "predicted":
                predicted_chief,
        }

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    target_symptoms = (
        build_symptom_map(
            target
        )
    )

    predicted_symptoms = (
        build_symptom_map(
            prediction
        )
    )

    target_names = set(
        target_symptoms.keys()
    )

    predicted_names = set(
        predicted_symptoms.keys()
    )

    # Missing symptoms
    missing_symptoms = sorted(
        target_names
        - predicted_names
    )

    if missing_symptoms:

        errors.append(
            "missing_symptom"
        )

        details[
            "missing_symptoms"
        ] = missing_symptoms

    # Extra symptoms
    extra_symptoms = sorted(
        predicted_names
        - target_names
    )

    if extra_symptoms:

        errors.append(
            "extra_symptom"
        )

        details[
            "extra_symptoms"
        ] = extra_symptoms

    # Duplicate symptoms
    duplicate_symptoms = sorted(
        name
        for name, symptom
        in predicted_symptoms.items()
        if symptom.get(
            "_duplicate"
        )
    )

    if duplicate_symptoms:

        errors.append(
            "duplicate_symptom"
        )

        details[
            "duplicate_symptoms"
        ] = duplicate_symptoms

    # --------------------------------------------------------
    # STATUS + DURATION
    # --------------------------------------------------------

    common_symptoms = (
        target_names
        & predicted_names
    )

    wrong_status = []
    wrong_duration = []

    for name in sorted(
        common_symptoms
    ):

        expected = (
            target_symptoms[
                name
            ]
        )

        predicted = (
            predicted_symptoms[
                name
            ]
        )

        if (
            expected[
                "status"
            ]
            != predicted[
                "status"
            ]
        ):

            wrong_status.append(
                {
                    "symptom":
                        name,

                    "expected":
                        expected[
                            "status"
                        ],

                    "predicted":
                        predicted[
                            "status"
                        ],
                }
            )

        if (
            expected[
                "duration"
            ]
            != predicted[
                "duration"
            ]
        ):

            wrong_duration.append(
                {
                    "symptom":
                        name,

                    "expected":
                        expected[
                            "duration"
                        ],

                    "predicted":
                        predicted[
                            "duration"
                        ],
                }
            )

    if wrong_status:

        errors.append(
            "wrong_symptom_status"
        )

        details[
            "wrong_status"
        ] = wrong_status

    if wrong_duration:

        errors.append(
            "wrong_duration"
        )

        details[
            "wrong_duration"
        ] = wrong_duration

    # --------------------------------------------------------
    # MEDICATIONS
    # --------------------------------------------------------

    target_medications = (
        normalize_information_group(
            target.get(
                "medications"
            )
        )
    )

    predicted_medications = (
        normalize_information_group(
            prediction.get(
                "medications"
            )
        )
    )

    if (
        target_medications
        != predicted_medications
    ):

        errors.append(
            "medication_error"
        )

        details[
            "medications"
        ] = {
            "expected":
                target_medications,

            "predicted":
                predicted_medications,
        }

    # --------------------------------------------------------
    # ALLERGIES
    # --------------------------------------------------------

    target_allergies = (
        normalize_information_group(
            target.get(
                "allergies"
            )
        )
    )

    predicted_allergies = (
        normalize_information_group(
            prediction.get(
                "allergies"
            )
        )
    )

    if (
        target_allergies
        != predicted_allergies
    ):

        errors.append(
            "allergy_error"
        )

        details[
            "allergies"
        ] = {
            "expected":
                target_allergies,

            "predicted":
                predicted_allergies,
        }

    return {
        "errors":
            errors,

        "details":
            details,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{path}"
        )

    records = []

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

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSONL "
                    f"at line "
                    f"{line_number}"
                ) from exc

            records.append(
                record
            )

    return records


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(
    input_file: Path,
    output_dir: Path,
):

    records = load_jsonl(
        input_file
    )

    print(
        "=" * 76
    )

    print(
        "NOOR HEALTH - REMAINING LORA FAILURES"
    )

    print(
        "=" * 76
    )

    print(
        f"\nSamples: "
        f"{len(records)}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    language_counts = Counter()

    error_counts = Counter()

    errors_by_language = defaultdict(
        Counter
    )

    error_combinations = Counter()

    cases_to_languages = defaultdict(
        list
    )

    analyzed_records = []

    # ========================================================
    # ANALYZE EACH SAMPLE
    # ========================================================

    for record in records:

        case_id = record.get(
            "case_id"
        )

        language = record.get(
            "language",
            "unknown",
        )

        target = record.get(
            "target",
            {}
        )

        lora_prediction = (
            record.get(
                "lora_prediction"
            )
        )

        result = analyze_failure(
            target=target,
            prediction_raw=(
                lora_prediction
            ),
        )

        errors = result[
            "errors"
        ]

        details = result[
            "details"
        ]

        language_counts[
            language
        ] += 1

        cases_to_languages[
            case_id
        ].append(
            language
        )

        for error in errors:

            error_counts[
                error
            ] += 1

            errors_by_language[
                language
            ][
                error
            ] += 1

        combination = (
            " + ".join(
                sorted(errors)
            )
            if errors
            else "unknown"
        )

        error_combinations[
            combination
        ] += 1

        analyzed_records.append(
            {
                "case_id":
                    case_id,

                "language":
                    language,

                "utterance":
                    record.get(
                        "utterance"
                    ),

                "target":
                    target,

                "lora_prediction":
                    lora_prediction,

                "errors":
                    errors,

                "details":
                    details,
            }
        )

    # ========================================================
    # LANGUAGE DISTRIBUTION
    # ========================================================

    print(
        "\nFAILURES BY LANGUAGE"
    )

    print(
        "-" * 76
    )

    for language, count in (
        language_counts
        .most_common()
    ):

        percentage = (
            count
            / len(records)
            * 100
        )

        print(
            f"{language:<12}"
            f"{count:>5}"
            f"{percentage:>10.2f}%"
        )

    # ========================================================
    # ERROR TYPES
    # ========================================================

    print(
        "\nERROR TYPES"
    )

    print(
        "-" * 76
    )

    for error, count in (
        error_counts
        .most_common()
    ):

        percentage = (
            count
            / len(records)
            * 100
        )

        print(
            f"{error:<30}"
            f"{count:>5}"
            f"{percentage:>10.2f}%"
        )

    # ========================================================
    # ERROR TYPES BY LANGUAGE
    # ========================================================

    print(
        "\nERROR TYPES BY LANGUAGE"
    )

    print(
        "=" * 76
    )

    for language in sorted(
        language_counts
    ):

        print(
            f"\n{language} "
            f"({language_counts[language]} failures)"
        )

        print(
            "-" * 76
        )

        for error, count in (
            errors_by_language[
                language
            ].most_common()
        ):

            percentage = (
                count
                / language_counts[
                    language
                ]
                * 100
            )

            print(
                f"{error:<30}"
                f"{count:>5}"
                f"{percentage:>10.2f}%"
            )

    # ========================================================
    # ERROR COMBINATIONS
    # ========================================================

    print(
        "\nMOST COMMON ERROR COMBINATIONS"
    )

    print(
        "-" * 76
    )

    for combination, count in (
        error_combinations
        .most_common(15)
    ):

        print(
            f"{count:>4}  "
            f"{combination}"
        )

    # ========================================================
    # CROSS-LANGUAGE CASE ANALYSIS
    # ========================================================

    multi_language_cases = {
        case_id:
            sorted(languages)

        for case_id, languages
        in cases_to_languages.items()

        if len(languages) > 1
    }

    single_language_cases = {
        case_id:
            languages

        for case_id, languages
        in cases_to_languages.items()

        if len(languages) == 1
    }

    print(
        "\nCROSS-LANGUAGE CASE ANALYSIS"
    )

    print(
        "-" * 76
    )

    print(
        f"Unique failing cases: "
        f"{len(cases_to_languages)}"
    )

    print(
        f"Cases failing in one language: "
        f"{len(single_language_cases)}"
    )

    print(
        f"Cases failing in multiple languages: "
        f"{len(multi_language_cases)}"
    )

    # --------------------------------------------------------
    # Distribution: number of failing languages per case
    # --------------------------------------------------------

    failing_language_count = Counter(
        len(languages)
        for languages
        in cases_to_languages.values()
    )

    print(
        "\nFailing languages per case:"
    )

    for number, count in sorted(
        failing_language_count.items()
    ):

        print(
            f"  {number} language(s): "
            f"{count} cases"
        )

    # --------------------------------------------------------
    # Language combinations
    # --------------------------------------------------------

    language_combinations = Counter(
        tuple(
            sorted(languages)
        )
        for languages
        in cases_to_languages.values()
    )

    print(
        "\nMost common failing language combinations:"
    )

    for combination, count in (
        language_combinations
        .most_common(15)
    ):

        languages = ", ".join(
            combination
        )

        print(
            f"{count:>4}  "
            f"{languages}"
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzed_file = (
        output_dir
        / "remaining_failures_analyzed.jsonl"
    )

    summary_file = (
        output_dir
        / "remaining_failures_summary.json"
    )

    human_file = (
        output_dir
        / "remaining_failures_review.txt"
    )

    # --------------------------------------------------------
    # Detailed JSONL
    # --------------------------------------------------------

    with analyzed_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in analyzed_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "total_failures":
            len(records),

        "failures_by_language":
            dict(
                language_counts
            ),

        "error_types":
            dict(
                error_counts
            ),

        "error_types_by_language": {
            language:
                dict(counter)

            for language, counter
            in errors_by_language.items()
        },

        "error_combinations": {
            combination:
                count

            for combination, count
            in error_combinations.items()
        },

        "unique_failing_cases":
            len(
                cases_to_languages
            ),

        "single_language_cases":
            len(
                single_language_cases
            ),

        "multi_language_cases":
            len(
                multi_language_cases
            ),

        "failing_languages_per_case": {
            str(number):
                count

            for number, count
            in failing_language_count.items()
        },

        "language_combinations": {
            " + ".join(
                combination
            ):
                count

            for combination, count
            in language_combinations.items()
        },
    }

    summary_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Human-readable review
    # --------------------------------------------------------

    with human_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for index, record in enumerate(
            analyzed_records,
            start=1,
        ):

            file.write(
                "=" * 90
                + "\n"
            )

            file.write(
                f"FAILURE {index}\n"
            )

            file.write(
                f"Case: "
                f"{record['case_id']}\n"
            )

            file.write(
                f"Language: "
                f"{record['language']}\n"
            )

            file.write(
                "Errors: "
                + ", ".join(
                    record["errors"]
                )
                + "\n\n"
            )

            file.write(
                "PATIENT INPUT\n"
            )

            file.write(
                str(
                    record[
                        "utterance"
                    ]
                )
                + "\n\n"
            )

            file.write(
                "EXPECTED\n"
            )

            file.write(
                json.dumps(
                    record[
                        "target"
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n\n"
            )

            file.write(
                "LORA PREDICTION\n"
            )

            prediction = (
                parse_prediction(
                    record[
                        "lora_prediction"
                    ]
                )
            )

            if prediction is not None:

                file.write(
                    json.dumps(
                        prediction,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

            else:

                file.write(
                    str(
                        record[
                            "lora_prediction"
                        ]
                    )
                )

            file.write(
                "\n\nERROR DETAILS\n"
            )

            file.write(
                json.dumps(
                    record[
                        "details"
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )

            file.write(
                "\n\n"
            )

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        f"  {summary_file}"
    )

    print(
        f"  {analyzed_file}"
    )

    print(
        f"  {human_file}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    # --------------------------------------------------------
    # PROJECT PATHS
    # --------------------------------------------------------

    # Current file:
    #
    # noor-health/
    # └── ml/
    #     └── evaluation/
    #         └── analyze_remaining_failures.py
    #
    # parents[0] -> evaluation
    # parents[1] -> ml
    # parents[2] -> noor-health

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    default_input = (
        project_root
        / "results"
        / "synthetic_test"
        / "comparison"
        / "base_vs_lora_both_wrong.jsonl"
    )

    default_output_dir = (
        project_root
        / "results"
        / "synthetic_test"
        / "failure_analysis"
    )

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Analyze Noor Health samples "
            "where both Base and LoRA fail."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            default_input
        ),
        help=(
            "Path to both-wrong JSONL."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            default_output_dir
        ),
        help=(
            "Directory for failure "
            "analysis results."
        ),
    )

    args = parser.parse_args()

    input_file = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    print(
        f"Project root:\n"
        f"{project_root}"
    )

    print(
        f"\nInput file:\n"
        f"{input_file}"
    )

    print(
        f"\nOutput directory:\n"
        f"{output_dir}"
    )

    print()

    analyze(
        input_file=input_file,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()