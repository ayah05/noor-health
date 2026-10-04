import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    ROOT
    / "results"
    / "lora_predictions.jsonl"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "language_failure_analysis.txt"
)

JSON_OUTPUT_FILE = (
    ROOT
    / "results"
    / "language_failure_analysis.json"
)

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "sw": "Swahili",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_string(value):

    if not isinstance(value, str):
        return value

    return value.strip().lower()


def normalize_duration(duration):

    if duration is None:
        return None

    if not isinstance(duration, dict):
        return duration

    return {
        "value": duration.get("value"),
        "unit": normalize_string(
            duration.get("unit")
        ),
    }


def normalize_information_group(group):

    if not isinstance(group, dict):
        return group

    items = group.get("items", [])

    if not isinstance(items, list):
        return group

    normalized_items = []

    for item in items:

        if isinstance(item, str):
            normalized_items.append(
                normalize_string(item)
            )
        else:
            normalized_items.append(item)

    try:
        normalized_items = sorted(
            normalized_items
        )
    except TypeError:
        pass

    return {
        "status": normalize_string(
            group.get("status")
        ),
        "items": normalized_items,
    }


def parse_prediction(prediction):

    if isinstance(prediction, dict):
        return prediction

    if not isinstance(prediction, str):
        return None

    text = prediction.strip()

    # --------------------------------------------------------
    # Remove Markdown fences
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Try extracting JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:

            parsed = json.loads(
                text[start:end + 1]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# SYMPTOM MAP
# ============================================================

def build_symptom_map(data):

    symptoms = data.get(
        "symptoms",
        []
    )

    if not isinstance(symptoms, list):
        return {}

    result = {}

    for symptom in symptoms:

        if not isinstance(symptom, dict):
            continue

        name = normalize_string(
            symptom.get("name")
        )

        if not isinstance(name, str):
            continue

        # Keep duplicates visible by storing list
        result.setdefault(
            name,
            []
        ).append(
            symptom
        )

    return result


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def analyze_prediction(
    prediction,
    target,
):

    failures = []

    if prediction is None:

        failures.append(
            {
                "type": "invalid_json",
                "detail": "Prediction could not be parsed.",
            }
        )

        return failures

    # --------------------------------------------------------
    # Chief complaint
    # --------------------------------------------------------

    predicted_chief = normalize_string(
        prediction.get(
            "chief_complaint"
        )
    )

    target_chief = normalize_string(
        target.get(
            "chief_complaint"
        )
    )

    if predicted_chief != target_chief:

        failures.append(
            {
                "type":
                    "wrong_chief_complaint",

                "expected":
                    target_chief,

                "predicted":
                    predicted_chief,
            }
        )

    # --------------------------------------------------------
    # Symptoms
    # --------------------------------------------------------

    predicted_map = build_symptom_map(
        prediction
    )

    target_map = build_symptom_map(
        target
    )

    predicted_names = set(
        predicted_map.keys()
    )

    target_names = set(
        target_map.keys()
    )

    # Missing symptoms
    for symptom in sorted(
        target_names
        - predicted_names
    ):

        failures.append(
            {
                "type":
                    "missing_symptom",

                "symptom":
                    symptom,
            }
        )

    # Hallucinated symptoms
    for symptom in sorted(
        predicted_names
        - target_names
    ):

        failures.append(
            {
                "type":
                    "extra_symptom",

                "symptom":
                    symptom,
            }
        )

    # --------------------------------------------------------
    # Duplicate symptoms
    # --------------------------------------------------------

    for (
        symptom_name,
        symptom_entries,
    ) in predicted_map.items():

        if len(symptom_entries) > 1:

            failures.append(
                {
                    "type":
                        "duplicate_symptom",

                    "symptom":
                        symptom_name,

                    "count":
                        len(symptom_entries),
                }
            )

    # --------------------------------------------------------
    # Shared symptoms
    # --------------------------------------------------------

    for symptom_name in sorted(
        target_names
        & predicted_names
    ):

        target_entries = target_map[
            symptom_name
        ]

        predicted_entries = (
            predicted_map[
                symptom_name
            ]
        )

        # If duplicate, compare first entry.
        target_symptom = (
            target_entries[0]
        )

        predicted_symptom = (
            predicted_entries[0]
        )

        # Status
        expected_status = (
            normalize_string(
                target_symptom.get(
                    "status"
                )
            )
        )

        predicted_status = (
            normalize_string(
                predicted_symptom.get(
                    "status"
                )
            )
        )

        if (
            expected_status
            != predicted_status
        ):

            failures.append(
                {
                    "type":
                        "wrong_symptom_status",

                    "symptom":
                        symptom_name,

                    "expected":
                        expected_status,

                    "predicted":
                        predicted_status,
                }
            )

        # Duration
        expected_duration = (
            normalize_duration(
                target_symptom.get(
                    "duration"
                )
            )
        )

        predicted_duration = (
            normalize_duration(
                predicted_symptom.get(
                    "duration"
                )
            )
        )

        if (
            expected_duration
            != predicted_duration
        ):

            failures.append(
                {
                    "type":
                        "wrong_duration",

                    "symptom":
                        symptom_name,

                    "expected":
                        expected_duration,

                    "predicted":
                        predicted_duration,
                }
            )

    # --------------------------------------------------------
    # Medication
    # --------------------------------------------------------

    expected_medications = (
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
        expected_medications
        != predicted_medications
    ):

        failures.append(
            {
                "type":
                    "medication_error",

                "expected":
                    expected_medications,

                "predicted":
                    predicted_medications,
            }
        )

    # --------------------------------------------------------
    # Allergy
    # --------------------------------------------------------

    expected_allergies = (
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
        expected_allergies
        != predicted_allergies
    ):

        failures.append(
            {
                "type":
                    "allergy_error",

                "expected":
                    expected_allergies,

                "predicted":
                    predicted_allergies,
            }
        )

    return failures


# ============================================================
# CORE CORRECT
# ============================================================

def core_is_correct(
    failures,
):

    return len(
        failures
    ) == 0


# ============================================================
# LOAD
# ============================================================

def load_records():

    records = []

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():

                records.append(
                    json.loads(
                        line
                    )
                )

    return records


# ============================================================
# FORMAT EXAMPLE
# ============================================================

def format_json(value):

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def format_failure_example(
    record,
):

    language = record[
        "language"
    ]

    parsed_prediction = (
        record[
            "_parsed_prediction"
        ]
    )

    failures = record[
        "_failures"
    ]

    lines = []

    lines.append(
        f"CASE: {record['case_id']}"
    )

    lines.append(
        f"LANGUAGE: "
        f"{LANGUAGE_NAMES.get(language, language)}"
    )

    lines.append("")

    lines.append(
        "INPUT:"
    )

    lines.append(
        record.get(
            "utterance",
            "<utterance unavailable>",
        )
    )

    lines.append("")

    lines.append(
        "TARGET:"
    )

    lines.append(
        format_json(
            record[
                "target"
            ]
        )
    )

    lines.append("")

    lines.append(
        "PREDICTION:"
    )

    if parsed_prediction is None:

        lines.append(
            str(
                record.get(
                    "prediction"
                )
            )
        )

    else:

        lines.append(
            format_json(
                parsed_prediction
            )
        )

    lines.append("")

    lines.append(
        "ERRORS:"
    )

    for failure in failures:

        lines.append(
            "- "
            + format_json(
                failure
            ).replace(
                "\n",
                " ",
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    records = load_records()

    print(
        "=" * 76
    )

    print(
        "NOOR HEALTH - CROSS-LANGUAGE FAILURE AUDIT"
    )

    print(
        "=" * 76
    )

    print(
        f"\nPredictions loaded: "
        f"{len(records)}"
    )

    # ========================================================
    # Analyze every record
    # ========================================================

    language_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "failed": 0,
            "failure_types": Counter(),
        }
    )

    case_results = defaultdict(
        dict
    )

    analyzed_records = []

    for record in records:

        language = record[
            "language"
        ]

        prediction = parse_prediction(
            record.get(
                "prediction"
            )
        )

        failures = analyze_prediction(
            prediction,
            record[
                "target"
            ],
        )

        correct = core_is_correct(
            failures
        )

        language_stats[
            language
        ][
            "total"
        ] += 1

        if correct:

            language_stats[
                language
            ][
                "correct"
            ] += 1

        else:

            language_stats[
                language
            ][
                "failed"
            ] += 1

        for failure in failures:

            language_stats[
                language
            ][
                "failure_types"
            ][
                failure["type"]
            ] += 1

        case_results[
            record["case_id"]
        ][language] = correct

        enriched = dict(
            record
        )

        enriched[
            "_parsed_prediction"
        ] = prediction

        enriched[
            "_failures"
        ] = failures

        enriched[
            "_core_correct"
        ] = correct

        analyzed_records.append(
            enriched
        )

    # ========================================================
    # Language summary
    # ========================================================

    print(
        "\nLANGUAGE SUMMARY"
    )

    print(
        "-" * 76
    )

    print(
        f"{'Language':<15}"
        f"{'Correct':>10}"
        f"{'Failed':>10}"
        f"{'Accuracy':>12}"
    )

    for language in sorted(
        language_stats
    ):

        stats = language_stats[
            language
        ]

        accuracy = (
            stats["correct"]
            / stats["total"]
            * 100
        )

        print(
            f"{LANGUAGE_NAMES.get(language, language):<15}"
            f"{stats['correct']:>10}"
            f"{stats['failed']:>10}"
            f"{accuracy:>11.2f}%"
        )

    # ========================================================
    # Failure types by language
    # ========================================================

    print(
        "\nFAILURE TYPES BY LANGUAGE"
    )

    print(
        "=" * 76
    )

    for language in sorted(
        language_stats
    ):

        stats = language_stats[
            language
        ]

        print()

        print(
            LANGUAGE_NAMES.get(
                language,
                language,
            )
        )

        print(
            "-" * 76
        )

        if not stats[
            "failure_types"
        ]:

            print(
                "No failures."
            )

            continue

        for (
            failure_type,
            count,
        ) in stats[
            "failure_types"
        ].most_common():

            print(
                f"{failure_type:<30}"
                f"{count:>5}"
            )

    # ========================================================
    # Cross-language case patterns
    # ========================================================

    all_languages = sorted(
        language_stats.keys()
    )

    all_correct = 0
    all_failed = 0

    only_hi_failed = 0
    only_sw_failed = 0

    hi_sw_failed_others_correct = 0

    mixed_cases = 0

    hi_failed_cases = []
    sw_failed_cases = []
    hi_sw_specific_cases = []

    for (
        case_id,
        results,
    ) in case_results.items():

        available = [
            results.get(language)
            for language
            in all_languages
            if language in results
        ]

        if available and all(
            available
        ):

            all_correct += 1
            continue

        if available and not any(
            available
        ):

            all_failed += 1
            continue

        mixed_cases += 1

        failed_languages = {
            language
            for language in all_languages
            if (
                language in results
                and not results[
                    language
                ]
            )
        }

        if failed_languages == {
            "hi"
        }:

            only_hi_failed += 1

            hi_failed_cases.append(
                case_id
            )

        if failed_languages == {
            "sw"
        }:

            only_sw_failed += 1

            sw_failed_cases.append(
                case_id
            )

        if failed_languages == {
            "hi",
            "sw",
        }:

            hi_sw_failed_others_correct += 1

            hi_sw_specific_cases.append(
                case_id
            )

    print(
        "\nCROSS-LANGUAGE CASE ANALYSIS"
    )

    print(
        "=" * 76
    )

    print(
        f"Unique cases: "
        f"{len(case_results)}"
    )

    print(
        f"Correct in every language: "
        f"{all_correct}"
    )

    print(
        f"Failed in every language: "
        f"{all_failed}"
    )

    print(
        f"Mixed success/failure: "
        f"{mixed_cases}"
    )

    print(
        f"Only Hindi failed: "
        f"{only_hi_failed}"
    )

    print(
        f"Only Swahili failed: "
        f"{only_sw_failed}"
    )

    print(
        "Hindi + Swahili failed, "
        "all other languages correct: "
        f"{hi_sw_failed_others_correct}"
    )

    # ========================================================
    # Human-readable report
    # ========================================================

    report_lines = []

    report_lines.append(
        "=" * 76
    )

    report_lines.append(
        "NOOR HEALTH - LANGUAGE FAILURE ANALYSIS"
    )

    report_lines.append(
        "=" * 76
    )

    report_lines.append("")

    report_lines.append(
        "LANGUAGE SUMMARY"
    )

    report_lines.append(
        "-" * 76
    )

    for language in sorted(
        language_stats
    ):

        stats = language_stats[
            language
        ]

        accuracy = (
            stats["correct"]
            / stats["total"]
            * 100
        )

        report_lines.append(
            f"{LANGUAGE_NAMES.get(language, language)}: "
            f"{stats['correct']}/"
            f"{stats['total']} "
            f"({accuracy:.2f}%)"
        )

        for (
            failure_type,
            count,
        ) in stats[
            "failure_types"
        ].most_common():

            report_lines.append(
                f"  {failure_type}: "
                f"{count}"
            )

        report_lines.append("")

    report_lines.append(
        "=" * 76
    )

    report_lines.append(
        "CROSS-LANGUAGE CASE ANALYSIS"
    )

    report_lines.append(
        "=" * 76
    )

    report_lines.append(
        f"Correct in every language: "
        f"{all_correct}"
    )

    report_lines.append(
        f"Failed in every language: "
        f"{all_failed}"
    )

    report_lines.append(
        f"Mixed success/failure: "
        f"{mixed_cases}"
    )

    report_lines.append(
        f"Only Hindi failed: "
        f"{only_hi_failed}"
    )

    report_lines.append(
        f"Only Swahili failed: "
        f"{only_sw_failed}"
    )

    report_lines.append(
        "Hindi + Swahili failed, "
        "all others correct: "
        f"{hi_sw_failed_others_correct}"
    )

    # ========================================================
    # Detailed Hindi / Swahili examples
    # ========================================================

    for language in [
        "hi",
        "sw",
    ]:

        report_lines.append("")

        report_lines.append(
            "=" * 76
        )

        report_lines.append(
            f"{LANGUAGE_NAMES[language].upper()} "
            f"FAILURE EXAMPLES"
        )

        report_lines.append(
            "=" * 76
        )

        failures_for_language = [
            record
            for record
            in analyzed_records
            if (
                record[
                    "language"
                ] == language
                and not record[
                    "_core_correct"
                ]
            )
        ]

        # First 20 examples
        for record in (
            failures_for_language[
                :20
            ]
        ):

            report_lines.append("")
            report_lines.append(
                format_failure_example(
                    record
                )
            )

            report_lines.append("")
            report_lines.append(
                "-" * 76
            )

    # ========================================================
    # Save TXT
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Save JSON summary
    # ========================================================

    json_summary = {
        "total_predictions":
            len(records),

        "unique_cases":
            len(case_results),

        "languages": {},

        "cross_language": {
            "all_correct":
                all_correct,

            "all_failed":
                all_failed,

            "mixed":
                mixed_cases,

            "only_hindi_failed":
                only_hi_failed,

            "only_swahili_failed":
                only_sw_failed,

            "hindi_and_swahili_failed_others_correct":
                hi_sw_failed_others_correct,

            "only_hindi_failed_case_ids":
                hi_failed_cases,

            "only_swahili_failed_case_ids":
                sw_failed_cases,

            "hindi_swahili_specific_case_ids":
                hi_sw_specific_cases,
        },
    }

    for (
        language,
        stats,
    ) in language_stats.items():

        json_summary[
            "languages"
        ][language] = {
            "total":
                stats["total"],

            "correct":
                stats["correct"],

            "failed":
                stats["failed"],

            "accuracy":
                (
                    stats["correct"]
                    / stats["total"]
                    * 100
                ),

            "failure_types":
                dict(
                    stats[
                        "failure_types"
                    ]
                ),
        }

    JSON_OUTPUT_FILE.write_text(
        json.dumps(
            json_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {JSON_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()