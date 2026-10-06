import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_string(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()

    return value


def normalize_duration(duration: Any) -> Any:
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


def normalize_information_group(
    group: Any,
) -> Any:

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


def normalize_missing_information(
    value: Any,
) -> Any:

    if not isinstance(value, list):
        return value

    normalized = [
        normalize_string(item)
        for item in value
    ]

    try:
        return sorted(normalized)
    except TypeError:
        return normalized


# ============================================================
# JSON PARSING
# ============================================================

def parse_prediction(
    prediction: Any,
) -> dict | None:

    if isinstance(prediction, dict):
        return prediction

    if not isinstance(prediction, str):
        return None

    text = prediction.strip()

    # Remove Markdown fences if present.
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

    # Strict JSON parsing.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        return None

    return None


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def is_valid_duration(
    duration: Any,
) -> bool:

    if duration is None:
        return True

    if not isinstance(duration, dict):
        return False

    if set(duration.keys()) != {
        "value",
        "unit",
    }:
        return False

    value = duration.get("value")
    unit = duration.get("unit")

    if not isinstance(value, int):
        return False

    if unit not in {
        "hours",
        "days",
        "weeks",
    }:
        return False

    return True


def is_valid_information_group(
    group: Any,
) -> bool:

    if not isinstance(group, dict):
        return False

    if set(group.keys()) != {
        "status",
        "items",
    }:
        return False

    status = group.get("status")
    items = group.get("items")

    if status not in {
        "unknown",
        "none",
        "reported",
    }:
        return False

    if not isinstance(items, list):
        return False

    if not all(
        isinstance(item, str)
        for item in items
    ):
        return False

    return True


def is_schema_valid(
    prediction: dict | None,
) -> bool:

    if not isinstance(prediction, dict):
        return False

    required_keys = {
        "chief_complaint",
        "symptoms",
        "medications",
        "allergies",
        "missing_information",
    }

    if set(prediction.keys()) != required_keys:
        return False

    # Chief complaint
    if not isinstance(
        prediction.get("chief_complaint"),
        str,
    ):
        return False

    # Symptoms
    symptoms = prediction.get("symptoms")

    if not isinstance(symptoms, list):
        return False

    for symptom in symptoms:

        if not isinstance(symptom, dict):
            return False

        if set(symptom.keys()) != {
            "name",
            "status",
            "duration",
        }:
            return False

        if not isinstance(
            symptom.get("name"),
            str,
        ):
            return False

        if symptom.get("status") not in {
            "present",
            "absent",
            "uncertain",
        }:
            return False

        if not is_valid_duration(
            symptom.get("duration")
        ):
            return False

        # Only present symptoms may have duration.
        if (
            symptom.get("status")
            != "present"
            and symptom.get("duration")
            is not None
        ):
            return False

    # Medications
    if not is_valid_information_group(
        prediction.get("medications")
    ):
        return False

    # Allergies
    if not is_valid_information_group(
        prediction.get("allergies")
    ):
        return False

    # Missing information
    missing = prediction.get(
        "missing_information"
    )

    if not isinstance(missing, list):
        return False

    if not all(
        item in {
            "duration",
            "medications",
            "allergies",
        }
        for item in missing
    ):
        return False

    return True


# ============================================================
# SYMPTOM HELPERS
# ============================================================

def build_symptom_map(
    data: dict,
) -> dict:

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

        # Duplicate names are represented separately
        # by making the structure invalid for exact match.
        if name in result:
            return {}

        result[name] = {
            "status": normalize_string(
                symptom.get("status")
            ),
            "duration": normalize_duration(
                symptom.get("duration")
            ),
        }

    return result


# ============================================================
# COMPONENT COMPARISON
# ============================================================

def chief_complaint_correct(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        normalize_string(
            prediction.get(
                "chief_complaint"
            )
        )
        ==
        normalize_string(
            target.get(
                "chief_complaint"
            )
        )
    )


def symptom_extraction_correct(
    prediction: dict,
    target: dict,
) -> bool:

    predicted = build_symptom_map(
        prediction
    )

    expected = build_symptom_map(
        target
    )

    return set(predicted.keys()) == set(
        expected.keys()
    )


def symptom_status_correct(
    prediction: dict,
    target: dict,
) -> bool:

    predicted = build_symptom_map(
        prediction
    )

    expected = build_symptom_map(
        target
    )

    if set(predicted.keys()) != set(
        expected.keys()
    ):
        return False

    for name in expected:

        if (
            predicted[name]["status"]
            != expected[name]["status"]
        ):
            return False

    return True


def duration_correct(
    prediction: dict,
    target: dict,
) -> bool:

    predicted = build_symptom_map(
        prediction
    )

    expected = build_symptom_map(
        target
    )

    if set(predicted.keys()) != set(
        expected.keys()
    ):
        return False

    for name in expected:

        if (
            predicted[name]["duration"]
            != expected[name]["duration"]
        ):
            return False

    return True


def medication_correct(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        normalize_information_group(
            prediction.get(
                "medications"
            )
        )
        ==
        normalize_information_group(
            target.get(
                "medications"
            )
        )
    )


def allergy_correct(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        normalize_information_group(
            prediction.get(
                "allergies"
            )
        )
        ==
        normalize_information_group(
            target.get(
                "allergies"
            )
        )
    )


def raw_missing_information_correct(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        normalize_missing_information(
            prediction.get(
                "missing_information"
            )
        )
        ==
        normalize_missing_information(
            target.get(
                "missing_information"
            )
        )
    )


# ============================================================
# DETERMINISTIC MISSING INFORMATION
# ============================================================

def derive_missing_information(
    result: dict,
) -> list[str]:

    missing = []

    chief = normalize_string(
        result.get(
            "chief_complaint"
        )
    )

    symptoms = result.get(
        "symptoms",
        []
    )

    chief_symptom = None

    if isinstance(symptoms, list):

        for symptom in symptoms:

            if not isinstance(
                symptom,
                dict,
            ):
                continue

            if (
                normalize_string(
                    symptom.get("name")
                )
                == chief
            ):
                chief_symptom = symptom
                break

    if (
        chief_symptom is None
        or chief_symptom.get(
            "duration"
        ) is None
    ):
        missing.append(
            "duration"
        )

    medications = result.get(
        "medications",
        {}
    )

    if (
        isinstance(medications, dict)
        and medications.get("status")
        == "unknown"
    ):
        missing.append(
            "medications"
        )

    allergies = result.get(
        "allergies",
        {}
    )

    if (
        isinstance(allergies, dict)
        and allergies.get("status")
        == "unknown"
    ):
        missing.append(
            "allergies"
        )

    return missing


def derived_missing_information_correct(
    prediction: dict,
    target: dict,
) -> bool:

    predicted_missing = (
        derive_missing_information(
            prediction
        )
    )

    target_missing = (
        normalize_missing_information(
            target.get(
                "missing_information"
            )
        )
    )

    return (
        normalize_missing_information(
            predicted_missing
        )
        == target_missing
    )


# ============================================================
# CORE EXACT MATCH
# ============================================================

def core_exact_match(
    prediction: dict,
    target: dict,
) -> bool:

    return all(
        [
            chief_complaint_correct(
                prediction,
                target,
            ),
            symptom_extraction_correct(
                prediction,
                target,
            ),
            symptom_status_correct(
                prediction,
                target,
            ),
            duration_correct(
                prediction,
                target,
            ),
            medication_correct(
                prediction,
                target,
            ),
            allergy_correct(
                prediction,
                target,
            ),
        ]
    )


# ============================================================
# RAW EXACT MATCH
# ============================================================

def raw_exact_match(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        core_exact_match(
            prediction,
            target,
        )
        and
        raw_missing_information_correct(
            prediction,
            target,
        )
    )


# ============================================================
# PIPELINE EXACT MATCH
# ============================================================

def pipeline_exact_match(
    prediction: dict,
    target: dict,
) -> bool:

    return (
        core_exact_match(
            prediction,
            target,
        )
        and
        derived_missing_information_correct(
            prediction,
            target,
        )
    )


# ============================================================
# METRIC NAMES
# ============================================================

METRICS = [
    "valid_json",
    "schema_valid",
    "raw_exact",
    "core_exact",
    "pipeline_exact",
    "chief_complaint",
    "symptom_extraction",
    "symptom_status",
    "duration",
    "medication",
    "allergy",
    "missing_information_raw",
    "missing_information_derived",
]


DISPLAY_NAMES = {
    "valid_json":
        "Valid JSON",

    "schema_valid":
        "Schema valid",

    "raw_exact":
        "Raw exact structured match",

    "core_exact":
        "Core extraction exact match",

    "pipeline_exact":
        "Pipeline exact match",

    "chief_complaint":
        "Chief complaint accuracy",

    "symptom_extraction":
        "Symptom extraction accuracy",

    "symptom_status":
        "Symptom status accuracy",

    "duration":
        "Duration accuracy",

    "medication":
        "Medication accuracy",

    "allergy":
        "Allergy accuracy",

    "missing_information_raw":
        "Missing information (raw)",

    "missing_information_derived":
        "Missing information (derived)",
}


# ============================================================
# EMPTY COUNTER
# ============================================================

def empty_metric_counter() -> dict:
    return {
        metric: 0
        for metric in METRICS
    }


# ============================================================
# EVALUATE ONE SAMPLE
# ============================================================

def evaluate_sample(
    record: dict,
) -> dict[str, bool]:

    target = record.get(
        "target",
        {}
    )

    parsed = parse_prediction(
        record.get(
            "prediction"
        )
    )

    result = {
        metric: False
        for metric in METRICS
    }

    if parsed is None:
        return result

    result[
        "valid_json"
    ] = True

    result[
        "schema_valid"
    ] = is_schema_valid(
        parsed
    )

    result[
        "chief_complaint"
    ] = chief_complaint_correct(
        parsed,
        target,
    )

    result[
        "symptom_extraction"
    ] = symptom_extraction_correct(
        parsed,
        target,
    )

    result[
        "symptom_status"
    ] = symptom_status_correct(
        parsed,
        target,
    )

    result[
        "duration"
    ] = duration_correct(
        parsed,
        target,
    )

    result[
        "medication"
    ] = medication_correct(
        parsed,
        target,
    )

    result[
        "allergy"
    ] = allergy_correct(
        parsed,
        target,
    )

    result[
        "missing_information_raw"
    ] = (
        raw_missing_information_correct(
            parsed,
            target,
        )
    )

    result[
        "missing_information_derived"
    ] = (
        derived_missing_information_correct(
            parsed,
            target,
        )
    )

    result[
        "core_exact"
    ] = core_exact_match(
        parsed,
        target,
    )

    result[
        "raw_exact"
    ] = raw_exact_match(
        parsed,
        target,
    )

    result[
        "pipeline_exact"
    ] = pipeline_exact_match(
        parsed,
        target,
    )

    return result


# ============================================================
# PRINT HELPERS
# ============================================================

def print_metric(
    name: str,
    correct: int,
    total: int,
):

    percentage = (
        correct / total * 100
        if total
        else 0.0
    )

    print(
        f"{name:<36}"
        f"{correct:>5} / "
        f"{total:<5}"
        f"{percentage:>8.2f}%"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_file(
    predictions_file: Path,
    model_name: str,
    output_dir: Path,
):

    if not predictions_file.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{predictions_file}"
        )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    records = []

    with predictions_file.open(
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
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSONL record at "
                    f"line {line_number}."
                ) from exc

            records.append(record)

    if not records:

        raise ValueError(
            "Prediction file is empty."
        )

    total = len(records)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    overall = (
        empty_metric_counter()
    )

    by_language = defaultdict(
        empty_metric_counter
    )

    language_totals = defaultdict(
        int
    )

    evaluated_samples = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for record in records:

        language = record.get(
            "language",
            "unknown",
        )

        sample_metrics = (
            evaluate_sample(
                record
            )
        )

        language_totals[
            language
        ] += 1

        for metric in METRICS:

            if sample_metrics[
                metric
            ]:

                overall[
                    metric
                ] += 1

                by_language[
                    language
                ][
                    metric
                ] += 1

        evaluated_samples.append(
            {
                "case_id":
                    record.get(
                        "case_id"
                    ),

                "language":
                    language,

                "metrics":
                    sample_metrics,
            }
        )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "=" * 76
    )

    print(
        f"NOOR HEALTH - {model_name}"
    )

    print(
        "=" * 76
    )

    print(
        f"\nPredictions: {total}"
    )

    print(
        f"Source: {predictions_file}"
    )

    print(
        "\nSTRUCTURED EXTRACTION SUMMARY"
    )

    print(
        "-" * 76
    )

    for metric in [
        "raw_exact",
        "core_exact",
        "pipeline_exact",
    ]:

        print_metric(
            DISPLAY_NAMES[
                metric
            ],
            overall[
                metric
            ],
            total,
        )

    print(
        "\nOVERALL"
    )

    print(
        "-" * 76
    )

    for metric in METRICS:

        print_metric(
            DISPLAY_NAMES[
                metric
            ],
            overall[
                metric
            ],
            total,
        )

    print(
        "\nFAILURE COUNTS"
    )

    print(
        "-" * 76
    )

    for metric in METRICS:

        failures = (
            total
            - overall[
                metric
            ]
        )

        print(
            f"{DISPLAY_NAMES[metric]:<36}"
            f"{failures:>5}"
        )

    # ========================================================
    # BY LANGUAGE
    # ========================================================

    print(
        "\nBY LANGUAGE"
    )

    print(
        "=" * 76
    )

    for language in sorted(
        language_totals
    ):

        language_total = (
            language_totals[
                language
            ]
        )

        print(
            f"\n{language}"
        )

        print(
            "-" * 76
        )

        for metric in METRICS:

            print_metric(
                DISPLAY_NAMES[
                    metric
                ],
                by_language[
                    language
                ][
                    metric
                ],
                language_total,
            )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_model_name = (
        model_name
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )

    metrics_json_file = (
        output_dir
        / f"{safe_model_name}_metrics.json"
    )

    sample_metrics_file = (
        output_dir
        / f"{safe_model_name}_sample_metrics.jsonl"
    )

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    summary = {
        "model_name":
            model_name,

        "predictions_file":
            str(
                predictions_file
            ),

        "total":
            total,

        "overall": {},

        "by_language": {},
    }

    for metric in METRICS:

        correct = overall[
            metric
        ]

        summary[
            "overall"
        ][metric] = {
            "correct":
                correct,

            "total":
                total,

            "accuracy":
                correct / total,
        }

    for language in sorted(
        language_totals
    ):

        language_total = (
            language_totals[
                language
            ]
        )

        summary[
            "by_language"
        ][language] = {}

        for metric in METRICS:

            correct = (
                by_language[
                    language
                ][
                    metric
                ]
            )

            summary[
                "by_language"
            ][language][metric] = {
                "correct":
                    correct,

                "total":
                    language_total,

                "accuracy":
                    (
                        correct
                        / language_total
                    ),
            }

    metrics_json_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Per-sample metrics
    # --------------------------------------------------------

    with sample_metrics_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for sample in evaluated_samples:

            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\nSaved:"
    )

    print(
        f"  {metrics_json_file}"
    )

    print(
        f"  {sample_metrics_file}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Noor Health model "
            "predictions using a shared "
            "model-independent evaluator."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "Path to the prediction "
            "JSONL file."
        ),
    )

    parser.add_argument(
        "--name",
        required=True,
        help=(
            "Human-readable model name."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for evaluation "
            "results."
        ),
    )

    args = parser.parse_args()

    predictions_file = Path(
        args.predictions
    ).resolve()

    # If no output directory is supplied,
    # store metrics next to the predictions.
    if args.output_dir:

        output_dir = Path(
            args.output_dir
        ).resolve()

    else:

        output_dir = (
            predictions_file.parent
        )

    evaluate_file(
        predictions_file=(
            predictions_file
        ),
        model_name=args.name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()