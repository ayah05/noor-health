import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "baseline_predictions.jsonl"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "baseline_metrics.json"
)

OUTPUT_TXT = (
    RESULTS_DIR
    / "baseline_metrics.txt"
)


# ============================================================
# BASIC HELPERS
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


def parse_prediction(
    prediction: Any,
) -> dict | None:

    if isinstance(prediction, dict):
        return prediction

    if not isinstance(prediction, str):
        return None

    text = prediction.strip()

    # Remove Markdown fences if the model produced them.
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

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Try extracting the outermost JSON object.
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
# NORMALIZATION
# ============================================================

def normalize_string(
    value: Any,
) -> str:

    if not isinstance(value, str):
        return ""

    return (
        value
        .strip()
        .lower()
    )


def normalize_string_list(
    values: Any,
) -> list[str]:

    if not isinstance(values, list):
        return []

    normalized = []

    for value in values:

        if isinstance(value, str):
            normalized.append(
                normalize_string(value)
            )

    return sorted(normalized)


def normalize_duration(
    duration: Any,
) -> dict | None:

    if duration is None:
        return None

    if not isinstance(duration, dict):
        return None

    value = duration.get("value")
    unit = duration.get("unit")

    if not isinstance(
        value,
        (int, float),
    ):
        return None

    if unit not in {
        "hours",
        "days",
        "weeks",
    }:
        return None

    return {
        "value": value,
        "unit": unit,
    }


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def is_schema_valid(
    prediction: dict,
) -> bool:

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
        prediction.get(
            "chief_complaint"
        ),
        str,
    ):
        return False

    # Symptoms
    symptoms = prediction.get(
        "symptoms"
    )

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

        if symptom.get(
            "status"
        ) not in {
            "present",
            "absent",
            "uncertain",
        }:
            return False

        duration = symptom.get(
            "duration"
        )

        if duration is not None:

            if not isinstance(
                duration,
                dict,
            ):
                return False

            if set(duration.keys()) != {
                "value",
                "unit",
            }:
                return False

            if not isinstance(
                duration.get("value"),
                (int, float),
            ):
                return False

            if duration.get(
                "unit"
            ) not in {
                "hours",
                "days",
                "weeks",
            }:
                return False

    # Medications + allergies
    for key in [
        "medications",
        "allergies",
    ]:

        group = prediction.get(key)

        if not isinstance(group, dict):
            return False

        if set(group.keys()) != {
            "status",
            "items",
        }:
            return False

        if group.get(
            "status"
        ) not in {
            "unknown",
            "none",
            "reported",
        }:
            return False

        items = group.get("items")

        if not isinstance(items, list):
            return False

        if not all(
            isinstance(item, str)
            for item in items
        ):
            return False

    missing = prediction.get(
        "missing_information"
    )

    if not isinstance(missing, list):
        return False

    if not all(
        isinstance(item, str)
        for item in missing
    ):
        return False

    return True


# ============================================================
# EXACT STRUCTURED MATCH
# ============================================================

def normalize_target_structure(
    data: dict,
) -> dict:

    symptoms = []

    for symptom in data.get(
        "symptoms",
        [],
    ):

        symptoms.append({
            "name": normalize_string(
                symptom.get("name")
            ),
            "status": symptom.get(
                "status"
            ),
            "duration": normalize_duration(
                symptom.get("duration")
            ),
        })

    symptoms.sort(
        key=lambda x: (
            x["name"],
            x["status"],
        )
    )

    def normalize_group(
        key: str,
    ) -> dict:

        group = data.get(
            key,
            {},
        )

        return {
            "status": group.get(
                "status"
            ),
            "items": normalize_string_list(
                group.get(
                    "items",
                    [],
                )
            ),
        }

    return {
        "chief_complaint":
            normalize_string(
                data.get(
                    "chief_complaint"
                )
            ),

        "symptoms":
            symptoms,

        "medications":
            normalize_group(
                "medications"
            ),

        "allergies":
            normalize_group(
                "allergies"
            ),

        "missing_information":
            sorted(
                normalize_string_list(
                    data.get(
                        "missing_information",
                        [],
                    )
                )
            ),
    }


def exact_match(
    target: dict,
    prediction: dict,
) -> bool:

    return (
        normalize_target_structure(
            target
        )
        ==
        normalize_target_structure(
            prediction
        )
    )


# ============================================================
# FIELD METRICS
# ============================================================

def chief_complaint_correct(
    target: dict,
    prediction: dict,
) -> bool:

    return (
        normalize_string(
            target.get(
                "chief_complaint"
            )
        )
        ==
        normalize_string(
            prediction.get(
                "chief_complaint"
            )
        )
    )


def symptom_names(
    data: dict,
) -> set[str]:

    result = set()

    symptoms = data.get(
        "symptoms",
        [],
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return result

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            continue

        name = normalize_string(
            symptom.get("name")
        )

        if name:
            result.add(name)

    return result


def symptom_extraction_correct(
    target: dict,
    prediction: dict,
) -> bool:

    return (
        symptom_names(target)
        ==
        symptom_names(prediction)
    )


def symptom_map(
    data: dict,
) -> dict[str, dict]:

    result = {}

    symptoms = data.get(
        "symptoms",
        [],
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return result

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            continue

        name = normalize_string(
            symptom.get("name")
        )

        if name:
            result[name] = symptom

    return result


def symptom_status_correct(
    target: dict,
    prediction: dict,
) -> bool:

    target_map = symptom_map(
        target
    )

    prediction_map = symptom_map(
        prediction
    )

    if (
        set(target_map.keys())
        !=
        set(prediction_map.keys())
    ):
        return False

    for name in target_map:

        if (
            target_map[name].get(
                "status"
            )
            !=
            prediction_map[name].get(
                "status"
            )
        ):
            return False

    return True


def duration_correct(
    target: dict,
    prediction: dict,
) -> bool:

    target_map = symptom_map(
        target
    )

    prediction_map = symptom_map(
        prediction
    )

    if (
        set(target_map.keys())
        !=
        set(prediction_map.keys())
    ):
        return False

    for name in target_map:

        target_duration = (
            normalize_duration(
                target_map[name].get(
                    "duration"
                )
            )
        )

        prediction_duration = (
            normalize_duration(
                prediction_map[name].get(
                    "duration"
                )
            )
        )

        if (
            target_duration
            !=
            prediction_duration
        ):
            return False

    return True


def information_group_correct(
    target: dict,
    prediction: dict,
    key: str,
) -> bool:

    target_group = target.get(
        key,
        {},
    )

    prediction_group = prediction.get(
        key,
        {},
    )

    if not isinstance(
        prediction_group,
        dict,
    ):
        return False

    if (
        target_group.get("status")
        !=
        prediction_group.get("status")
    ):
        return False

    return (
        normalize_string_list(
            target_group.get(
                "items",
                [],
            )
        )
        ==
        normalize_string_list(
            prediction_group.get(
                "items",
                [],
            )
        )
    )


def missing_information_correct(
    target: dict,
    prediction: dict,
) -> bool:

    return (
        set(
            normalize_string_list(
                target.get(
                    "missing_information",
                    [],
                )
            )
        )
        ==
        set(
            normalize_string_list(
                prediction.get(
                    "missing_information",
                    [],
                )
            )
        )
    )


# ============================================================
# METRIC CONTAINER
# ============================================================

METRIC_NAMES = [
    "valid_json",
    "schema_valid",
    "exact_match",
    "chief_complaint",
    "symptom_extraction",
    "symptom_status",
    "duration",
    "medications",
    "allergies",
    "missing_information",
]


def empty_metrics() -> dict:

    return {
        "total": 0,
        **{
            metric: 0
            for metric in METRIC_NAMES
        },
    }


def percentage(
    value: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        100.0 * value / total,
        2,
    )


def metrics_with_percentages(
    metrics: dict,
) -> dict:

    total = metrics["total"]

    output = {
        "total": total,
    }

    for metric in METRIC_NAMES:

        count = metrics[metric]

        output[metric] = {
            "count": count,
            "percentage": percentage(
                count,
                total,
            ),
        }

    return output


# ============================================================
# EVALUATE ONE RECORD
# ============================================================

def evaluate_record(
    record: dict,
) -> dict[str, bool]:

    target = record["target"]

    prediction = parse_prediction(
        record.get(
            "prediction"
        )
    )

    if prediction is None:

        return {
            metric: False
            for metric in METRIC_NAMES
        }

    results = {
        "valid_json": True,

        "schema_valid":
            is_schema_valid(
                prediction
            ),

        "exact_match":
            exact_match(
                target,
                prediction,
            ),

        "chief_complaint":
            chief_complaint_correct(
                target,
                prediction,
            ),

        "symptom_extraction":
            symptom_extraction_correct(
                target,
                prediction,
            ),

        "symptom_status":
            symptom_status_correct(
                target,
                prediction,
            ),

        "duration":
            duration_correct(
                target,
                prediction,
            ),

        "medications":
            information_group_correct(
                target,
                prediction,
                "medications",
            ),

        "allergies":
            information_group_correct(
                target,
                prediction,
                "allergies",
            ),

        "missing_information":
            missing_information_correct(
                target,
                prediction,
            ),
    }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NOOR HEALTH - BASELINE EVALUATION")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    records = load_jsonl(
        PREDICTIONS_FILE
    )

    print(
        f"\nPredictions loaded: "
        f"{len(records)}"
    )

    overall = empty_metrics()

    by_language = defaultdict(
        empty_metrics
    )

    failures = Counter()

    for record in records:

        language = record.get(
            "language",
            "unknown",
        )

        result = evaluate_record(
            record
        )

        overall["total"] += 1
        by_language[
            language
        ]["total"] += 1

        for metric, passed in (
            result.items()
        ):

            if passed:

                overall[
                    metric
                ] += 1

                by_language[
                    language
                ][
                    metric
                ] += 1

            else:

                failures[
                    metric
                ] += 1

    overall_output = (
        metrics_with_percentages(
            overall
        )
    )

    language_output = {}

    for language in sorted(
        by_language.keys()
    ):

        language_output[
            language
        ] = (
            metrics_with_percentages(
                by_language[
                    language
                ]
            )
        )

    final_output = {
        "model": "Qwen/Qwen3-0.6B",
        "evaluation": "baseline",
        "overall": overall_output,
        "by_language": language_output,
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            final_output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TERMINAL / TXT REPORT
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "NOOR HEALTH - BASELINE EVALUATION"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"Model:   Qwen/Qwen3-0.6B"
    )

    lines.append(
        f"Samples: {overall['total']}"
    )

    lines.append("")

    labels = {
        "valid_json":
            "Valid JSON",

        "schema_valid":
            "Schema valid",

        "exact_match":
            "Exact structured match",

        "chief_complaint":
            "Chief complaint accuracy",

        "symptom_extraction":
            "Symptom extraction accuracy",

        "symptom_status":
            "Symptom status accuracy",

        "duration":
            "Duration accuracy",

        "medications":
            "Medication accuracy",

        "allergies":
            "Allergy accuracy",

        "missing_information":
            "Missing information accuracy",
    }

    for metric in METRIC_NAMES:

        count = overall[
            metric
        ]

        pct = percentage(
            count,
            overall["total"],
        )

        lines.append(
            f"{labels[metric]:<32}"
            f"{count:>4} / "
            f"{overall['total']:<4} "
            f"{pct:>7.2f}%"
        )

    # --------------------------------------------------------
    # BY LANGUAGE
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 72
    )

    lines.append(
        "BY LANGUAGE"
    )

    lines.append(
        "=" * 72
    )

    language_names = {
        "en": "English",
        "de": "German",
        "ar_msa": "Arabic",
        "fr": "French",
        "es": "Spanish",
        "hi": "Hindi",
        "sw": "Swahili",
    }

    for language in [
        "en",
        "de",
        "ar_msa",
        "fr",
        "es",
        "hi",
        "sw",
    ]:

        metrics = by_language.get(
            language
        )

        if not metrics:
            continue

        lines.append("")
        lines.append(
            language_names.get(
                language,
                language,
            )
        )

        lines.append(
            "-" * 72
        )

        lines.append(
            f"Samples: "
            f"{metrics['total']}"
        )

        for metric in METRIC_NAMES:

            count = metrics[
                metric
            ]

            pct = percentage(
                count,
                metrics["total"],
            )

            lines.append(
                f"{labels[metric]:<32}"
                f"{count:>4} / "
                f"{metrics['total']:<4} "
                f"{pct:>7.2f}%"
            )

    report = "\n".join(
        lines
    )

    print()
    print(report)

    OUTPUT_TXT.write_text(
        report,
        encoding="utf-8",
    )

    print("\nSaved:")
    print(
        f"  {OUTPUT_JSON}"
    )
    print(
        f"  {OUTPUT_TXT}"
    )


if __name__ == "__main__":
    main()