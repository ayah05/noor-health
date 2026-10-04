import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    ROOT_DIR
    / "results"
    / "lora_predictions.jsonl"
)

METRICS_JSON_FILE = (
    ROOT_DIR
    / "results"
    / "lora_metrics.json"
)

METRICS_TXT_FILE = (
    ROOT_DIR
    / "results"
    / "lora_metrics.txt"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_SAMPLES = 700

EXPECTED_TOP_LEVEL_KEYS = {
    "chief_complaint",
    "symptoms",
    "medications",
    "allergies",
    "missing_information",
}

EXPECTED_SYMPTOM_KEYS = {
    "name",
    "status",
    "duration",
}

VALID_SYMPTOM_STATUSES = {
    "present",
    "absent",
    "uncertain",
}

VALID_INFORMATION_STATUSES = {
    "unknown",
    "none",
    "reported",
}

VALID_DURATION_UNITS = {
    "hours",
    "days",
    "weeks",
}

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
# JSON PARSING
# ============================================================

def parse_prediction(prediction):

    if isinstance(prediction, dict):
        return prediction

    if not isinstance(prediction, str):
        return None

    text = prediction.strip()

    # Remove Markdown fences
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

    # Direct JSON
    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Extract outermost JSON object
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

def normalize_string(value):

    if not isinstance(value, str):
        return value

    return value.strip().lower()


def normalize_string_list(values):

    if not isinstance(values, list):
        return values

    return sorted(
        normalize_string(value)
        for value in values
    )


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


def normalize_symptom(symptom):

    if not isinstance(symptom, dict):
        return symptom

    return {
        "name": normalize_string(
            symptom.get("name")
        ),
        "status": normalize_string(
            symptom.get("status")
        ),
        "duration": normalize_duration(
            symptom.get("duration")
        ),
    }


def normalize_symptoms(symptoms):

    if not isinstance(symptoms, list):
        return symptoms

    normalized = [
        normalize_symptom(symptom)
        for symptom in symptoms
    ]

    return sorted(
        normalized,
        key=lambda symptom: (
            str(symptom.get("name")),
            str(symptom.get("status")),
            json.dumps(
                symptom.get("duration"),
                sort_keys=True,
            ),
        ),
    )


def normalize_information_group(group):

    if not isinstance(group, dict):
        return group

    return {
        "status": normalize_string(
            group.get("status")
        ),
        "items": normalize_string_list(
            group.get("items", [])
        ),
    }


def normalize_structure(data):

    if not isinstance(data, dict):
        return data

    return {
        "chief_complaint":
            normalize_string(
                data.get(
                    "chief_complaint"
                )
            ),

        "symptoms":
            normalize_symptoms(
                data.get(
                    "symptoms",
                    []
                )
            ),

        "medications":
            normalize_information_group(
                data.get(
                    "medications"
                )
            ),

        "allergies":
            normalize_information_group(
                data.get(
                    "allergies"
                )
            ),

        "missing_information":
            normalize_string_list(
                data.get(
                    "missing_information",
                    []
                )
            ),
    }


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def is_schema_valid(prediction):

    if not isinstance(prediction, dict):
        return False

    if set(prediction.keys()) != EXPECTED_TOP_LEVEL_KEYS:
        return False

    # --------------------------------------------------------
    # Chief complaint
    # --------------------------------------------------------

    chief_complaint = prediction.get(
        "chief_complaint"
    )

    if not isinstance(
        chief_complaint,
        str,
    ):
        return False

    # --------------------------------------------------------
    # Symptoms
    # --------------------------------------------------------

    symptoms = prediction.get(
        "symptoms"
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return False

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            return False

        if (
            set(symptom.keys())
            != EXPECTED_SYMPTOM_KEYS
        ):
            return False

        if not isinstance(
            symptom.get("name"),
            str,
        ):
            return False

        if (
            symptom.get("status")
            not in VALID_SYMPTOM_STATUSES
        ):
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

            value = duration.get(
                "value"
            )

            if (
                not isinstance(
                    value,
                    (int, float),
                )
                or isinstance(
                    value,
                    bool,
                )
            ):
                return False

            if (
                duration.get("unit")
                not in VALID_DURATION_UNITS
            ):
                return False

    # --------------------------------------------------------
    # Medications / allergies
    # --------------------------------------------------------

    for field_name in [
        "medications",
        "allergies",
    ]:

        group = prediction.get(
            field_name
        )

        if not isinstance(
            group,
            dict,
        ):
            return False

        if set(group.keys()) != {
            "status",
            "items",
        }:
            return False

        if (
            group.get("status")
            not in VALID_INFORMATION_STATUSES
        ):
            return False

        items = group.get(
            "items"
        )

        if not isinstance(
            items,
            list,
        ):
            return False

        if not all(
            isinstance(item, str)
            for item in items
        ):
            return False

    # --------------------------------------------------------
    # Missing information
    # --------------------------------------------------------

    missing = prediction.get(
        "missing_information"
    )

    if not isinstance(
        missing,
        list,
    ):
        return False

    if not all(
        isinstance(item, str)
        for item in missing
    ):
        return False

    return True


# ============================================================
# INDIVIDUAL METRICS
# ============================================================

def chief_complaint_correct(
    prediction,
    target,
):

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


def get_symptom_names(data):

    symptoms = data.get(
        "symptoms",
        []
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return None

    names = []

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            return None

        names.append(
            normalize_string(
                symptom.get("name")
            )
        )

    return sorted(names)


def symptom_extraction_correct(
    prediction,
    target,
):

    return (
        get_symptom_names(prediction)
        ==
        get_symptom_names(target)
    )


def symptom_map(data):

    symptoms = data.get(
        "symptoms",
        []
    )

    if not isinstance(
        symptoms,
        list,
    ):
        return None

    result = {}

    for symptom in symptoms:

        if not isinstance(
            symptom,
            dict,
        ):
            return None

        name = normalize_string(
            symptom.get("name")
        )

        if name in result:
            return None

        result[name] = symptom

    return result


def symptom_status_correct(
    prediction,
    target,
):

    prediction_map = symptom_map(
        prediction
    )

    target_map = symptom_map(
        target
    )

    if (
        prediction_map is None
        or target_map is None
    ):
        return False

    if (
        set(prediction_map.keys())
        != set(target_map.keys())
    ):
        return False

    for name in target_map:

        prediction_status = (
            normalize_string(
                prediction_map[
                    name
                ].get("status")
            )
        )

        target_status = (
            normalize_string(
                target_map[
                    name
                ].get("status")
            )
        )

        if (
            prediction_status
            != target_status
        ):
            return False

    return True


def duration_correct(
    prediction,
    target,
):

    prediction_map = symptom_map(
        prediction
    )

    target_map = symptom_map(
        target
    )

    if (
        prediction_map is None
        or target_map is None
    ):
        return False

    if (
        set(prediction_map.keys())
        != set(target_map.keys())
    ):
        return False

    for name in target_map:

        prediction_duration = (
            normalize_duration(
                prediction_map[
                    name
                ].get(
                    "duration"
                )
            )
        )

        target_duration = (
            normalize_duration(
                target_map[
                    name
                ].get(
                    "duration"
                )
            )
        )

        if (
            prediction_duration
            != target_duration
        ):
            return False

    return True


def information_group_correct(
    prediction,
    target,
    field_name,
):

    prediction_group = (
        normalize_information_group(
            prediction.get(
                field_name
            )
        )
    )

    target_group = (
        normalize_information_group(
            target.get(
                field_name
            )
        )
    )

    return (
        prediction_group
        == target_group
    )


def missing_information_correct(
    prediction,
    target,
):

    prediction_missing = (
        normalize_string_list(
            prediction.get(
                "missing_information",
                []
            )
        )
    )

    target_missing = (
        normalize_string_list(
            target.get(
                "missing_information",
                []
            )
        )
    )

    return (
        prediction_missing
        == target_missing
    )


# ============================================================
# RAW EXACT MATCH
# ============================================================

def raw_exact_match(
    prediction,
    target,
):

    return (
        normalize_structure(
            prediction
        )
        ==
        normalize_structure(
            target
        )
    )


# ============================================================
# CORE EXTRACTION EXACT MATCH
# ============================================================

def normalize_core_extraction(data):

    normalized = normalize_structure(
        data
    )

    return {
        "chief_complaint":
            normalized[
                "chief_complaint"
            ],

        "symptoms":
            normalized[
                "symptoms"
            ],

        "medications":
            normalized[
                "medications"
            ],

        "allergies":
            normalized[
                "allergies"
            ],
    }


def core_extraction_exact_match(
    prediction,
    target,
):

    return (
        normalize_core_extraction(
            prediction
        )
        ==
        normalize_core_extraction(
            target
        )
    )


# ============================================================
# DETERMINISTIC MISSING INFORMATION
# ============================================================

def derive_missing_information(
    prediction,
):

    missing = []

    chief_complaint = (
        normalize_string(
            prediction.get(
                "chief_complaint"
            )
        )
    )

    symptoms = prediction.get(
        "symptoms",
        []
    )

    chief_symptom = None

    if isinstance(
        symptoms,
        list,
    ):

        for symptom in symptoms:

            if not isinstance(
                symptom,
                dict,
            ):
                continue

            symptom_name = (
                normalize_string(
                    symptom.get(
                        "name"
                    )
                )
            )

            if (
                symptom_name
                == chief_complaint
            ):

                chief_symptom = (
                    symptom
                )

                break

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if (
        chief_symptom is None
        or chief_symptom.get(
            "duration"
        ) is None
    ):
        missing.append(
            "duration"
        )

    # --------------------------------------------------------
    # Medications
    # --------------------------------------------------------

    medications = prediction.get(
        "medications"
    )

    if (
        isinstance(
            medications,
            dict,
        )
        and medications.get(
            "status"
        ) == "unknown"
    ):
        missing.append(
            "medications"
        )

    # --------------------------------------------------------
    # Allergies
    # --------------------------------------------------------

    allergies = prediction.get(
        "allergies"
    )

    if (
        isinstance(
            allergies,
            dict,
        )
        and allergies.get(
            "status"
        ) == "unknown"
    ):
        missing.append(
            "allergies"
        )

    return missing


# ============================================================
# PIPELINE EXACT MATCH
# ============================================================

def build_pipeline_prediction(
    prediction,
):

    pipeline_prediction = {
        "chief_complaint":
            prediction.get(
                "chief_complaint"
            ),

        "symptoms":
            prediction.get(
                "symptoms",
                []
            ),

        "medications":
            prediction.get(
                "medications"
            ),

        "allergies":
            prediction.get(
                "allergies"
            ),

        "missing_information":
            derive_missing_information(
                prediction
            ),
    }

    return pipeline_prediction


def pipeline_exact_match(
    prediction,
    target,
):

    pipeline_prediction = (
        build_pipeline_prediction(
            prediction
        )
    )

    return (
        normalize_structure(
            pipeline_prediction
        )
        ==
        normalize_structure(
            target
        )
    )


# ============================================================
# METRICS
# ============================================================

METRIC_NAMES = [
    "valid_json",
    "schema_valid",
    "raw_exact_match",
    "core_exact_match",
    "pipeline_exact_match",
    "chief_complaint",
    "symptom_extraction",
    "symptom_status",
    "duration",
    "medications",
    "allergies",
    "missing_information_raw",
    "missing_information_derived",
]


METRIC_LABELS = {
    "valid_json":
        "Valid JSON",

    "schema_valid":
        "Schema valid",

    "raw_exact_match":
        "Raw exact structured match",

    "core_exact_match":
        "Core extraction exact match",

    "pipeline_exact_match":
        "Pipeline exact match",

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

    "missing_information_raw":
        "Missing information (raw)",

    "missing_information_derived":
        "Missing information (derived)",
}


def empty_metrics():

    return {
        metric: 0
        for metric in METRIC_NAMES
    }


def evaluate_record(
    record,
):

    result = {
        metric: False
        for metric in METRIC_NAMES
    }

    target = record.get(
        "target"
    )

    if not isinstance(
        target,
        dict,
    ):
        return result

    prediction = parse_prediction(
        record.get(
            "prediction"
        )
    )

    if prediction is None:
        return result

    result[
        "valid_json"
    ] = True

    result[
        "schema_valid"
    ] = is_schema_valid(
        prediction
    )

    # --------------------------------------------------------
    # Raw model metrics
    # --------------------------------------------------------

    result[
        "raw_exact_match"
    ] = raw_exact_match(
        prediction,
        target,
    )

    result[
        "core_exact_match"
    ] = (
        core_extraction_exact_match(
            prediction,
            target,
        )
    )

    result[
        "chief_complaint"
    ] = (
        chief_complaint_correct(
            prediction,
            target,
        )
    )

    result[
        "symptom_extraction"
    ] = (
        symptom_extraction_correct(
            prediction,
            target,
        )
    )

    result[
        "symptom_status"
    ] = (
        symptom_status_correct(
            prediction,
            target,
        )
    )

    result[
        "duration"
    ] = duration_correct(
        prediction,
        target,
    )

    result[
        "medications"
    ] = (
        information_group_correct(
            prediction,
            target,
            "medications",
        )
    )

    result[
        "allergies"
    ] = (
        information_group_correct(
            prediction,
            target,
            "allergies",
        )
    )

    result[
        "missing_information_raw"
    ] = (
        missing_information_correct(
            prediction,
            target,
        )
    )

    # --------------------------------------------------------
    # Deterministic pipeline metrics
    # --------------------------------------------------------

    derived_missing = (
        derive_missing_information(
            prediction
        )
    )

    target_missing = (
        normalize_string_list(
            target.get(
                "missing_information",
                []
            )
        )
    )

    result[
        "missing_information_derived"
    ] = (
        normalize_string_list(
            derived_missing
        )
        ==
        target_missing
    )

    result[
        "pipeline_exact_match"
    ] = (
        pipeline_exact_match(
            prediction,
            target,
        )
    )

    return result


# ============================================================
# REPORTING
# ============================================================

def percentage(
    correct,
    total,
):

    if total == 0:
        return 0.0

    return (
        correct
        / total
        * 100
    )


def format_metric(
    label,
    correct,
    total,
):

    pct = percentage(
        correct,
        total,
    )

    return (
        f"{label:<36}"
        f"{correct:>4} / "
        f"{total:<4} "
        f"{pct:>7.2f}%"
    )


def build_report(
    overall,
    by_language,
    total,
):

    lines = []

    lines.append(
        "=" * 76
    )

    lines.append(
        "NOOR HEALTH - LORA EVALUATION"
    )

    lines.append(
        "=" * 76
    )

    lines.append("")

    lines.append(
        "Model: Qwen/Qwen3-0.6B + LoRA"
    )

    lines.append(
        f"Samples: {total}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Main comparison
    # --------------------------------------------------------

    lines.append(
        "STRUCTURED EXTRACTION SUMMARY"
    )

    lines.append(
        "-" * 76
    )

    for metric in [
        "raw_exact_match",
        "core_exact_match",
        "pipeline_exact_match",
    ]:

        lines.append(
            format_metric(
                METRIC_LABELS[
                    metric
                ],
                overall[
                    metric
                ],
                total,
            )
        )

    lines.append("")

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    lines.append(
        "OVERALL"
    )

    lines.append(
        "-" * 76
    )

    for metric in METRIC_NAMES:

        lines.append(
            format_metric(
                METRIC_LABELS[
                    metric
                ],
                overall[
                    metric
                ],
                total,
            )
        )

    lines.append("")

    # --------------------------------------------------------
    # Failure counts
    # --------------------------------------------------------

    lines.append(
        "=" * 76
    )

    lines.append(
        "FAILURE COUNTS"
    )

    lines.append(
        "=" * 76
    )

    for metric in METRIC_NAMES:

        failures = (
            total
            - overall[
                metric
            ]
        )

        lines.append(
            f"{METRIC_LABELS[metric]:<36}"
            f"{failures:>4}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Languages
    # --------------------------------------------------------

    lines.append(
        "=" * 76
    )

    lines.append(
        "BY LANGUAGE"
    )

    lines.append(
        "=" * 76
    )

    for language in sorted(
        by_language.keys()
    ):

        language_data = (
            by_language[
                language
            ]
        )

        language_total = (
            language_data[
                "total"
            ]
        )

        display_name = (
            LANGUAGE_NAMES.get(
                language,
                language,
            )
        )

        lines.append("")

        lines.append(
            display_name
        )

        lines.append(
            "-" * 76
        )

        lines.append(
            f"Samples: "
            f"{language_total}"
        )

        for metric in METRIC_NAMES:

            lines.append(
                format_metric(
                    METRIC_LABELS[
                        metric
                    ],
                    language_data[
                        "metrics"
                    ][
                        metric
                    ],
                    language_total,
                )
            )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 76
    )

    print(
        "NOOR HEALTH - LORA EVALUATION"
    )

    print(
        "=" * 76
    )

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions not found:\n"
            f"{PREDICTIONS_FILE}"
        )

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

    print(
        f"\nPredictions loaded: "
        f"{len(records)}"
    )

    if (
        len(records)
        != EXPECTED_SAMPLES
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_SAMPLES} "
            f"predictions, got "
            f"{len(records)}."
        )

    # --------------------------------------------------------
    # Dataset checks
    # --------------------------------------------------------

    for record in records:

        if (
            record.get("split")
            != "test"
        ):

            raise ValueError(
                "Found non-test "
                "sample in predictions."
            )

        for required_key in [
            "case_id",
            "language",
            "target",
            "prediction",
        ]:

            if (
                required_key
                not in record
            ):

                raise ValueError(
                    f"Missing key "
                    f"'{required_key}' "
                    f"in prediction record."
                )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    overall = (
        empty_metrics()
    )

    by_language = defaultdict(
        lambda: {
            "total": 0,
            "metrics":
                empty_metrics(),
        }
    )

    for record in records:

        language = record[
            "language"
        ]

        result = evaluate_record(
            record
        )

        by_language[
            language
        ][
            "total"
        ] += 1

        for metric in METRIC_NAMES:

            if result[
                metric
            ]:

                overall[
                    metric
                ] += 1

                by_language[
                    language
                ][
                    "metrics"
                ][
                    metric
                ] += 1

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = build_report(
        overall,
        by_language,
        len(records),
    )

    print("\n")
    print(report)

    # --------------------------------------------------------
    # JSON result
    # --------------------------------------------------------

    json_result = {
        "model":
            "Qwen/Qwen3-0.6B + LoRA",

        "evaluation":
            "lora",

        "samples":
            len(records),

        "overall": {},

        "by_language": {},
    }

    for metric in METRIC_NAMES:

        json_result[
            "overall"
        ][metric] = {
            "correct":
                overall[
                    metric
                ],

            "total":
                len(records),

            "accuracy":
                percentage(
                    overall[
                        metric
                    ],
                    len(records),
                ),
        }

    for (
        language,
        language_data,
    ) in by_language.items():

        language_total = (
            language_data[
                "total"
            ]
        )

        json_result[
            "by_language"
        ][language] = {
            "samples":
                language_total,

            "metrics": {},
        }

        for metric in METRIC_NAMES:

            correct = (
                language_data[
                    "metrics"
                ][metric]
            )

            json_result[
                "by_language"
            ][
                language
            ][
                "metrics"
            ][metric] = {
                "correct":
                    correct,

                "total":
                    language_total,

                "accuracy":
                    percentage(
                        correct,
                        language_total,
                    ),
            }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    METRICS_JSON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with METRICS_TXT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )

    print("\nSaved:")

    print(
        f"  {METRICS_JSON_FILE}"
    )

    print(
        f"  {METRICS_TXT_FILE}"
    )


if __name__ == "__main__":
    main()