import json
from collections import defaultdict
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

FAILURE_ANALYSIS_FILE = (
    ROOT
    / "results"
    / "language_failure_analysis.json"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "multilingual_case_audit.txt"
)

OUTPUT_JSON_FILE = (
    ROOT
    / "results"
    / "multilingual_case_audit.json"
)

LANGUAGE_ORDER = [
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
]

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "sw": "Swahili",
}

# Number of cases we want to inspect manually.
ONLY_HINDI_LIMIT = 22
ONLY_SWAHILI_LIMIT = 13
HINDI_SWAHILI_LIMIT = 10
CONTROL_LIMIT = 10


# ============================================================
# NORMALIZATION
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

    items = group.get(
        "items",
        [],
    )

    if not isinstance(items, list):
        return group

    normalized_items = []

    for item in items:

        if isinstance(item, str):
            normalized_items.append(
                normalize_string(item)
            )
        else:
            normalized_items.append(
                item
            )

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

        text = "\n".join(
            lines
        ).strip()

    # Direct JSON
    try:

        parsed = json.loads(
            text
        )

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Try outermost JSON object
    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:

            parsed = json.loads(
                text[
                    start:end + 1
                ]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# SYMPTOMS
# ============================================================

def build_symptom_map(data):

    symptoms = data.get(
        "symptoms",
        []
    )

    if not isinstance(symptoms, list):
        return {}

    result = defaultdict(
        list
    )

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

        result[name].append(
            symptom
        )

    return dict(
        result
    )


# ============================================================
# CORE FAILURE ANALYSIS
# ============================================================

def analyze_prediction(
    prediction,
    target,
):

    failures = []

    if prediction is None:

        return [
            {
                "type":
                    "invalid_json"
            }
        ]

    # --------------------------------------------------------
    # Chief complaint
    # --------------------------------------------------------

    predicted_chief = (
        normalize_string(
            prediction.get(
                "chief_complaint"
            )
        )
    )

    target_chief = (
        normalize_string(
            target.get(
                "chief_complaint"
            )
        )
    )

    if (
        predicted_chief
        != target_chief
    ):

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

    predicted_map = (
        build_symptom_map(
            prediction
        )
    )

    target_map = (
        build_symptom_map(
            target
        )
    )

    predicted_names = set(
        predicted_map
    )

    target_names = set(
        target_map
    )

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
        entries,
    ) in predicted_map.items():

        if len(entries) > 1:

            failures.append(
                {
                    "type":
                        "duplicate_symptom",

                    "symptom":
                        symptom_name,

                    "count":
                        len(entries),
                }
            )

    # --------------------------------------------------------
    # Shared symptoms
    # --------------------------------------------------------

    for symptom_name in sorted(
        target_names
        & predicted_names
    ):

        target_symptom = (
            target_map[
                symptom_name
            ][0]
        )

        predicted_symptom = (
            predicted_map[
                symptom_name
            ][0]
        )

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
    # Medications
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
    # Allergies
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
# FORMATTING
# ============================================================

def pretty_json(value):

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def format_target_summary(
    target,
):

    lines = []

    lines.append(
        f"Chief complaint: "
        f"{target.get('chief_complaint')}"
    )

    lines.append(
        "Symptoms:"
    )

    for symptom in target.get(
        "symptoms",
        []
    ):

        duration = symptom.get(
            "duration"
        )

        if duration is None:

            duration_text = "null"

        else:

            duration_text = (
                f"{duration.get('value')} "
                f"{duration.get('unit')}"
            )

        lines.append(
            "  - "
            f"{symptom.get('name')} | "
            f"{symptom.get('status')} | "
            f"{duration_text}"
        )

    medications = target.get(
        "medications",
        {}
    )

    lines.append(
        "Medications: "
        f"{medications.get('status')} | "
        f"{medications.get('items')}"
    )

    allergies = target.get(
        "allergies",
        {}
    )

    lines.append(
        "Allergies: "
        f"{allergies.get('status')} | "
        f"{allergies.get('items')}"
    )

    lines.append(
        "Missing information: "
        f"{target.get('missing_information')}"
    )

    return "\n".join(
        lines
    )


def format_errors(
    failures,
):

    if not failures:
        return "None"

    lines = []

    for failure in failures:

        lines.append(
            "- "
            + json.dumps(
                failure,
                ensure_ascii=False,
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path):

    records = []

    with path.open(
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
# MAIN
# ============================================================

def main():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    if not FAILURE_ANALYSIS_FILE.exists():

        raise FileNotFoundError(
            "Run analyze_language_failures.py first.\n"
            f"Missing:\n"
            f"{FAILURE_ANALYSIS_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_jsonl(
        PREDICTIONS_FILE
    )

    with FAILURE_ANALYSIS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        failure_summary = (
            json.load(
                file
            )
        )

    print(
        "=" * 76
    )

    print(
        "NOOR HEALTH - MULTILINGUAL CASE AUDIT"
    )

    print(
        "=" * 76
    )

    print(
        f"\nLoaded predictions: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Organize records by case
    # --------------------------------------------------------

    by_case = defaultdict(
        dict
    )

    for record in records:

        prediction = (
            parse_prediction(
                record.get(
                    "prediction"
                )
            )
        )

        failures = (
            analyze_prediction(
                prediction,
                record[
                    "target"
                ],
            )
        )

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
            "_correct"
        ] = len(
            failures
        ) == 0

        by_case[
            record["case_id"]
        ][
            record["language"]
        ] = enriched

    # --------------------------------------------------------
    # Get categories from previous audit
    # --------------------------------------------------------

    cross = failure_summary[
        "cross_language"
    ]

    only_hindi_cases = (
        cross.get(
            "only_hindi_failed_case_ids",
            [],
        )
    )

    only_swahili_cases = (
        cross.get(
            "only_swahili_failed_case_ids",
            [],
        )
    )

    hi_sw_cases = (
        cross.get(
            "hindi_swahili_specific_case_ids",
            [],
        )
    )

    # --------------------------------------------------------
    # Control cases
    #
    # Cases where every language is correct.
    # --------------------------------------------------------

    control_cases = []

    for (
        case_id,
        language_records,
    ) in by_case.items():

        if (
            len(language_records)
            == len(LANGUAGE_ORDER)
            and all(
                language_records[
                    language
                ][
                    "_correct"
                ]
                for language
                in LANGUAGE_ORDER
                if language
                in language_records
            )
        ):

            control_cases.append(
                case_id
            )

    # Keep deterministic ordering
    only_hindi_cases = sorted(
        only_hindi_cases
    )[
        :ONLY_HINDI_LIMIT
    ]

    only_swahili_cases = sorted(
        only_swahili_cases
    )[
        :ONLY_SWAHILI_LIMIT
    ]

    hi_sw_cases = sorted(
        hi_sw_cases
    )[
        :HINDI_SWAHILI_LIMIT
    ]

    control_cases = sorted(
        control_cases
    )[
        :CONTROL_LIMIT
    ]

    categories = [
        (
            "ONLY HINDI FAILED",
            only_hindi_cases,
        ),
        (
            "ONLY SWAHILI FAILED",
            only_swahili_cases,
        ),
        (
            "HINDI + SWAHILI FAILED",
            hi_sw_cases,
        ),
        (
            "CONTROL - ALL LANGUAGES CORRECT",
            control_cases,
        ),
    ]

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        "\nSELECTED CASES"
    )

    print(
        "-" * 76
    )

    print(
        f"Only Hindi failed: "
        f"{len(only_hindi_cases)}"
    )

    print(
        f"Only Swahili failed: "
        f"{len(only_swahili_cases)}"
    )

    print(
        f"Hindi + Swahili failed: "
        f"{len(hi_sw_cases)}"
    )

    print(
        f"Control cases: "
        f"{len(control_cases)}"
    )

    # --------------------------------------------------------
    # Build TXT report
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 76
    )

    report.append(
        "NOOR HEALTH - MULTILINGUAL HUMAN DATA AUDIT"
    )

    report.append(
        "=" * 76
    )

    report.append(
        ""
    )

    report.append(
        "Purpose:"
    )

    report.append(
        "Compare the same clinical case across languages "
        "to identify language-specific data or model failures."
    )

    report.append(
        ""
    )

    report.append(
        "IMPORTANT:"
    )

    report.append(
        "This report does not automatically judge whether a translation "
        "is natural or clinically correct. That requires human review."
    )

    # JSON export structure
    json_export = {
        "categories": {}
    }

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    for (
        category_name,
        case_ids,
    ) in categories:

        report.append("")
        report.append(
            "=" * 76
        )
        report.append(
            category_name
        )
        report.append(
            "=" * 76
        )

        json_export[
            "categories"
        ][category_name] = []

        for case_id in case_ids:

            language_records = (
                by_case.get(
                    case_id,
                    {}
                )
            )

            if not language_records:
                continue

            # All translations should share target.
            first_record = next(
                iter(
                    language_records.values()
                )
            )

            target = first_record[
                "target"
            ]

            report.append("")
            report.append(
                "#" * 76
            )

            report.append(
                f"CASE: {case_id}"
            )

            report.append(
                "#" * 76
            )

            report.append("")
            report.append(
                "CANONICAL TARGET"
            )

            report.append(
                "-" * 76
            )

            report.append(
                format_target_summary(
                    target
                )
            )

            case_export = {
                "case_id":
                    case_id,

                "category":
                    category_name,

                "target":
                    target,

                "languages":
                    {},
            }

            # -----------------------------------------------
            # Languages
            # -----------------------------------------------

            for language in LANGUAGE_ORDER:

                if (
                    language
                    not in language_records
                ):
                    continue

                record = (
                    language_records[
                        language
                    ]
                )

                correct = record[
                    "_correct"
                ]

                status = (
                    "CORRECT"
                    if correct
                    else "FAILED"
                )

                report.append("")
                report.append(
                    "-" * 76
                )

                report.append(
                    f"{LANGUAGE_NAMES[language].upper()} "
                    f"[{language}] — {status}"
                )

                report.append(
                    "-" * 76
                )

                report.append(
                    "UTTERANCE:"
                )

                report.append(
                    record.get(
                        "utterance",
                        "<utterance unavailable>",
                    )
                )

                report.append("")
                report.append(
                    "ERRORS:"
                )

                report.append(
                    format_errors(
                        record[
                            "_failures"
                        ]
                    )
                )

                # For failed languages include full prediction.
                if not correct:

                    report.append("")
                    report.append(
                        "PREDICTION:"
                    )

                    report.append(
                        pretty_json(
                            record[
                                "_parsed_prediction"
                            ]
                        )
                        if record[
                            "_parsed_prediction"
                        ] is not None
                        else str(
                            record.get(
                                "prediction"
                            )
                        )
                    )

                case_export[
                    "languages"
                ][language] = {
                    "language_name":
                        LANGUAGE_NAMES[
                            language
                        ],

                    "correct":
                        correct,

                    "utterance":
                        record.get(
                            "utterance"
                        ),

                    "failures":
                        record[
                            "_failures"
                        ],

                    "prediction":
                        record[
                            "_parsed_prediction"
                        ],
                }

            json_export[
                "categories"
            ][
                category_name
            ].append(
                case_export
            )

    # --------------------------------------------------------
    # Save TXT
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_JSON_FILE.write_text(
        json.dumps(
            json_export,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nAudit created:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {OUTPUT_JSON_FILE}"
    )

    print(
        "\nOpen multilingual_case_audit.txt "
        "and inspect the cases side by side."
    )


if __name__ == "__main__":
    main()