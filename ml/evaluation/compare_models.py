import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate_predictions import (
    DISPLAY_NAMES,
    METRICS,
    evaluate_sample,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_COMPARISON_METRIC = "core_exact"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(
    path: Path,
) -> dict[tuple[str, str], dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found:\n{path}"
        )

    records = {}

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
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number} in:\n{path}"
                ) from exc

            case_id = record.get("case_id")
            language = record.get("language")

            if not case_id:
                raise ValueError(
                    f"Missing case_id at line "
                    f"{line_number} in:\n{path}"
                )

            if not language:
                raise ValueError(
                    f"Missing language at line "
                    f"{line_number} in:\n{path}"
                )

            key = (
                case_id,
                language,
            )

            if key in records:
                raise ValueError(
                    "Duplicate sample found:\n"
                    f"case_id={case_id}, "
                    f"language={language}\n"
                    f"File: {path}"
                )

            records[key] = record

    if not records:
        raise ValueError(
            f"No predictions found in:\n{path}"
        )

    return records


# ============================================================
# TARGET CONSISTENCY
# ============================================================

def verify_same_targets(
    base_records: dict,
    lora_records: dict,
):

    base_keys = set(
        base_records.keys()
    )

    lora_keys = set(
        lora_records.keys()
    )

    if base_keys != lora_keys:

        only_base = (
            base_keys - lora_keys
        )

        only_lora = (
            lora_keys - base_keys
        )

        raise ValueError(
            "Base and LoRA files do not contain "
            "the same samples.\n"
            f"Only in Base: {len(only_base)}\n"
            f"Only in LoRA: {len(only_lora)}"
        )

    for key in sorted(base_keys):

        base = base_records[key]
        lora = lora_records[key]

        if (
            base.get("target")
            != lora.get("target")
        ):
            raise ValueError(
                "Target mismatch for "
                f"{key[0]} / {key[1]}"
            )

        if (
            base.get("utterance")
            != lora.get("utterance")
        ):
            raise ValueError(
                "Utterance mismatch for "
                f"{key[0]} / {key[1]}"
            )


# ============================================================
# COMPARISON CATEGORY
# ============================================================

def get_category(
    base_correct: bool,
    lora_correct: bool,
) -> str:

    if base_correct and lora_correct:
        return "both_correct"

    if (
        not base_correct
        and lora_correct
    ):
        return "lora_improved"

    if (
        base_correct
        and not lora_correct
    ):
        return "lora_regressed"

    return "both_wrong"


# ============================================================
# EMPTY COUNTERS
# ============================================================

def empty_categories() -> dict[str, int]:

    return {
        "both_correct": 0,
        "lora_improved": 0,
        "lora_regressed": 0,
        "both_wrong": 0,
    }


# ============================================================
# PRINT HELPERS
# ============================================================

def percentage(
    value: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return (
        value / total * 100
    )


def print_category_table(
    counts: dict,
    total: int,
):

    labels = {
        "both_correct":
            "Both correct",

        "lora_improved":
            "LoRA improved",

        "lora_regressed":
            "LoRA regressed",

        "both_wrong":
            "Both wrong",
    }

    for category in [
        "both_correct",
        "lora_improved",
        "lora_regressed",
        "both_wrong",
    ]:

        count = counts[
            category
        ]

        print(
            f"{labels[category]:<24}"
            f"{count:>5} / "
            f"{total:<5}"
            f"{percentage(count, total):>8.2f}%"
        )


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare_models(
    base_file: Path,
    lora_file: Path,
    output_dir: Path,
    primary_metric: str,
):

    if primary_metric not in METRICS:
        raise ValueError(
            f"Unknown metric: "
            f"{primary_metric}\n"
            f"Available metrics: "
            f"{', '.join(METRICS)}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    print(
        "Loading predictions..."
    )

    base_records = load_predictions(
        base_file
    )

    lora_records = load_predictions(
        lora_file
    )

    verify_same_targets(
        base_records,
        lora_records,
    )

    keys = sorted(
        base_records.keys()
    )

    total = len(keys)

    print(
        f"Matched samples: {total}"
    )

    if total != 700:
        print(
            "WARNING: Expected 700 samples."
        )

    # ========================================================
    # COUNTERS
    # ========================================================

    overall_by_metric = {
        metric: empty_categories()
        for metric in METRICS
    }

    by_language = defaultdict(
        lambda: {
            metric: empty_categories()
            for metric in METRICS
        }
    )

    language_totals = defaultdict(
        int
    )

    sample_comparisons = []

    # ========================================================
    # COMPARE SAMPLE BY SAMPLE
    # ========================================================

    for key in keys:

        base_record = (
            base_records[key]
        )

        lora_record = (
            lora_records[key]
        )

        base_metrics = evaluate_sample(
            base_record
        )

        lora_metrics = evaluate_sample(
            lora_record
        )

        case_id = key[0]
        language = key[1]

        language_totals[
            language
        ] += 1

        categories = {}

        for metric in METRICS:

            category = get_category(
                base_metrics[metric],
                lora_metrics[metric],
            )

            categories[
                metric
            ] = category

            overall_by_metric[
                metric
            ][
                category
            ] += 1

            by_language[
                language
            ][
                metric
            ][
                category
            ] += 1

        sample_comparisons.append(
            {
                "case_id":
                    case_id,

                "language":
                    language,

                "utterance":
                    base_record.get(
                        "utterance"
                    ),

                "target":
                    base_record.get(
                        "target"
                    ),

                "base_prediction":
                    base_record.get(
                        "prediction"
                    ),

                "lora_prediction":
                    lora_record.get(
                        "prediction"
                    ),

                "base_metrics":
                    base_metrics,

                "lora_metrics":
                    lora_metrics,

                "categories":
                    categories,
            }
        )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "NOOR HEALTH - BASE VS LORA"
    )

    print(
        "=" * 76
    )

    print(
        f"\nSamples: {total}"
    )

    print(
        f"Primary metric: "
        f"{DISPLAY_NAMES[primary_metric]}"
    )

    # ========================================================
    # PRIMARY METRIC
    # ========================================================

    print(
        "\nPRIMARY METRIC"
    )

    print(
        "-" * 76
    )

    primary_counts = (
        overall_by_metric[
            primary_metric
        ]
    )

    print_category_table(
        primary_counts,
        total,
    )

    base_correct = (
        primary_counts[
            "both_correct"
        ]
        +
        primary_counts[
            "lora_regressed"
        ]
    )

    lora_correct = (
        primary_counts[
            "both_correct"
        ]
        +
        primary_counts[
            "lora_improved"
        ]
    )

    base_accuracy = percentage(
        base_correct,
        total,
    )

    lora_accuracy = percentage(
        lora_correct,
        total,
    )

    delta = (
        lora_accuracy
        - base_accuracy
    )

    print(
        "\nAccuracy"
    )

    print(
        f"{'Base':<24}"
        f"{base_correct:>5} / "
        f"{total:<5}"
        f"{base_accuracy:>8.2f}%"
    )

    print(
        f"{'LoRA':<24}"
        f"{lora_correct:>5} / "
        f"{total:<5}"
        f"{lora_accuracy:>8.2f}%"
    )

    print(
        f"{'Delta':<24}"
        f"{delta:>18.2f} pp"
    )

    # ========================================================
    # ALL METRICS SUMMARY
    # ========================================================

    print(
        "\nALL METRICS"
    )

    print(
        "=" * 76
    )

    print(
        f"{'Metric':<34}"
        f"{'Base':>10}"
        f"{'LoRA':>10}"
        f"{'Delta':>12}"
    )

    print(
        "-" * 76
    )

    metric_summary = {}

    for metric in METRICS:

        counts = (
            overall_by_metric[
                metric
            ]
        )

        metric_base_correct = (
            counts[
                "both_correct"
            ]
            +
            counts[
                "lora_regressed"
            ]
        )

        metric_lora_correct = (
            counts[
                "both_correct"
            ]
            +
            counts[
                "lora_improved"
            ]
        )

        metric_base_accuracy = (
            percentage(
                metric_base_correct,
                total,
            )
        )

        metric_lora_accuracy = (
            percentage(
                metric_lora_correct,
                total,
            )
        )

        metric_delta = (
            metric_lora_accuracy
            - metric_base_accuracy
        )

        metric_summary[
            metric
        ] = {
            "base_correct":
                metric_base_correct,

            "lora_correct":
                metric_lora_correct,

            "base_accuracy":
                metric_base_accuracy,

            "lora_accuracy":
                metric_lora_accuracy,

            "delta_percentage_points":
                metric_delta,

            **counts,
        }

        print(
            f"{DISPLAY_NAMES[metric]:<34}"
            f"{metric_base_accuracy:>9.2f}%"
            f"{metric_lora_accuracy:>9.2f}%"
            f"{metric_delta:>10.2f} pp"
        )

    # ========================================================
    # PRIMARY METRIC BY LANGUAGE
    # ========================================================

    print(
        "\nPRIMARY METRIC BY LANGUAGE"
    )

    print(
        "=" * 76
    )

    print(
        f"{'Language':<12}"
        f"{'Base':>9}"
        f"{'LoRA':>9}"
        f"{'Delta':>11}"
        f"{'Improved':>11}"
        f"{'Regressed':>12}"
        f"{'Both wrong':>12}"
    )

    print(
        "-" * 76
    )

    language_summary = {}

    for language in sorted(
        language_totals
    ):

        language_total = (
            language_totals[
                language
            ]
        )

        counts = (
            by_language[
                language
            ][
                primary_metric
            ]
        )

        language_base_correct = (
            counts[
                "both_correct"
            ]
            +
            counts[
                "lora_regressed"
            ]
        )

        language_lora_correct = (
            counts[
                "both_correct"
            ]
            +
            counts[
                "lora_improved"
            ]
        )

        language_base_accuracy = (
            percentage(
                language_base_correct,
                language_total,
            )
        )

        language_lora_accuracy = (
            percentage(
                language_lora_correct,
                language_total,
            )
        )

        language_delta = (
            language_lora_accuracy
            - language_base_accuracy
        )

        language_summary[
            language
        ] = {
            "total":
                language_total,

            "base_correct":
                language_base_correct,

            "lora_correct":
                language_lora_correct,

            "base_accuracy":
                language_base_accuracy,

            "lora_accuracy":
                language_lora_accuracy,

            "delta_percentage_points":
                language_delta,

            **counts,
        }

        print(
            f"{language:<12}"
            f"{language_base_accuracy:>8.2f}%"
            f"{language_lora_accuracy:>8.2f}%"
            f"{language_delta:>9.2f} pp"
            f"{counts['lora_improved']:>11}"
            f"{counts['lora_regressed']:>12}"
            f"{counts['both_wrong']:>12}"
        )

    # ========================================================
    # REGRESSIONS
    # ========================================================

    regressions = [
        sample
        for sample
        in sample_comparisons
        if (
            sample[
                "categories"
            ][
                primary_metric
            ]
            == "lora_regressed"
        )
    ]

    improvements = [
        sample
        for sample
        in sample_comparisons
        if (
            sample[
                "categories"
            ][
                primary_metric
            ]
            == "lora_improved"
        )
    ]

    both_wrong = [
        sample
        for sample
        in sample_comparisons
        if (
            sample[
                "categories"
            ][
                primary_metric
            ]
            == "both_wrong"
        )
    ]

    print(
        "\nPRIMARY METRIC TRANSITIONS"
    )

    print(
        "-" * 76
    )

    print(
        f"LoRA improvements: "
        f"{len(improvements)}"
    )

    print(
        f"LoRA regressions:  "
        f"{len(regressions)}"
    )

    print(
        f"Both wrong:        "
        f"{len(both_wrong)}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_dir
        / "base_vs_lora_summary.json"
    )

    samples_file = (
        output_dir
        / "base_vs_lora_samples.jsonl"
    )

    regressions_file = (
        output_dir
        / "base_vs_lora_regressions.jsonl"
    )

    improvements_file = (
        output_dir
        / "base_vs_lora_improvements.jsonl"
    )

    both_wrong_file = (
        output_dir
        / "base_vs_lora_both_wrong.jsonl"
    )

    summary = {
        "base_file":
            str(base_file),

        "lora_file":
            str(lora_file),

        "total_samples":
            total,

        "primary_metric":
            primary_metric,

        "primary_metric_display_name":
            DISPLAY_NAMES[
                primary_metric
            ],

        "overall_metrics":
            metric_summary,

        "primary_metric_by_language":
            language_summary,
    }

    summary_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def write_jsonl(
        path: Path,
        rows: list[dict],
    ):

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

    write_jsonl(
        samples_file,
        sample_comparisons,
    )

    write_jsonl(
        regressions_file,
        regressions,
    )

    write_jsonl(
        improvements_file,
        improvements,
    )

    write_jsonl(
        both_wrong_file,
        both_wrong,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        f"  {summary_file}"
    )

    print(
        f"  {samples_file}"
    )

    print(
        f"  {improvements_file}"
    )

    print(
        f"  {regressions_file}"
    )

    print(
        f"  {both_wrong_file}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Noor Health Base and "
            "LoRA predictions sample by sample."
        )
    )

    parser.add_argument(
        "--base",
        required=True,
        help=(
            "Path to Base model "
            "predictions JSONL."
        ),
    )

    parser.add_argument(
        "--lora",
        required=True,
        help=(
            "Path to LoRA model "
            "predictions JSONL."
        ),
    )

    parser.add_argument(
        "--metric",
        default=(
            DEFAULT_COMPARISON_METRIC
        ),
        choices=METRICS,
        help=(
            "Primary metric used for "
            "sample-level comparison."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/evaluation/comparison"
        ),
        help=(
            "Directory for comparison "
            "results."
        ),
    )

    args = parser.parse_args()

    compare_models(
        base_file=Path(
            args.base
        ).resolve(),

        lora_file=Path(
            args.lora
        ).resolve(),

        output_dir=Path(
            args.output_dir
        ).resolve(),

        primary_metric=(
            args.metric
        ),
    )


if __name__ == "__main__":
    main()