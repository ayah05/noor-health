import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train_sft.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "training_coverage.json"

LANGUAGE_ORDER = [
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return records


def symptom_count_bucket(count: int) -> str:
    if count >= 4:
        return "4+"
    return str(count)


def analyze_record(record: dict) -> dict:
    target = record["target"]

    symptoms = target.get("symptoms", [])
    medications = target.get("medications", {})
    allergies = target.get("allergies", {})

    symptom_statuses = Counter()

    for symptom in symptoms:
        status = symptom.get("status")

        if status:
            symptom_statuses[status] += 1

    durations = [
        symptom.get("duration")
        for symptom in symptoms
        if symptom.get("duration") is not None
    ]

    absent_count = symptom_statuses.get("absent", 0)
    uncertain_count = symptom_statuses.get("uncertain", 0)

    return {
        "symptom_statuses": symptom_statuses,
        "symptom_count": len(symptoms),
        "duration_count": len(durations),
        "medication_status": medications.get("status"),
        "allergy_status": allergies.get("status"),
        "has_negation": absent_count > 0,
        "has_uncertainty": uncertain_count > 0,
        "has_multiple_durations": len(durations) >= 2,
    }


def empty_stats() -> dict:
    return {
        "samples": 0,
        "symptom_status": Counter(),
        "medication_status": Counter(),
        "allergy_status": Counter(),
        "symptom_count": Counter(),
        "duration_count": Counter(),
        "samples_with_negation": 0,
        "samples_with_uncertainty": 0,
        "samples_with_multiple_durations": 0,
    }


def update_stats(
    stats: dict,
    analysis: dict,
) -> None:
    stats["samples"] += 1

    stats["symptom_status"].update(
        analysis["symptom_statuses"]
    )

    medication_status = analysis["medication_status"]

    if medication_status:
        stats["medication_status"][medication_status] += 1

    allergy_status = analysis["allergy_status"]

    if allergy_status:
        stats["allergy_status"][allergy_status] += 1

    symptom_bucket = symptom_count_bucket(
        analysis["symptom_count"]
    )

    stats["symptom_count"][symptom_bucket] += 1

    duration_count = analysis["duration_count"]

    if duration_count >= 2:
        duration_bucket = "2+"
    else:
        duration_bucket = str(duration_count)

    stats["duration_count"][duration_bucket] += 1

    if analysis["has_negation"]:
        stats["samples_with_negation"] += 1

    if analysis["has_uncertainty"]:
        stats["samples_with_uncertainty"] += 1

    if analysis["has_multiple_durations"]:
        stats["samples_with_multiple_durations"] += 1


def percentage(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return 100.0 * count / total


def print_counter(
    counter: Counter,
    total: int,
    indent: int = 4,
) -> None:
    prefix = " " * indent

    for key in sorted(counter):
        count = counter[key]

        print(
            f"{prefix}{key:<12}"
            f"{count:>6} "
            f"({percentage(count, total):6.2f}%)"
        )


def print_language_stats(
    language: str,
    stats: dict,
) -> None:
    total_samples = stats["samples"]

    print()
    print("=" * 72)
    print(f"{language} - {total_samples} TRAINING SAMPLES")
    print("=" * 72)

    # ============================================================
    # SYMPTOM STATUS
    # ============================================================

    # Symptom statuses count individual symptoms, not samples.
    # Therefore percentages must use the total number of symptoms
    # as denominator.
    total_symptoms = sum(
        stats["symptom_status"].values()
    )

    print(
        f"\nSYMPTOM STATUS "
        f"({total_symptoms} symptoms total)"
    )

    print_counter(
        stats["symptom_status"],
        total_symptoms,
    )

    # ============================================================
    # MEDICATION STATUS
    # ============================================================

    print("\nMEDICATION STATUS")

    print_counter(
        stats["medication_status"],
        total_samples,
    )

    # ============================================================
    # ALLERGY STATUS
    # ============================================================

    print("\nALLERGY STATUS")

    print_counter(
        stats["allergy_status"],
        total_samples,
    )

    # ============================================================
    # NUMBER OF SYMPTOMS PER SAMPLE
    # ============================================================

    print("\nNUMBER OF SYMPTOMS")

    print_counter(
        stats["symptom_count"],
        total_samples,
    )

    # ============================================================
    # NUMBER OF DURATIONS PER SAMPLE
    # ============================================================

    print("\nNUMBER OF EXPLICIT DURATIONS")

    print_counter(
        stats["duration_count"],
        total_samples,
    )

    # ============================================================
    # SEMANTIC COMPLEXITY
    # ============================================================

    print("\nSEMANTIC COMPLEXITY")

    metrics = [
        (
            "Negation",
            stats["samples_with_negation"],
        ),
        (
            "Uncertainty",
            stats["samples_with_uncertainty"],
        ),
        (
            "Multiple durations",
            stats["samples_with_multiple_durations"],
        ),
    ]

    for label, count in metrics:
        print(
            f"    {label:<22}"
            f"{count:>6} "
            f"({percentage(count, total_samples):6.2f}%)"
        )

def counter_to_dict(counter: Counter) -> dict:
    return dict(
        sorted(counter.items())
    )


def serialize_stats(stats: dict) -> dict:
    total = stats["samples"]

    return {
        "samples": total,
        "symptom_status": counter_to_dict(
            stats["symptom_status"]
        ),
        "medication_status": counter_to_dict(
            stats["medication_status"]
        ),
        "allergy_status": counter_to_dict(
            stats["allergy_status"]
        ),
        "symptom_count": counter_to_dict(
            stats["symptom_count"]
        ),
        "duration_count": counter_to_dict(
            stats["duration_count"]
        ),
        "semantic_complexity": {
            "samples_with_negation": (
                stats["samples_with_negation"]
            ),
            "negation_percent": percentage(
                stats["samples_with_negation"],
                total,
            ),
            "samples_with_uncertainty": (
                stats["samples_with_uncertainty"]
            ),
            "uncertainty_percent": percentage(
                stats["samples_with_uncertainty"],
                total,
            ),
            "samples_with_multiple_durations": (
                stats["samples_with_multiple_durations"]
            ),
            "multiple_durations_percent": percentage(
                stats["samples_with_multiple_durations"],
                total,
            ),
        },
    }


def print_comparison_table(
    all_stats: dict,
) -> None:
    print()
    print("=" * 100)
    print("CROSS-LANGUAGE TRAINING COVERAGE")
    print("=" * 100)

    header = (
        f"{'Lang':<8}"
        f"{'N':>7}"
        f"{'Present':>10}"
        f"{'Absent':>10}"
        f"{'Uncertain':>11}"
        f"{'Med none':>11}"
        f"{'Allergy none':>14}"
        f"{'Negation':>11}"
        f"{'Uncertain':>12}"
    )

    print(header)
    print("-" * len(header))

    for language in LANGUAGE_ORDER:
        stats = all_stats.get(language)

        if not stats:
            continue

        total = stats["samples"]

        print(
            f"{language:<8}"
            f"{total:>7}"
            f"{stats['symptom_status'].get('present', 0):>10}"
            f"{stats['symptom_status'].get('absent', 0):>10}"
            f"{stats['symptom_status'].get('uncertain', 0):>11}"
            f"{stats['medication_status'].get('none', 0):>11}"
            f"{stats['allergy_status'].get('none', 0):>14}"
            f"{stats['samples_with_negation']:>11}"
            f"{stats['samples_with_uncertainty']:>12}"
        )


def main() -> None:
    print("=" * 72)
    print("NOOR HEALTH - TRAINING COVERAGE AUDIT")
    print("=" * 72)

    print(f"\nTraining file:\n{TRAIN_FILE}")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    records = load_jsonl(TRAIN_FILE)

    print(f"\nTraining samples: {len(records)}")

    stats_by_language = defaultdict(empty_stats)

    for record in records:
        language = record.get("language")

        if not language:
            raise ValueError(
                f"Record missing language: {record.get('case_id')}"
            )

        analysis = analyze_record(record)

        update_stats(
            stats_by_language[language],
            analysis,
        )

    print_comparison_table(stats_by_language)

    for language in LANGUAGE_ORDER:
        if language in stats_by_language:
            print_language_stats(
                language,
                stats_by_language[language],
            )

    unknown_languages = (
        set(stats_by_language)
        - set(LANGUAGE_ORDER)
    )

    for language in sorted(unknown_languages):
        print_language_stats(
            language,
            stats_by_language[language],
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "training_file": str(TRAIN_FILE),
        "total_samples": len(records),
        "languages": {
            language: serialize_stats(stats)
            for language, stats
            in sorted(stats_by_language.items())
        },
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)
    print(f"\nSaved:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()