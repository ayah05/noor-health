import json
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"

VALIDATION_FILE = (
    RESULTS_DIR
    / "semantic_validation.jsonl"
)

OUTPUT_JSONL = (
    RESULTS_DIR
    / "semantic_failures.jsonl"
)

OUTPUT_TXT = (
    RESULTS_DIR
    / "semantic_failures_report.txt"
)


# ============================================================
# HELPERS
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


def get_error_types(
    record: dict,
    key: str,
) -> list[str]:

    value = record.get(key, [])

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return value

    return [str(value)]


def format_json(value) -> str:

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NOOR HEALTH - SEMANTIC FAILURE ANALYSIS")
    print("=" * 60)

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    records = load_jsonl(
        VALIDATION_FILE
    )

    print(
        f"\nValidation results loaded: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # FIND ALL FAILED SAMPLES
    # --------------------------------------------------------

    failures = []

    for record in records:

        semantic_passed = record.get(
            "semantic_passed",
            False,
        )

        language_passed = record.get(
            "language_passed",
            False,
        )

        if (
            not semantic_passed
            or not language_passed
        ):
            failures.append(record)

    print(
        f"Failed either check: "
        f"{len(failures)}"
    )

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    split_counter = Counter()

    language_counter = Counter()

    semantic_error_counter = Counter()

    language_error_counter = Counter()

    semantic_only = 0
    language_only = 0
    both_failed = 0

    for record in failures:

        split_counter[
            record.get("split", "unknown")
        ] += 1

        language_counter[
            record.get("language", "unknown")
        ] += 1

        semantic_passed = record.get(
            "semantic_passed",
            False,
        )

        language_passed = record.get(
            "language_passed",
            False,
        )

        if (
            not semantic_passed
            and not language_passed
        ):
            both_failed += 1

        elif not semantic_passed:
            semantic_only += 1

        elif not language_passed:
            language_only += 1

        for error in get_error_types(
            record,
            "semantic_error_types",
        ):
            semantic_error_counter[
                error
            ] += 1

        for error in get_error_types(
            record,
            "language_error_types",
        ):
            language_error_counter[
                error
            ] += 1

    # --------------------------------------------------------
    # SORT FAILURES
    # --------------------------------------------------------

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    failures.sort(
        key=lambda record: (
            split_order.get(
                record.get(
                    "split",
                    "unknown",
                ),
                99,
            ),
            record.get(
                "language",
                "",
            ),
            record.get(
                "case_id",
                "",
            ),
        )
    )

    # --------------------------------------------------------
    # SAVE JSONL
    # --------------------------------------------------------

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in failures:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # BUILD HUMAN-READABLE REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "NOOR HEALTH - SEMANTIC FAILURE REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Total validation results: "
        f"{len(records)}"
    )

    report.append(
        f"Failed either check: "
        f"{len(failures)}"
    )

    report.append(
        f"Semantic only: "
        f"{semantic_only}"
    )

    report.append(
        f"Language only: "
        f"{language_only}"
    )

    report.append(
        f"Both failed: "
        f"{both_failed}"
    )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    report.append("")
    report.append("FAILURES BY SPLIT")
    report.append("-" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        report.append(
            f"{split:<12} "
            f"{split_counter.get(split, 0)}"
        )

    # --------------------------------------------------------
    # LANGUAGES
    # --------------------------------------------------------

    report.append("")
    report.append("FAILURES BY LANGUAGE")
    report.append("-" * 70)

    for language, count in sorted(
        language_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):

        report.append(
            f"{language:<12} "
            f"{count}"
        )

    # --------------------------------------------------------
    # SEMANTIC ERRORS
    # --------------------------------------------------------

    report.append("")
    report.append("SEMANTIC ERROR TYPES")
    report.append("-" * 70)

    if semantic_error_counter:

        for error, count in (
            semantic_error_counter
            .most_common()
        ):

            report.append(
                f"{error:<35} "
                f"{count}"
            )

    else:

        report.append(
            "None"
        )

    # --------------------------------------------------------
    # LANGUAGE ERRORS
    # --------------------------------------------------------

    report.append("")
    report.append("LANGUAGE ERROR TYPES")
    report.append("-" * 70)

    if language_error_counter:

        for error, count in (
            language_error_counter
            .most_common()
        ):

            report.append(
                f"{error:<35} "
                f"{count}"
            )

    else:

        report.append(
            "None"
        )

    # --------------------------------------------------------
    # INDIVIDUAL FAILURES
    # --------------------------------------------------------

    report.append("")
    report.append("")
    report.append(
        "=" * 70
    )

    report.append(
        "INDIVIDUAL FAILED SAMPLES"
    )

    report.append(
        "=" * 70
    )

    for index, record in enumerate(
        failures,
        start=1,
    ):

        report.append("")
        report.append(
            "#" * 70
        )

        report.append(
            f"FAILURE {index}/{len(failures)}"
        )

        report.append(
            "#" * 70
        )

        report.append("")

        report.append(
            f"Case ID:  "
            f"{record.get('case_id')}"
        )

        report.append(
            f"Split:    "
            f"{record.get('split')}"
        )

        report.append(
            f"Language: "
            f"{record.get('language')}"
        )

        report.append("")

        report.append(
            f"Semantic passed: "
            f"{record.get('semantic_passed')}"
        )

        report.append(
            f"Language passed: "
            f"{record.get('language_passed')}"
        )

        report.append("")

        semantic_errors = (
            get_error_types(
                record,
                "semantic_error_types",
            )
        )

        language_errors = (
            get_error_types(
                record,
                "language_error_types",
            )
        )

        report.append(
            "Semantic errors: "
            + (
                ", ".join(
                    semantic_errors
                )
                if semantic_errors
                else "None"
            )
        )

        report.append(
            "Language errors: "
            + (
                ", ".join(
                    language_errors
                )
                if language_errors
                else "None"
            )
        )

        # ----------------------------------------------------
        # UTTERANCE
        # ----------------------------------------------------

        report.append("")
        report.append("UTTERANCE")
        report.append("-" * 70)

        report.append(
            str(
                record.get(
                    "utterance",
                    "<not stored>",
                )
            )
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        report.append("")
        report.append("TARGET")
        report.append("-" * 70)

        target = record.get(
            "target",
            "<not stored>",
        )

        if isinstance(
            target,
            (dict, list),
        ):

            report.append(
                format_json(target)
            )

        else:

            report.append(
                str(target)
            )

        # ----------------------------------------------------
        # JUDGE REASON
        # ----------------------------------------------------

        report.append("")
        report.append("JUDGE REASON")
        report.append("-" * 70)

        reason = (
            record.get("reason")
            or record.get(
                "judge_reason"
            )
            or record.get(
                "explanation"
            )
            or "<not stored>"
        )

        report.append(
            str(reason)
        )

        # ----------------------------------------------------
        # RAW JUDGE RESULT
        # ----------------------------------------------------

        if "judge_result" in record:

            report.append("")
            report.append(
                "RAW JUDGE RESULT"
            )

            report.append(
                "-" * 70
            )

            report.append(
                format_json(
                    record[
                        "judge_result"
                    ]
                )
            )

    # --------------------------------------------------------
    # WRITE TXT
    # --------------------------------------------------------

    report_text = "\n".join(
        report
    )

    OUTPUT_TXT.write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\nFailures by split:")

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:<12}"
            f"{split_counter.get(split, 0)}"
        )

    print(
        "\nFailures by language:"
    )

    for language, count in sorted(
        language_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):

        print(
            f"  {language:<12}"
            f"{count}"
        )

    print(
        "\nSemantic error types:"
    )

    for error, count in (
        semantic_error_counter
        .most_common()
    ):

        print(
            f"  {error:<35}"
            f"{count}"
        )

    print(
        "\nLanguage error types:"
    )

    for error, count in (
        language_error_counter
        .most_common()
    ):

        print(
            f"  {error:<35}"
            f"{count}"
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_JSONL}"
    )

    print(
        f"  {OUTPUT_TXT}"
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()