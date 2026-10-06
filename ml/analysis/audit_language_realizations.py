import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_sft.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "language_realizations.json"
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


# ============================================================
# BASIC IO
# ============================================================

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
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

    return records


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def tokenize(text: str) -> list[str]:
    """
    Simple Unicode-aware tokenization.

    This is intentionally language-agnostic.
    We do not want to introduce different NLP libraries
    or tokenizers for different languages in this audit.
    """

    text = normalize_text(text)

    return re.findall(
        r"\b[^\W_]+\b",
        text,
        flags=re.UNICODE,
    )


# ============================================================
# CHARACTER SCRIPT HELPERS
# ============================================================

def contains_devanagari(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0900-\u097F]",
            text,
        )
    )


def contains_arabic_script(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            text,
        )
    )


def contains_ascii_digit(text: str) -> bool:
    return bool(
        re.search(
            r"[0-9]",
            text,
        )
    )


def contains_devanagari_digit(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0966-\u096F]",
            text,
        )
    )


def contains_arabic_indic_digit(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0660-\u0669]",
            text,
        )
    )


# ============================================================
# TARGET FEATURES
# ============================================================

def get_target_features(record: dict) -> dict:
    target = record["target"]

    symptoms = target.get(
        "symptoms",
        [],
    )

    medications = target.get(
        "medications",
        {},
    )

    allergies = target.get(
        "allergies",
        {},
    )

    statuses = [
        symptom.get("status")
        for symptom in symptoms
    ]

    durations = [
        symptom.get("duration")
        for symptom in symptoms
        if symptom.get("duration") is not None
    ]

    return {
        "has_absent_symptom": (
            "absent" in statuses
        ),
        "has_uncertain_symptom": (
            "uncertain" in statuses
        ),
        "medication_status": (
            medications.get("status")
        ),
        "allergy_status": (
            allergies.get("status")
        ),
        "symptom_count": len(symptoms),
        "duration_count": len(durations),
    }


# ============================================================
# LEXICAL DIVERSITY
# ============================================================

def type_token_ratio(
    tokens: list[str],
) -> float:
    if not tokens:
        return 0.0

    return (
        len(set(tokens))
        / len(tokens)
    )


def normalized_entropy(
    counter: Counter,
) -> float:
    """
    Normalized Shannon entropy.

    0 = one item dominates completely
    1 = maximally diverse distribution

    Useful as a rough diversity indicator.
    """

    total = sum(counter.values())

    if total == 0:
        return 0.0

    if len(counter) <= 1:
        return 0.0

    entropy = 0.0

    for count in counter.values():
        probability = count / total

        entropy -= (
            probability
            * math.log(probability)
        )

    max_entropy = math.log(
        len(counter)
    )

    return entropy / max_entropy


# ============================================================
# N-GRAMS
# ============================================================

def get_ngrams(
    tokens: list[str],
    n: int,
) -> list[str]:

    if len(tokens) < n:
        return []

    return [
        " ".join(
            tokens[index:index + n]
        )
        for index in range(
            len(tokens) - n + 1
        )
    ]


# ============================================================
# LANGUAGE STATS
# ============================================================

def empty_language_stats() -> dict:
    return {
        "samples": 0,

        "utterances": [],
        "normalized_utterances": [],

        "all_tokens": [],

        "token_counts": Counter(),
        "bigram_counts": Counter(),
        "trigram_counts": Counter(),

        "utterance_lengths_words": [],
        "utterance_lengths_chars": [],

        "semantic_groups": {
            "negation": [],
            "uncertainty": [],
            "medications_none": [],
            "medications_unknown": [],
            "medications_reported": [],
            "allergies_none": [],
            "allergies_unknown": [],
            "allergies_reported": [],
            "multiple_symptoms": [],
            "multiple_durations": [],
        },

        "digit_types": Counter(),

        "contains_devanagari": 0,
        "contains_arabic_script": 0,
    }


def update_language_stats(
    stats: dict,
    record: dict,
) -> None:

    utterance = record.get(
        "utterance",
        "",
    ).strip()

    normalized = normalize_text(
        utterance
    )

    tokens = tokenize(
        utterance
    )

    target_features = get_target_features(
        record
    )

    stats["samples"] += 1

    stats["utterances"].append(
        utterance
    )

    stats["normalized_utterances"].append(
        normalized
    )

    stats["all_tokens"].extend(
        tokens
    )

    stats["token_counts"].update(
        tokens
    )

    stats["bigram_counts"].update(
        get_ngrams(
            tokens,
            2,
        )
    )

    stats["trigram_counts"].update(
        get_ngrams(
            tokens,
            3,
        )
    )

    stats[
        "utterance_lengths_words"
    ].append(
        len(tokens)
    )

    stats[
        "utterance_lengths_chars"
    ].append(
        len(utterance)
    )

    # --------------------------------------------------------
    # Semantic groups
    # --------------------------------------------------------

    if target_features[
        "has_absent_symptom"
    ]:
        stats[
            "semantic_groups"
        ]["negation"].append(
            utterance
        )

    if target_features[
        "has_uncertain_symptom"
    ]:
        stats[
            "semantic_groups"
        ]["uncertainty"].append(
            utterance
        )

    medication_status = (
        target_features[
            "medication_status"
        ]
    )

    if medication_status:
        key = (
            f"medications_"
            f"{medication_status}"
        )

        if key in stats[
            "semantic_groups"
        ]:
            stats[
                "semantic_groups"
            ][key].append(
                utterance
            )

    allergy_status = (
        target_features[
            "allergy_status"
        ]
    )

    if allergy_status:
        key = (
            f"allergies_"
            f"{allergy_status}"
        )

        if key in stats[
            "semantic_groups"
        ]:
            stats[
                "semantic_groups"
            ][key].append(
                utterance
            )

    if (
        target_features[
            "symptom_count"
        ] >= 2
    ):
        stats[
            "semantic_groups"
        ]["multiple_symptoms"].append(
            utterance
        )

    if (
        target_features[
            "duration_count"
        ] >= 2
    ):
        stats[
            "semantic_groups"
        ]["multiple_durations"].append(
            utterance
        )

    # --------------------------------------------------------
    # Number systems
    # --------------------------------------------------------

    if contains_ascii_digit(
        utterance
    ):
        stats[
            "digit_types"
        ]["ascii"] += 1

    if contains_devanagari_digit(
        utterance
    ):
        stats[
            "digit_types"
        ]["devanagari"] += 1

    if contains_arabic_indic_digit(
        utterance
    ):
        stats[
            "digit_types"
        ]["arabic_indic"] += 1

    # --------------------------------------------------------
    # Script checks
    # --------------------------------------------------------

    if contains_devanagari(
        utterance
    ):
        stats[
            "contains_devanagari"
        ] += 1

    if contains_arabic_script(
        utterance
    ):
        stats[
            "contains_arabic_script"
        ] += 1


# ============================================================
# SUMMARY HELPERS
# ============================================================

def mean(values: list[int]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(
    values: list[int],
) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    n = len(ordered)

    midpoint = n // 2

    if n % 2 == 1:
        return float(
            ordered[midpoint]
        )

    return (
        ordered[midpoint - 1]
        + ordered[midpoint]
    ) / 2


def percentile(
    values: list[int],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    index = (
        len(ordered) - 1
    ) * percentile_value

    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return float(
            ordered[lower]
        )

    lower_value = ordered[lower]
    upper_value = ordered[upper]

    fraction = index - lower

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def summarize_text_group(
    utterances: list[str],
) -> dict:

    normalized = [
        normalize_text(text)
        for text in utterances
    ]

    tokens = []

    for utterance in utterances:
        tokens.extend(
            tokenize(
                utterance
            )
        )

    token_counter = Counter(
        tokens
    )

    bigram_counter = Counter()

    trigram_counter = Counter()

    for utterance in utterances:
        utterance_tokens = tokenize(
            utterance
        )

        bigram_counter.update(
            get_ngrams(
                utterance_tokens,
                2,
            )
        )

        trigram_counter.update(
            get_ngrams(
                utterance_tokens,
                3,
            )
        )

    unique_utterances = len(
        set(normalized)
    )

    return {
        "samples": len(
            utterances
        ),

        "unique_utterances": (
            unique_utterances
        ),

        "unique_ratio": (
            unique_utterances
            / len(utterances)
            if utterances
            else 0.0
        ),

        "total_tokens": len(
            tokens
        ),

        "unique_tokens": len(
            token_counter
        ),

        "type_token_ratio": (
            type_token_ratio(
                tokens
            )
        ),

        "token_entropy": (
            normalized_entropy(
                token_counter
            )
        ),

        "top_tokens": (
            token_counter.most_common(
                15
            )
        ),

        "top_bigrams": (
            bigram_counter.most_common(
                10
            )
        ),

        "top_trigrams": (
            trigram_counter.most_common(
                10
            )
        ),
    }


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_language_stats(
    stats: dict,
) -> dict:

    normalized_utterances = (
        stats[
            "normalized_utterances"
        ]
    )

    unique_utterances = len(
        set(
            normalized_utterances
        )
    )

    total_samples = stats[
        "samples"
    ]

    semantic_groups = {}

    for group_name, utterances in (
        stats[
            "semantic_groups"
        ].items()
    ):
        semantic_groups[
            group_name
        ] = summarize_text_group(
            utterances
        )

    return {
        "samples": total_samples,

        "unique_utterances": (
            unique_utterances
        ),

        "duplicate_utterances": (
            total_samples
            - unique_utterances
        ),

        "unique_utterance_ratio": (
            unique_utterances
            / total_samples
            if total_samples
            else 0.0
        ),

        "utterance_length_words": {
            "mean": mean(
                stats[
                    "utterance_lengths_words"
                ]
            ),
            "median": median(
                stats[
                    "utterance_lengths_words"
                ]
            ),
            "p95": percentile(
                stats[
                    "utterance_lengths_words"
                ],
                0.95,
            ),
            "max": max(
                stats[
                    "utterance_lengths_words"
                ],
                default=0,
            ),
        },

        "utterance_length_chars": {
            "mean": mean(
                stats[
                    "utterance_lengths_chars"
                ]
            ),
            "median": median(
                stats[
                    "utterance_lengths_chars"
                ]
            ),
            "p95": percentile(
                stats[
                    "utterance_lengths_chars"
                ],
                0.95,
            ),
            "max": max(
                stats[
                    "utterance_lengths_chars"
                ],
                default=0,
            ),
        },

        "lexical": {
            "total_tokens": len(
                stats[
                    "all_tokens"
                ]
            ),

            "unique_tokens": len(
                stats[
                    "token_counts"
                ]
            ),

            "type_token_ratio": (
                type_token_ratio(
                    stats[
                        "all_tokens"
                    ]
                )
            ),

            "token_entropy": (
                normalized_entropy(
                    stats[
                        "token_counts"
                    ]
                )
            ),

            "top_tokens": (
                stats[
                    "token_counts"
                ].most_common(
                    20
                )
            ),

            "top_bigrams": (
                stats[
                    "bigram_counts"
                ].most_common(
                    15
                )
            ),

            "top_trigrams": (
                stats[
                    "trigram_counts"
                ].most_common(
                    15
                )
            ),
        },

        "digit_types": dict(
            stats[
                "digit_types"
            ]
        ),

        "script_checks": {
            "contains_devanagari": (
                stats[
                    "contains_devanagari"
                ]
            ),

            "contains_arabic_script": (
                stats[
                    "contains_arabic_script"
                ]
            ),
        },

        "semantic_groups": (
            semantic_groups
        ),
    }


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_group_summary(
    name: str,
    group: dict,
) -> None:

    print(
        f"    {name:<24}"
        f"{group['samples']:>5} samples | "
        f"{group['unique_utterances']:>5} unique | "
        f"{group['unique_ratio'] * 100:6.2f}% | "
        f"TTR {group['type_token_ratio']:.3f}"
    )


def print_language_summary(
    language: str,
    summary: dict,
) -> None:

    print()
    print("=" * 88)
    print(
        f"{language} - "
        f"{summary['samples']} TRAINING SAMPLES"
    )
    print("=" * 88)

    print("\nUTTERANCE DIVERSITY")

    print(
        f"    Unique utterances:     "
        f"{summary['unique_utterances']} "
        f"/ {summary['samples']} "
        f"("
        f"{summary['unique_utterance_ratio'] * 100:.2f}%"
        f")"
    )

    print(
        f"    Duplicate utterances:  "
        f"{summary['duplicate_utterances']}"
    )

    word_length = (
        summary[
            "utterance_length_words"
        ]
    )

    print("\nUTTERANCE LENGTH - WORDS")

    print(
        f"    Mean:   "
        f"{word_length['mean']:.2f}"
    )

    print(
        f"    Median: "
        f"{word_length['median']:.2f}"
    )

    print(
        f"    P95:    "
        f"{word_length['p95']:.2f}"
    )

    print(
        f"    Max:    "
        f"{word_length['max']}"
    )

    lexical = summary[
        "lexical"
    ]

    print("\nLEXICAL DIVERSITY")

    print(
        f"    Total tokens:  "
        f"{lexical['total_tokens']}"
    )

    print(
        f"    Unique tokens: "
        f"{lexical['unique_tokens']}"
    )

    print(
        f"    TTR:           "
        f"{lexical['type_token_ratio']:.4f}"
    )

    print(
        f"    Token entropy: "
        f"{lexical['token_entropy']:.4f}"
    )

    print("\nSEMANTIC REALIZATION GROUPS")

    groups = summary[
        "semantic_groups"
    ]

    group_order = [
        "negation",
        "uncertainty",
        "medications_none",
        "medications_unknown",
        "medications_reported",
        "allergies_none",
        "allergies_unknown",
        "allergies_reported",
        "multiple_symptoms",
        "multiple_durations",
    ]

    for group_name in group_order:
        print_group_summary(
            group_name,
            groups[group_name],
        )

    print("\nNUMBER SYSTEMS")

    digit_types = summary[
        "digit_types"
    ]

    print(
        f"    ASCII digits:        "
        f"{digit_types.get('ascii', 0)}"
    )

    print(
        f"    Devanagari digits:   "
        f"{digit_types.get('devanagari', 0)}"
    )

    print(
        f"    Arabic-Indic digits: "
        f"{digit_types.get('arabic_indic', 0)}"
    )

    print("\nTOP TOKENS")

    for token, count in (
        lexical[
            "top_tokens"
        ][:10]
    ):
        print(
            f"    {token:<25}"
            f"{count:>5}"
        )

    print("\nTOP BIGRAMS")

    for ngram, count in (
        lexical[
            "top_bigrams"
        ][:8]
    ):
        print(
            f"    {ngram:<40}"
            f"{count:>5}"
        )


def print_cross_language_summary(
    summaries: dict,
) -> None:

    print()
    print("=" * 110)
    print(
        "CROSS-LANGUAGE REALIZATION SUMMARY"
    )
    print("=" * 110)

    header = (
        f"{'Lang':<8}"
        f"{'N':>7}"
        f"{'Unique%':>11}"
        f"{'Mean words':>13}"
        f"{'Tokens':>10}"
        f"{'Vocab':>9}"
        f"{'TTR':>9}"
        f"{'Entropy':>10}"
        f"{'Neg uniq%':>12}"
        f"{'Unc uniq%':>12}"
    )

    print(header)
    print("-" * len(header))

    for language in LANGUAGE_ORDER:
        summary = summaries.get(
            language
        )

        if not summary:
            continue

        lexical = summary[
            "lexical"
        ]

        negation = summary[
            "semantic_groups"
        ]["negation"]

        uncertainty = summary[
            "semantic_groups"
        ]["uncertainty"]

        print(
            f"{language:<8}"
            f"{summary['samples']:>7}"
            f"{summary['unique_utterance_ratio'] * 100:>10.2f}%"
            f"{summary['utterance_length_words']['mean']:>13.2f}"
            f"{lexical['total_tokens']:>10}"
            f"{lexical['unique_tokens']:>9}"
            f"{lexical['type_token_ratio']:>9.3f}"
            f"{lexical['token_entropy']:>10.3f}"
            f"{negation['unique_ratio'] * 100:>11.2f}%"
            f"{uncertainty['unique_ratio'] * 100:>11.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 88)
    print(
        "NOOR HEALTH - "
        "LANGUAGE REALIZATION AUDIT"
    )
    print("=" * 88)

    print(
        f"\nTraining file:\n"
        f"{TRAIN_FILE}"
    )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: "
            f"{TRAIN_FILE}"
        )

    records = load_jsonl(
        TRAIN_FILE
    )

    print(
        f"\nTraining samples: "
        f"{len(records)}"
    )

    stats_by_language = defaultdict(
        empty_language_stats
    )

    for record in records:

        language = record.get(
            "language"
        )

        if not language:
            raise ValueError(
                "Record missing language: "
                f"{record.get('case_id')}"
            )

        update_language_stats(
            stats_by_language[
                language
            ],
            record,
        )

    summaries = {}

    for language, stats in (
        stats_by_language.items()
    ):
        summaries[
            language
        ] = (
            serialize_language_stats(
                stats
            )
        )

    print_cross_language_summary(
        summaries
    )

    for language in LANGUAGE_ORDER:

        if language not in summaries:
            continue

        print_language_summary(
            language,
            summaries[
                language
            ],
        )

    unknown_languages = (
        set(summaries)
        - set(LANGUAGE_ORDER)
    )

    for language in sorted(
        unknown_languages
    ):
        print_language_summary(
            language,
            summaries[
                language
            ],
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "training_file": str(
            TRAIN_FILE
        ),
        "total_samples": len(
            records
        ),
        "languages": summaries,
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
    print("=" * 88)
    print("AUDIT COMPLETE")
    print("=" * 88)

    print(
        f"\nSaved:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()