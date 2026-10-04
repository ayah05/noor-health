from pathlib import Path
import json
from collections import defaultdict

import modal


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "train_sft.jsonl"

MODEL_VOLUME_NAME = "noor-health-models"

MODEL_DIR = "/models/qwen3-0.6b"

MAX_LENGTH = 512


# ============================================================
# MODAL
# ============================================================

app = modal.App(
    "noor-health-token-audit"
)

model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME
)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
    )
)


# ============================================================
# PROMPT
#
# IMPORTANT:
# This should match the training prompt.
# ============================================================

SYSTEM_PROMPT = """You are Noor Health, a clinical intake structuring assistant.

Convert the patient statement into structured JSON.

Rules:
- Do not diagnose.
- Use canonical English clinical terms.
- Preserve uncertainty.
- A symptom status must be present, absent, or uncertain.
- Only present symptoms may have a duration.
- If a duration is not reported, use null.
- Medication status must be unknown, none, or reported.
- Allergy status must be unknown, none, or reported.
- Return JSON only.
"""


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
                    json.loads(line)
                )

    return records


# ============================================================
# MODAL AUDIT
# ============================================================

@app.function(
    image=image,
    volumes={
        "/models": model_volume,
    },
    timeout=1800,
)
def audit_records(records):

    import numpy as np

    from transformers import (
        AutoTokenizer,
    )

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_DIR
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    results = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        language = record[
            "language"
        ]

        utterance = record[
            "utterance"
        ]

        target = record[
            "target"
        ]

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        user_content = (
            f"Language: {language}\n\n"
            f"Patient statement:\n"
            f"{utterance}"
        )

        target_json = json.dumps(
            target,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # ----------------------------------------------------
        # Prompt without assistant response
        # ----------------------------------------------------

        prompt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_text = (
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        # ----------------------------------------------------
        # Full training conversation
        # ----------------------------------------------------

        full_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": target_json,
            },
        ]

        full_text = (
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        prompt_length = len(
            prompt_tokens
        )

        full_length = len(
            full_tokens
        )

        assistant_length = max(
            0,
            full_length
            - prompt_length,
        )

        # ----------------------------------------------------
        # Truncation simulation
        # ----------------------------------------------------

        truncated = (
            full_length
            > MAX_LENGTH
        )

        tokens_lost = max(
            0,
            full_length
            - MAX_LENGTH,
        )

        assistant_tokens_available = max(
            0,
            MAX_LENGTH
            - prompt_length,
        )

        assistant_tokens_kept = min(
            assistant_length,
            assistant_tokens_available,
        )

        assistant_tokens_lost = max(
            0,
            assistant_length
            - assistant_tokens_kept,
        )

        assistant_loss_pct = (
            assistant_tokens_lost
            / assistant_length
            * 100
            if assistant_length > 0
            else 0.0
        )

        results.append(
            {
                "case_id":
                    record["case_id"],

                "language":
                    language,

                "prompt_tokens":
                    prompt_length,

                "assistant_tokens":
                    assistant_length,

                "full_tokens":
                    full_length,

                "truncated":
                    truncated,

                "tokens_lost":
                    tokens_lost,

                "assistant_tokens_lost":
                    assistant_tokens_lost,

                "assistant_loss_pct":
                    assistant_loss_pct,
            }
        )

        if index % 500 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(records)}"
            )

    return results


# ============================================================
# STATISTICS
# ============================================================

def percentile(
    values,
    percentile_value,
):

    import math

    values = sorted(values)

    if not values:
        return 0

    index = (
        percentile_value
        / 100
        * (len(values) - 1)
    )

    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower]
        * (1 - weight)
        +
        values[upper]
        * weight
    )


def summarize(
    records,
):

    lengths = [
        record["full_tokens"]
        for record in records
    ]

    prompt_lengths = [
        record["prompt_tokens"]
        for record in records
    ]

    assistant_lengths = [
        record["assistant_tokens"]
        for record in records
    ]

    truncated = [
        record
        for record in records
        if record["truncated"]
    ]

    target_truncated = [
        record
        for record in records
        if record[
            "assistant_tokens_lost"
        ] > 0
    ]

    print()
    print(
        "=" * 72
    )
    print(
        "NOOR HEALTH - TRAINING TOKEN AUDIT"
    )
    print(
        "=" * 72
    )

    print(
        f"\nTraining examples: "
        f"{len(records)}"
    )

    print(
        f"Max sequence length: "
        f"{MAX_LENGTH}"
    )

    print(
        "\nFULL SEQUENCE LENGTH"
    )

    print(
        "-" * 72
    )

    print(
        f"Mean:    "
        f"{sum(lengths) / len(lengths):.1f}"
    )

    print(
        f"Median:  "
        f"{percentile(lengths, 50):.1f}"
    )

    print(
        f"P90:     "
        f"{percentile(lengths, 90):.1f}"
    )

    print(
        f"P95:     "
        f"{percentile(lengths, 95):.1f}"
    )

    print(
        f"P99:     "
        f"{percentile(lengths, 99):.1f}"
    )

    print(
        f"Maximum: "
        f"{max(lengths)}"
    )

    print(
        "\nPROMPT LENGTH"
    )

    print(
        "-" * 72
    )

    print(
        f"Mean:    "
        f"{sum(prompt_lengths) / len(prompt_lengths):.1f}"
    )

    print(
        f"P95:     "
        f"{percentile(prompt_lengths, 95):.1f}"
    )

    print(
        f"Maximum: "
        f"{max(prompt_lengths)}"
    )

    print(
        "\nTARGET / ASSISTANT LENGTH"
    )

    print(
        "-" * 72
    )

    print(
        f"Mean:    "
        f"{sum(assistant_lengths) / len(assistant_lengths):.1f}"
    )

    print(
        f"P95:     "
        f"{percentile(assistant_lengths, 95):.1f}"
    )

    print(
        f"Maximum: "
        f"{max(assistant_lengths)}"
    )

    print(
        "\nTRUNCATION"
    )

    print(
        "-" * 72
    )

    print(
        f"Sequences > {MAX_LENGTH}: "
        f"{len(truncated)} / "
        f"{len(records)} "
        f"({len(truncated) / len(records) * 100:.2f}%)"
    )

    print(
        "Examples losing assistant tokens: "
        f"{len(target_truncated)} / "
        f"{len(records)} "
        f"({len(target_truncated) / len(records) * 100:.2f}%)"
    )

    if target_truncated:

        lost = [
            record[
                "assistant_tokens_lost"
            ]
            for record
            in target_truncated
        ]

        print(
            f"Mean assistant tokens lost: "
            f"{sum(lost) / len(lost):.1f}"
        )

        print(
            f"Maximum assistant tokens lost: "
            f"{max(lost)}"
        )

    # ========================================================
    # LANGUAGE BREAKDOWN
    # ========================================================

    by_language = defaultdict(
        list
    )

    for record in records:

        by_language[
            record["language"]
        ].append(
            record
        )

    print(
        "\nBY LANGUAGE"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Language':<12}"
        f"{'N':>6}"
        f"{'Mean':>10}"
        f"{'P95':>10}"
        f"{'Max':>8}"
        f"{'Trunc':>10}"
    )

    for language in sorted(
        by_language
    ):

        language_records = (
            by_language[
                language
            ]
        )

        language_lengths = [
            record["full_tokens"]
            for record
            in language_records
        ]

        language_truncated = sum(
            1
            for record
            in language_records
            if record[
                "assistant_tokens_lost"
            ] > 0
        )

        print(
            f"{language:<12}"
            f"{len(language_records):>6}"
            f"{sum(language_lengths) / len(language_lengths):>10.1f}"
            f"{percentile(language_lengths, 95):>10.1f}"
            f"{max(language_lengths):>8}"
            f"{language_truncated:>10}"
        )

    # ========================================================
    # WORST EXAMPLES
    # ========================================================

    print(
        "\nWORST TARGET TRUNCATION CASES"
    )

    print(
        "-" * 72
    )

    worst = sorted(
        target_truncated,
        key=lambda record:
            record[
                "assistant_tokens_lost"
            ],
        reverse=True,
    )[:20]

    if not worst:

        print(
            "No assistant targets were truncated."
        )

    else:

        for record in worst:

            print(
                f"{record['case_id']} | "
                f"{record['language']} | "
                f"full={record['full_tokens']} | "
                f"target={record['assistant_tokens']} | "
                f"lost={record['assistant_tokens_lost']} "
                f"({record['assistant_loss_pct']:.1f}%)"
            )


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAIN_FILE}"
        )

    print(
        "Loading training data..."
    )

    records = load_jsonl(
        TRAIN_FILE
    )

    print(
        f"Loaded {len(records)} "
        f"training examples."
    )

    print(
        "\nRunning tokenizer audit "
        "on Modal..."
    )

    results = (
        audit_records.remote(
            records
        )
    )

    summarize(
        results
    )

    # Save detailed audit
    output_file = (
        ROOT
        / "results"
        / "training_token_audit.jsonl"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in results:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"\nDetailed audit saved to:\n"
        f"{output_file}"
    )