import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from config import SYNTHETIC_DATA_DIR, RESULTS_DIR


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAPI_KEY")
)

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2_full.jsonl"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "semantic_validation.jsonl"
)

MODEL = "gpt-5-mini"

MAX_WORKERS = 10
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


EXPECTED_LANGUAGES = {
    "en",
    "de",
    "ar_msa",
    "fr",
    "es",
    "hi",
    "sw",
}


LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Modern Standard Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "sw": "Swahili",
}


write_lock = threading.Lock()


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path) -> list[dict]:

    rows = []

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

                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON in "
                    f"{path.name}, "
                    f"line {line_number}: "
                    f"{error}"
                ) from error

    return rows


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_existing_results():

    results = {}

    if not OUTPUT_FILE.exists():
        return results

    with OUTPUT_FILE.open(
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

                row = json.loads(line)

            except json.JSONDecodeError:

                print(
                    f"WARNING: skipping invalid "
                    f"result line {line_number}"
                )

                continue

            key = (
                row.get("case_id"),
                row.get("split"),
                row.get("language"),
            )

            results[key] = row

    return results


# ============================================================
# PROMPT
# ============================================================

def build_prompt(sample: dict) -> str:

    language_code = sample["language"]

    language_name = LANGUAGE_NAMES.get(
        language_code,
        language_code,
    )

    utterance = sample["utterance"]

    target_json = json.dumps(
        sample["target"],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are a strict multilingual clinical-data quality reviewer.

You are validating one synthetic training sample for Noor Health,
an offline-first multilingual clinical intake assistant.

The system is NOT a diagnostic system.

Your task is to compare a patient utterance against its canonical
structured clinical target.

LANGUAGE:
{language_name}

LANGUAGE CODE:
{language_code}

PATIENT UTTERANCE:
{utterance}

EXPECTED STRUCTURED TARGET:
{target_json}


============================================================
TASK 1 — CLINICAL SEMANTIC CONSISTENCY
============================================================

Determine whether the patient utterance faithfully expresses the
clinical information represented by the target.

Check ALL of the following carefully:

1. CHIEF COMPLAINT

The chief complaint must be expressed as present.

Do not require the utterance to literally use the English canonical
term. Natural translations and normal synonyms are allowed.


2. SYMPTOMS

For every symptom in the target, verify:

- the symptom itself is represented,
- "present" is expressed as present,
- "absent" is clearly negated,
- "uncertain" is clearly expressed as uncertain.

Do not confuse absence with uncertainty.

Do not allow clinically meaningful symptoms to be added if they are
not represented in the target.


3. DURATIONS

Durations are symptom-specific.

If a symptom has a duration in the target, the utterance must express
that same duration for that symptom.

Example:

Target:
cough = 4 days
fever = present, duration unknown

Correct:
"I have had a cough for four days and I also have a fever."

Incorrect:
"I have had a cough and fever for four days."

The incorrect version assigns four days to both symptoms.

If duration is null, do NOT require a duration to be mentioned.

If the utterance invents a specific duration for a symptom whose
duration is null, that is a semantic error.


4. MEDICATIONS

Interpret the target statuses exactly:

unknown:
The utterance must NOT claim whether the patient takes medication.

none:
The utterance must explicitly state that the patient takes no
medication.

reported:
The utterance must communicate the listed medication item(s).

Do not treat "not mentioned" as "none".


5. ALLERGIES

Interpret the target statuses exactly:

unknown:
The utterance must NOT claim whether allergies exist.

none:
The utterance must explicitly communicate no known allergies.

reported:
The utterance must communicate the listed allergy item(s).

Do not treat "not mentioned" as "none".


6. MISSING INFORMATION

The field "missing_information" describes information that is absent
from the utterance and should later be requested by the intake system.

Therefore:

- "duration" means the chief complaint duration is not stated,
- "medications" means medication status is not stated,
- "allergies" means allergy status is not stated.

Do NOT require the utterance to literally mention that information is
missing.

Instead verify that the corresponding information is genuinely absent.


7. HALLUCINATIONS / CONTRADICTIONS

Fail semantic consistency if the utterance:

- adds clinically meaningful facts not represented in the target,
- contradicts the target,
- changes symptom status,
- changes duration,
- changes medication,
- changes allergy information,
- associates a duration with the wrong symptom.


============================================================
TASK 2 — LANGUAGE QUALITY
============================================================

Separately judge whether the utterance is natural and grammatically
acceptable in {language_name}.

Minor stylistic variation is acceptable.

Do NOT fail language quality merely because:

- wording is informal,
- a synonym is used,
- punctuation differs,
- numbers are written as words instead of digits.

Fail language quality when there is a genuine linguistic problem such
as:

- clearly incorrect grammar,
- malformed number/unit agreement,
- unnatural construction severe enough to be poor training data,
- wrong-language text,
- broken or corrupted text.

For Modern Standard Arabic, pay particular attention to natural
number/unit grammar.

For example:

"منذ أسبوعين"
is natural for "for two weeks".

"منذ ٢ أسبوعين"
is grammatically malformed and should fail language quality.

For Hindi, ensure the sentence is understandable natural Hindi.

For Swahili, ensure the sentence is understandable natural Swahili.


============================================================
IMPORTANT DISTINCTION
============================================================

A sample may be semantically correct but linguistically flawed.

Example:

The utterance communicates exactly "two weeks" but uses malformed
Arabic grammar.

In that situation:

semantic_passed = true
language_passed = false

Do NOT mark a semantic failure purely because of grammar if the
clinical meaning is still unambiguous.


============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "semantic_passed": true,
  "language_passed": true,
  "semantic_errors": [],
  "language_errors": []
}}

If an error exists, use short objects like:

{{
  "type": "duration_mismatch",
  "description": "The utterance says three days but the target says four days."
}}

Possible semantic error types include:

- missing_chief_complaint
- missing_symptom
- symptom_status_mismatch
- duration_missing
- duration_mismatch
- duration_wrong_symptom
- invented_duration
- medication_mismatch
- allergy_mismatch
- missing_information_mismatch
- hallucinated_information
- contradiction
- other_semantic_error

Possible language error types include:

- grammar_error
- malformed_duration_expression
- wrong_language
- unnatural_language
- corrupted_text
- other_language_error

Be strict about clinical meaning.

Do not invent errors.

Return JSON only.
""".strip()


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_json_response(text: str) -> dict:

    text = text.strip()

    # Defensive cleanup in case the model still uses fences.
    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    result = json.loads(text)

    required_fields = {
        "semantic_passed",
        "language_passed",
        "semantic_errors",
        "language_errors",
    }

    missing_fields = (
        required_fields
        - set(result.keys())
    )

    if missing_fields:

        raise ValueError(
            "Judge response missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(
        result["semantic_passed"],
        bool,
    ):

        raise ValueError(
            "semantic_passed must be boolean"
        )

    if not isinstance(
        result["language_passed"],
        bool,
    ):

        raise ValueError(
            "language_passed must be boolean"
        )

    if not isinstance(
        result["semantic_errors"],
        list,
    ):

        raise ValueError(
            "semantic_errors must be list"
        )

    if not isinstance(
        result["language_errors"],
        list,
    ):

        raise ValueError(
            "language_errors must be list"
        )

    return result


# ============================================================
# API CALL WITH RETRIES
# ============================================================

def judge_sample(sample: dict) -> dict:

    prompt = build_prompt(sample)

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = client.responses.create(
                model=MODEL,
                input=prompt,
            )

            result = parse_json_response(
                response.output_text
            )

            return {
                "case_id":
                    sample["case_id"],

                "split":
                    sample["split"],

                "language":
                    sample["language"],

                "semantic_passed":
                    result[
                        "semantic_passed"
                    ],

                "language_passed":
                    result[
                        "language_passed"
                    ],

                "semantic_errors":
                    result[
                        "semantic_errors"
                    ],

                "language_errors":
                    result[
                        "language_errors"
                    ],

                "judge_model":
                    MODEL,
            }

        except Exception as error:

            last_error = error

            if attempt == MAX_RETRIES:
                break

            delay = (
                RETRY_BASE_DELAY
                * (2 ** (attempt - 1))
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Judge failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# SAVE
# ============================================================

def save_result(result: dict):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = json.dumps(
        result,
        ensure_ascii=False,
    )

    with write_lock:

        with OUTPUT_FILE.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                line + "\n"
            )


# ============================================================
# ERROR TYPE EXTRACTION
# ============================================================

def get_error_types(
    errors: list,
) -> list[str]:

    types = []

    for error in errors:

        if isinstance(error, dict):

            error_type = error.get(
                "type",
                "unknown",
            )

        else:

            error_type = "unknown"

        types.append(error_type)

    return types


# ============================================================
# REPORT
# ============================================================

def print_report(
    results: list[dict],
    failed_api_jobs: list,
):

    print("\n" + "=" * 60)
    print(
        "SEMANTIC VALIDATION REPORT"
    )
    print("=" * 60)

    total = len(results)

    semantic_passed = sum(
        result["semantic_passed"]
        for result in results
    )

    language_passed = sum(
        result["language_passed"]
        for result in results
    )

    fully_passed = sum(
        (
            result["semantic_passed"]
            and result["language_passed"]
        )
        for result in results
    )

    semantic_failed = (
        total - semantic_passed
    )

    language_failed = (
        total - language_passed
    )

    fully_failed = (
        total - fully_passed
    )

    print(
        f"\nResults available: "
        f"{total}"
    )

    print(
        f"API jobs failed:    "
        f"{len(failed_api_jobs)}"
    )

    print(
        "\nClinical semantics:"
    )

    print(
        f"  Passed: {semantic_passed}"
    )

    print(
        f"  Failed: {semantic_failed}"
    )

    print(
        "\nLanguage quality:"
    )

    print(
        f"  Passed: {language_passed}"
    )

    print(
        f"  Failed: {language_failed}"
    )

    print(
        "\nFully usable samples:"
    )

    print(
        f"  Passed both: {fully_passed}"
    )

    print(
        f"  Failed either: {fully_failed}"
    )

    # --------------------------------------------------------
    # LANGUAGE BREAKDOWN
    # --------------------------------------------------------

    by_language = defaultdict(list)

    for result in results:

        by_language[
            result["language"]
        ].append(result)

    print(
        "\nResults by language:"
    )

    print(
        "  "
        f"{'Language':<10}"
        f"{'Total':>8}"
        f"{'SemFail':>10}"
        f"{'LangFail':>10}"
        f"{'Either':>10}"
    )

    for language in sorted(
        EXPECTED_LANGUAGES
    ):

        rows = by_language[
            language
        ]

        sem_fail = sum(
            not row["semantic_passed"]
            for row in rows
        )

        lang_fail = sum(
            not row["language_passed"]
            for row in rows
        )

        either_fail = sum(
            not (
                row["semantic_passed"]
                and row["language_passed"]
            )
            for row in rows
        )

        print(
            "  "
            f"{language:<10}"
            f"{len(rows):>8}"
            f"{sem_fail:>10}"
            f"{lang_fail:>10}"
            f"{either_fail:>10}"
        )

    # --------------------------------------------------------
    # ERROR TYPES
    # --------------------------------------------------------

    semantic_error_counts = Counter()

    language_error_counts = Counter()

    for result in results:

        semantic_error_counts.update(
            get_error_types(
                result[
                    "semantic_errors"
                ]
            )
        )

        language_error_counts.update(
            get_error_types(
                result[
                    "language_errors"
                ]
            )
        )

    print(
        "\nSemantic error types:"
    )

    if semantic_error_counts:

        for (
            error_type,
            count,
        ) in (
            semantic_error_counts
            .most_common()
        ):

            print(
                f"  {error_type:<32} "
                f"{count}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nLanguage error types:"
    )

    if language_error_counts:

        for (
            error_type,
            count,
        ) in (
            language_error_counts
            .most_common()
        ):

            print(
                f"  {error_type:<32} "
                f"{count}"
            )

    else:

        print(
            "  None"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - SEMANTIC DATASET VALIDATION"
    )
    print("=" * 60)

    samples = load_jsonl(
        INPUT_FILE
    )

    existing_results = (
        load_existing_results()
    )

    print(
        f"\nDataset samples: "
        f"{len(samples)}"
    )

    print(
        f"Existing judge results: "
        f"{len(existing_results)}"
    )

    jobs = []

    for sample in samples:

        key = (
            sample["case_id"],
            sample["split"],
            sample["language"],
        )

        if key not in existing_results:

            jobs.append(sample)

    print(
        f"Remaining samples: "
        f"{len(jobs)}"
    )

    print(
        f"Workers: "
        f"{MAX_WORKERS}"
    )

    # --------------------------------------------------------
    # NOTHING LEFT
    # --------------------------------------------------------

    if not jobs:

        print(
            "\nAll samples have already "
            "been validated."
        )

        print_report(
            list(
                existing_results.values()
            ),
            [],
        )

        return

    # --------------------------------------------------------
    # PARALLEL VALIDATION
    # --------------------------------------------------------

    print(
        "\nStarting parallel semantic "
        "validation...\n"
    )

    start_time = time.time()

    completed_now = 0

    failed_api_jobs = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_sample = {
            executor.submit(
                judge_sample,
                sample,
            ): sample

            for sample in jobs
        }

        for future in as_completed(
            future_to_sample
        ):

            sample = (
                future_to_sample[
                    future
                ]
            )

            try:

                result = future.result()

                save_result(result)

                key = (
                    result["case_id"],
                    result["split"],
                    result["language"],
                )

                existing_results[
                    key
                ] = result

                completed_now += 1

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    completed_now
                    / elapsed
                    * 60
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    len(jobs)
                    - completed_now
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                status = []

                if result[
                    "semantic_passed"
                ]:

                    status.append(
                        "SEM:PASS"
                    )

                else:

                    status.append(
                        "SEM:FAIL"
                    )

                if result[
                    "language_passed"
                ]:

                    status.append(
                        "LANG:PASS"
                    )

                else:

                    status.append(
                        "LANG:FAIL"
                    )

                status_text = " | ".join(
                    status
                )

                print(
                    f"[{completed_now}/"
                    f"{len(jobs)}] "
                    f"{result['case_id']}/"
                    f"{result['language']} "
                    f"{status_text} | "
                    f"{rate:.1f}/min | "
                    f"ETA {eta:.1f} min"
                )

            except Exception as error:

                failed_api_jobs.append({
                    "case_id":
                        sample["case_id"],

                    "split":
                        sample["split"],

                    "language":
                        sample["language"],

                    "error":
                        str(error),
                })

                print(
                    f"ERROR: "
                    f"{sample['case_id']}/"
                    f"{sample['language']} "
                    f"{error}"
                )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    runtime_minutes = (
        time.time()
        - start_time
    ) / 60

    print(
        "\n" + "=" * 60
    )

    print(
        "VALIDATION RUN COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nValidated now: "
        f"{completed_now}"
    )

    print(
        f"API failures: "
        f"{len(failed_api_jobs)}"
    )

    print(
        f"Total results available: "
        f"{len(existing_results)}/"
        f"{len(samples)}"
    )

    print(
        f"Runtime: "
        f"{runtime_minutes:.1f} minutes"
    )

    print_report(
        list(
            existing_results.values()
        ),
        failed_api_jobs,
    )

    if failed_api_jobs:

        print(
            "\nSome API calls failed."
        )

        print(
            "Simply run this script again. "
            "Completed samples will be skipped."
        )


if __name__ == "__main__":
    main()