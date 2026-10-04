import json
import os
import time
from collections import Counter, defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from config import SYNTHETIC_DATA_DIR, RESULTS_DIR


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2.jsonl"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "semantic_validation_v2.jsonl"
)

VALIDATOR_MODEL = "gpt-5-mini"


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_keys() -> set[tuple[str, str]]:
    """
    Allows validation to resume after interruption.
    """

    existing = set()

    if not OUTPUT_FILE.exists():
        return existing

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            result = json.loads(line)

            existing.add(
                (
                    result["case_id"],
                    result["language"],
                )
            )

    return existing


# ============================================================
# PROMPT
# ============================================================

def build_validation_prompt(
    sample: dict,
) -> str:

    target_json = json.dumps(
        sample["target"],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are validating multilingual synthetic training data for
a clinical information extraction research project.

Your task is NOT to diagnose the patient.

Your task is ONLY to determine whether the patient utterance
expresses exactly the information contained in the ground
truth target.


============================================================
LANGUAGE
============================================================

Language code:

{sample["language"]}


============================================================
PATIENT UTTERANCE
============================================================

{sample["utterance"]}


============================================================
EXPECTED GROUND TRUTH
============================================================

{target_json}


============================================================
VALIDATION RULES
============================================================

Compare the patient utterance with the expected ground truth.

Evaluate each category independently.


1. SYMPTOMS

Check that every symptom represented in the target is
expressed correctly in the utterance.

The utterance must not introduce additional symptoms.


2. SYMPTOM STATUS

Status semantics are strict:

"present"
= the patient clearly reports having the symptom.

"absent"
= the patient clearly denies having the symptom.

"uncertain"
= the patient explicitly expresses uncertainty about whether
  the symptom is present.

Do not treat "uncertain" as "present".

Do not treat "absent" as "unknown".


3. DURATION

Check every duration independently.

A duration must:
- have the correct value
- have the correct unit
- refer to the correct symptom

If duration is null, the utterance must NOT invent a duration
for that symptom.


4. MEDICATIONS

"unknown":
Medication must not be mentioned.

"none":
The patient must explicitly state that they are not taking
medication.

"reported":
The correct medication or medications must be mentioned.

Translated or transliterated medication names are allowed
when they refer to the same medication.


5. ALLERGIES

"unknown":
Allergies must not be mentioned.

"none":
The patient must explicitly indicate no known allergies.

"reported":
The correct allergy or allergies must be mentioned.

Translated or transliterated names are allowed.


6. MISSING INFORMATION

Missing information must remain missing.

For example:

If medications are unknown, the utterance must not turn this
into "I take no medication".

If allergies are unknown, the utterance must not turn this
into "I have no allergies".

If duration is missing, no duration may be invented.


7. EXTRA INFORMATION

The utterance must not introduce additional clinical facts
that are absent from the target.

Minor stylistic wording differences are NOT errors.

Natural grammatical transformations are NOT errors.

For example:

1 day -> "one day"
1 day -> "يوم واحد"

are semantically equivalent.


============================================================
OUTPUT
============================================================

Return ONLY valid JSON with exactly this structure:

{{
  "symptoms_correct": true,
  "statuses_correct": true,
  "durations_correct": true,
  "medications_correct": true,
  "allergies_correct": true,
  "missing_information_correct": true,
  "extra_information": false,
  "pass": true,
  "issues": []
}}

"pass" must be true ONLY if:

- symptoms_correct is true
- statuses_correct is true
- durations_correct is true
- medications_correct is true
- allergies_correct is true
- missing_information_correct is true
- extra_information is false

If something is incorrect, add a short description to
"issues".

Do not add any other fields.
Do not return Markdown.
Do not explain anything outside the JSON.
""".strip()


# ============================================================
# VALIDATE ONE SAMPLE
# ============================================================

def validate_sample(
    sample: dict,
) -> dict:

    prompt = build_validation_prompt(
        sample
    )

    response = client.responses.create(
        model=VALIDATOR_MODEL,
        input=prompt,
    )

    raw_output = (
        response.output_text.strip()
    )

    # Protect against accidental Markdown fences.
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]

    elif raw_output.startswith("```"):
        raw_output = raw_output[3:]

    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]

    raw_output = raw_output.strip()

    result = json.loads(
        raw_output
    )

    return result


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result: dict,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# VALIDATE RESULT FORMAT
# ============================================================

def validate_result_format(
    validation: dict,
) -> None:

    boolean_fields = [
        "symptoms_correct",
        "statuses_correct",
        "durations_correct",
        "medications_correct",
        "allergies_correct",
        "missing_information_correct",
        "extra_information",
        "pass",
    ]

    for field in boolean_fields:

        if field not in validation:
            raise ValueError(
                f"Validator response missing "
                f"field: {field}"
            )

        if not isinstance(
            validation[field],
            bool,
        ):
            raise ValueError(
                f"{field} must be boolean"
            )

    if "issues" not in validation:
        raise ValueError(
            "Validator response missing "
            "'issues'"
        )

    if not isinstance(
        validation["issues"],
        list,
    ):
        raise ValueError(
            "'issues' must be a list"
        )

    # Do not blindly trust the LLM's own pass field.
    expected_pass = (
        validation["symptoms_correct"]
        and validation["statuses_correct"]
        and validation["durations_correct"]
        and validation["medications_correct"]
        and validation["allergies_correct"]
        and validation[
            "missing_information_correct"
        ]
        and not validation[
            "extra_information"
        ]
    )

    validation["pass"] = expected_pass


# ============================================================
# REPORT
# ============================================================

def print_report(
    results: list[dict],
) -> None:

    print("\n" + "=" * 60)
    print("SEMANTIC VALIDATION REPORT")
    print("=" * 60)

    total = len(results)

    passed = sum(
        result["validation"]["pass"]
        for result in results
    )

    failed = total - passed

    pass_rate = (
        passed / total * 100
        if total
        else 0
    )

    print(
        f"\nSamples checked: {total}"
    )

    print(
        f"Passed:          {passed}"
    )

    print(
        f"Failed:          {failed}"
    )

    print(
        f"Pass rate:       "
        f"{pass_rate:.1f}%"
    )

    # --------------------------------------------------------
    # BY LANGUAGE
    # --------------------------------------------------------

    language_stats = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
        }
    )

    for result in results:

        language = result["language"]

        language_stats[
            language
        ]["total"] += 1

        if result[
            "validation"
        ]["pass"]:

            language_stats[
                language
            ]["passed"] += 1

    print("\nBy language:")

    for language in sorted(
        language_stats
    ):

        stats = language_stats[
            language
        ]

        print(
            f"  {language:<8} "
            f"{stats['passed']}/"
            f"{stats['total']}"
        )

    # --------------------------------------------------------
    # ERROR CATEGORIES
    # --------------------------------------------------------

    error_counts = Counter()

    fields = [
        "symptoms_correct",
        "statuses_correct",
        "durations_correct",
        "medications_correct",
        "allergies_correct",
        "missing_information_correct",
    ]

    for result in results:

        validation = result[
            "validation"
        ]

        for field in fields:

            if not validation[field]:
                error_counts[field] += 1

        if validation[
            "extra_information"
        ]:
            error_counts[
                "extra_information"
            ] += 1

    print("\nError categories:")

    if not error_counts:
        print("  None")

    else:
        for field, count in (
            error_counts.most_common()
        ):
            print(
                f"  {field:<30} "
                f"{count}"
            )

    # --------------------------------------------------------
    # FAILED SAMPLES
    # --------------------------------------------------------

    failed_results = [
        result
        for result in results
        if not result[
            "validation"
        ]["pass"]
    ]

    if failed_results:

        print("\nFailed samples:")

        for result in failed_results:

            print(
                f"\n  "
                f"{result['case_id']}/"
                f"{result['language']}"
            )

            print(
                f"  Utterance: "
                f"{result['utterance']}"
            )

            for issue in (
                result["validation"][
                    "issues"
                ]
            ):
                print(
                    f"    - {issue}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "SEMANTIC VALIDATION"
    )
    print("=" * 60)

    samples = load_jsonl(
        INPUT_FILE
    )

    existing_keys = (
        load_existing_keys()
    )

    print(
        f"\nSamples: {len(samples)}"
    )

    print(
        f"Already validated: "
        f"{len(existing_keys)}"
    )

    generated = 0
    skipped = 0
    failed_requests = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        case_id = sample["case_id"]
        language = sample["language"]

        key = (
            case_id,
            language,
        )

        print(
            f"\n[{index}/{len(samples)}] "
            f"{case_id}/{language}"
        )

        if key in existing_keys:

            print(
                "  already validated"
            )

            skipped += 1
            continue

        try:

            validation = validate_sample(
                sample
            )

            validate_result_format(
                validation
            )

            result = {
                "case_id": case_id,
                "language": language,
                "utterance":
                    sample["utterance"],
                "validation":
                    validation,
            }

            save_result(
                result
            )

            existing_keys.add(
                key
            )

            generated += 1

            status = (
                "PASS"
                if validation["pass"]
                else "FAIL"
            )

            print(
                f"  {status}"
            )

            if not validation["pass"]:

                for issue in (
                    validation["issues"]
                ):
                    print(
                        f"    - {issue}"
                    )

            time.sleep(0.2)

        except Exception as error:

            failed_requests += 1

            print(
                "  ERROR"
            )

            print(
                f"    {error}"
            )

    # ========================================================
    # LOAD ALL RESULTS FOR FINAL REPORT
    # ========================================================

    results = load_jsonl(
        OUTPUT_FILE
    )

    print_report(
        results
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"New validations: {generated}"
    )

    print(
        f"Skipped:         {skipped}"
    )

    print(
        f"Request errors:  "
        f"{failed_requests}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()