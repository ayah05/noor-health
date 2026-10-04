import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Number of simultaneous OpenAI requests.
# Start with 10. If this runs without rate-limit problems,
# you can later try 15.
MAX_WORKERS = 10

# Retry temporarily failed API requests.
MAX_RETRIES = 5

# Retry delays:
# 2s -> 4s -> 8s -> 16s -> 32s
RETRY_BASE_DELAY = 2

# None = process all cases.
MAX_CASES = None


# ============================================================
# INPUT SPLITS
# ============================================================

SPLIT_FILES = {
    "train": (
        PROCESSED_DATA_DIR
        / "train_cases.jsonl"
    ),
    "validation": (
        PROCESSED_DATA_DIR
        / "validation_cases.jsonl"
    ),
    "test": (
        PROCESSED_DATA_DIR
        / "test_cases.jsonl"
    ),
}


OUTPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2_full.jsonl"
)


LANGUAGES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Modern Standard Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "sw": "Swahili",
}


# Prevent multiple threads from writing to the JSONL file
# at exactly the same time.
write_lock = threading.Lock()


# ============================================================
# LOAD CASES
# ============================================================

def load_cases() -> list[dict]:

    cases = []

    for split, input_file in SPLIT_FILES.items():

        if not input_file.exists():

            raise FileNotFoundError(
                f"Missing split file: "
                f"{input_file}"
            )

        with input_file.open(
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

                    clinical_case = (
                        json.loads(line)
                    )

                except json.JSONDecodeError as error:

                    raise ValueError(
                        f"Invalid JSON in "
                        f"{input_file.name}, "
                        f"line {line_number}: "
                        f"{error}"
                    ) from error

                # Keep split only as dataset metadata.
                clinical_case["_split"] = split

                cases.append(
                    clinical_case
                )

    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]

    return cases


# ============================================================
# BUILD TARGET
# ============================================================

def build_target(
    clinical_case: dict,
) -> dict:
    """
    Build the structured output that Noor should learn to
    predict.

    Generator metadata such as case_id, case_pattern,
    symptom_group and split is intentionally excluded.
    """

    return {
        "chief_complaint":
            clinical_case["chief_complaint"],

        "symptoms":
            clinical_case["symptoms"],

        "medications":
            clinical_case["medications"],

        "allergies":
            clinical_case["allergies"],

        "missing_information":
            clinical_case["missing_information"],
    }


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def convert_to_arabic_indic_digits(
    text: str,
) -> str:
    """
    Convert Western digits to Arabic-Indic digits.

    Example:
        4 -> ٤
        12 -> ١٢
    """

    translation_table = str.maketrans(
        "0123456789",
        "٠١٢٣٤٥٦٧٨٩",
    )

    return text.translate(
        translation_table
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    clinical_case: dict,
    language_name: str,
) -> str:

    # Only provide clinically relevant information to the
    # generation model.
    prompt_case = {
        key: value
        for key, value
        in clinical_case.items()
        if key not in {
            "_split",
            "case_pattern",
            "symptom_group",
            "case_id",
        }
    }

    case_json = json.dumps(
        prompt_case,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are generating synthetic multilingual training data for
an offline clinical intake research project.

Generate exactly ONE realistic patient utterance in
{language_name}.

The utterance must preserve the clinical case EXACTLY.

Do not diagnose the patient.
Do not give medical advice.
Do not add medical facts.
Do not add symptoms.
Do not add demographic information.
Do not infer information that is not explicitly provided.


============================================================
SYMPTOMS
============================================================

Each symptom has a status.

"present":
The patient must clearly report having the symptom.

"absent":
The patient must explicitly say they do NOT have the symptom.

"uncertain":
The patient must clearly express uncertainty about whether
they have the symptom.

Do not change one status into another.


============================================================
DURATION
============================================================

Duration belongs to the specific symptom in which it appears.

If a symptom has:

"duration": null

do NOT invent or mention a duration for that symptom.

If a symptom has a duration, preserve both its value and unit
and make it clear which symptom the duration refers to.

Do not transfer a duration from one symptom to another.

GRAMMATICAL DURATION FORM:

The structured duration uses canonical units such as
"hours", "days", and "weeks".

When expressing the duration in natural language, use the
grammatically correct singular or plural form for the
requested language.

For example:

English:
1 day -> "one day"
2 days -> "two days"

German:
1 day -> "einem Tag"
2 days -> "zwei Tagen"

Modern Standard Arabic:
1 day -> "يوم واحد"
2 days -> "يومين"
1 hour -> "ساعة واحدة"
2 hours -> "ساعتين"
1 week -> "أسبوع واحد"
2 weeks -> "أسبوعين"

Do not mechanically copy the canonical plural unit from the
structured data.

The numeric duration value must remain unchanged.


============================================================
MEDICATIONS
============================================================

Medication status has strict semantics.

"unknown":
Do NOT mention medication at all.

"none":
Explicitly state that the patient is not taking medication.

"reported":
Mention exactly the SAME medication or medications listed in
"items".

The medication identity must remain unchanged.

However, the written form of the medication name should be
natural for the requested language.

You may transliterate medication names into the writing
system normally used for that language.

For languages using a non-Latin script, do not simply copy
the English medication name in Latin characters.

For Modern Standard Arabic:
- Write medication names in Arabic script.
- Example:
  "ibuprofen" -> "إيبوبروفين"
  "paracetamol" -> "باراسيتامول"
  "aspirin" -> "أسبرين"
  "amoxicillin" -> "أموكسيسيلين"

For Hindi:
- Write medication names in Devanagari script.

Do not replace a medication with another active ingredient.

Do not add other medications.


============================================================
ALLERGIES
============================================================

Allergy status also has strict semantics.

"unknown":
Do NOT mention allergies at all.

"none":
Explicitly state that the patient reports no known allergies.

"reported":
Mention exactly the allergy or allergies listed in "items".

The identity of the allergy must remain unchanged, but its
written form may be adapted naturally to the requested
language and writing system.

Do not add other allergies.


============================================================
MISSING INFORMATION
============================================================

"missing_information" is metadata for the structured target.

Do NOT turn missing information into facts.

For example:

"medications" in missing_information

means that medications were NOT mentioned by the patient.

It does NOT mean that the patient takes no medication.

Similarly:

"allergies" in missing_information

means allergies were not mentioned.

It does NOT mean that the patient has no allergies.

If "duration" is missing, do NOT invent a duration.

Do not ask follow-up questions in the utterance.


============================================================
LANGUAGE STYLE
============================================================

The utterance should sound like a real patient speaking to a
healthcare worker.

It should NOT sound like:
- a medical report
- a structured questionnaire
- a literal machine translation

Use natural everyday phrasing appropriate for the requested
language.

Vary sentence structure naturally while preserving all
clinical facts exactly.


============================================================
MODERN STANDARD ARABIC
============================================================

If the requested language is Modern Standard Arabic:

- Write in natural, simple Modern Standard Arabic.
- Use Arabic script throughout the entire utterance.
- Do not use Arabic dialects.
- Do not insert English or Latin-script words.
- Medication names must also be written in Arabic script.
- Allergy names must also be written in Arabic script.
- Avoid unnecessarily literary, academic, or overly formal
  expressions.
- Prefer simple expressions that a patient could realistically
  use when speaking with a healthcare worker.
- Preserve the original simplicity of the patient statement.

When a number is naturally written as a digit in Arabic,
use Arabic-Indic digits:

٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩

Examples:

4 hours -> ٤ ساعات
7 days -> ٧ أيام
2 weeks -> أسبوعان / أسبوعين as grammatically appropriate

For singular durations, prefer natural grammatical forms:

1 day -> يوم واحد
1 hour -> ساعة واحدة
1 week -> أسبوع واحد

Do NOT use Western digits 0-9.


============================================================
HINDI
============================================================

If the requested language is Hindi:

- Always use Devanagari script.
- Do not use Latin transliteration.
- Medication names should also be written in Devanagari
  script.


============================================================
IMPORTANT SEMANTIC RULE
============================================================

Natural wording may vary.

Clinical meaning may NOT vary.

The generated utterance must contain exactly the information
represented by the clinical case.

Do not add severity, intensity, frequency, or other clinical
qualifiers unless they are explicitly represented in the
clinical case.

For example, if the case only contains "fatigue", do not
change it to "severe fatigue", "extreme fatigue", or
"mild fatigue".

If severity is not represented in the structured case,
keep the symptom neutral.

No more and no less.


============================================================
CLINICAL CASE
============================================================

{case_json}


============================================================

Return ONLY the patient utterance.

Do not return JSON.
Do not explain the answer.
Do not use quotation marks around the utterance.
""".strip()


# ============================================================
# GENERATE UTTERANCE
# ============================================================

def generate_utterance(
    clinical_case: dict,
    language_name: str,
) -> str:

    prompt = build_prompt(
        clinical_case,
        language_name,
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )

            utterance = (
                response.output_text.strip()
            )

            # Remove accidental surrounding quotation marks.
            if (
                len(utterance) >= 2
                and utterance[0] == '"'
                and utterance[-1] == '"'
            ):
                utterance = (
                    utterance[1:-1]
                )

            return utterance.strip()

        except Exception as error:

            last_error = error

            if attempt == MAX_RETRIES:
                break

            delay = (
                RETRY_BASE_DELAY
                * (2 ** (attempt - 1))
            )

            print(
                f"  API error. "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"in {delay}s: {error}"
            )

            time.sleep(
                delay
            )

    raise last_error


# ============================================================
# SAVE
# ============================================================

def save_sample(
    sample: dict,
) -> None:

    # Only one thread may write at a time.
    with write_lock:

        with OUTPUT_FILE.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# EXISTING SAMPLES / RESUME
# ============================================================

def load_existing_keys() -> set[
    tuple[str, str, str]
]:
    """
    Allows generation to be safely resumed.

    A sample is uniquely identified by:

        (case_id, split, language)
    """

    existing = set()

    if not OUTPUT_FILE.exists():
        return existing

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

                sample = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid existing JSONL "
                    f"at line {line_number}: "
                    f"{error}"
                ) from error

            existing.add(
                (
                    sample["case_id"],
                    sample["split"],
                    sample["language"],
                )
            )

    return existing


# ============================================================
# WORKER
# ============================================================

def process_sample(
    clinical_case: dict,
    language_code: str,
    language_name: str,
) -> dict:

    case_id = clinical_case[
        "case_id"
    ]

    split = clinical_case[
        "_split"
    ]

    target = build_target(
        clinical_case
    )

    utterance = generate_utterance(
        clinical_case,
        language_name,
    )

    # Deterministically enforce Arabic-Indic digits for MSA.
    if language_code == "ar_msa":

        utterance = (
            convert_to_arabic_indic_digits(
                utterance
            )
        )

    return {
        "case_id": case_id,
        "split": split,
        "language": language_code,
        "utterance": utterance,
        "target": target,
        "source": "synthetic",
        "schema_version": "v2",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "PARALLEL MULTILINGUAL DATA GENERATION V2"
    )
    print("=" * 60)

    cases = load_cases()

    existing_keys = (
        load_existing_keys()
    )

    total_possible = (
        len(cases)
        * len(LANGUAGES)
    )

    print(
        f"\nClinical cases: "
        f"{len(cases)}"
    )

    print(
        f"Languages: "
        f"{len(LANGUAGES)}"
    )

    print(
        f"Maximum samples: "
        f"{total_possible}"
    )

    print(
        f"Existing samples: "
        f"{len(existing_keys)}"
    )

    print(
        f"Workers: "
        f"{MAX_WORKERS}"
    )

    # ========================================================
    # BUILD REMAINING JOBS
    # ========================================================

    jobs = []

    skipped = 0

    for clinical_case in cases:

        case_id = clinical_case[
            "case_id"
        ]

        split = clinical_case[
            "_split"
        ]

        for (
            language_code,
            language_name,
        ) in LANGUAGES.items():

            key = (
                case_id,
                split,
                language_code,
            )

            if key in existing_keys:

                skipped += 1

                continue

            jobs.append(
                (
                    clinical_case,
                    language_code,
                    language_name,
                )
            )

    print(
        f"Remaining samples: "
        f"{len(jobs)}"
    )

    if not jobs:

        print(
            "\nNothing to generate. "
            "Dataset is already complete."
        )

        return

    print(
        "\nStarting parallel generation...\n"
    )

    generated = 0
    failed = 0

    start_time = time.time()

    # ========================================================
    # PARALLEL GENERATION
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_job = {}

        for (
            clinical_case,
            language_code,
            language_name,
        ) in jobs:

            future = executor.submit(
                process_sample,
                clinical_case,
                language_code,
                language_name,
            )

            future_to_job[
                future
            ] = (
                clinical_case[
                    "case_id"
                ],
                clinical_case[
                    "_split"
                ],
                language_code,
            )

        for future in as_completed(
            future_to_job
        ):

            (
                case_id,
                split,
                language_code,
            ) = future_to_job[
                future
            ]

            try:

                sample = future.result()

                save_sample(
                    sample
                )

                generated += 1

                completed = (
                    generated
                    + failed
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    generated
                    / elapsed
                    * 60
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    len(jobs)
                    - completed
                )

                eta_minutes = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"{case_id}/{language_code} "
                    f"PASS | "
                    f"{rate:.1f} samples/min | "
                    f"ETA {eta_minutes:.1f} min"
                )

            except Exception as error:

                failed += 1

                completed = (
                    generated
                    + failed
                )

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"{case_id}/{language_code} "
                    f"ERROR"
                )

                print(
                    f"    {error}"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    total_after_run = (
        len(existing_keys)
        + generated
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Generated now: "
        f"{generated}"
    )

    print(
        f"Already existed: "
        f"{skipped}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Total available: "
        f"{total_after_run}"
        f"/{total_possible}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.1f} minutes"
    )

    if elapsed > 0:

        print(
            f"Average rate: "
            f"{generated / elapsed * 60:.1f} "
            f"samples/min"
        )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    if failed > 0:

        print(
            "\nSome samples failed. "
            "Run this script again after completion. "
            "The resume mechanism will generate only "
            "the missing samples."
        )


if __name__ == "__main__":
    main()