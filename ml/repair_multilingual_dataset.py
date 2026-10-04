import json
import os
import shutil
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MAX_WORKERS = 10
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


# ============================================================
# PATHS
# ============================================================

DATASET_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2_full.jsonl"
)

BACKUP_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2_before_repair.jsonl"
)

REPAIR_REPORT_FILE = (
    RESULTS_DIR
    / "multilingual_repair_report.json"
)

CLINICAL_REPAIR_REPORT = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2_repair_report.json"
)

SEMANTIC_VALIDATION_FILE = (
    RESULTS_DIR
    / "semantic_validation.jsonl"
)


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


LANGUAGES = {
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
# JSON HELPERS
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict]:

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


def save_jsonl(
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


# ============================================================
# LOAD CANONICAL CASES
# ============================================================

def load_cases() -> dict[
    tuple[str, str],
    dict,
]:

    cases = {}

    for split, path in SPLIT_FILES.items():

        rows = load_jsonl(
            path
        )

        for case in rows:

            case = dict(case)

            case["_split"] = split

            key = (
                case["case_id"],
                split,
            )

            if key in cases:

                raise ValueError(
                    f"Duplicate clinical case: "
                    f"{key}"
                )

            cases[key] = case

    return cases


# ============================================================
# TARGET
# ============================================================

def build_target(
    clinical_case: dict,
) -> dict:

    return {
        "chief_complaint":
            clinical_case[
                "chief_complaint"
            ],

        "symptoms":
            clinical_case[
                "symptoms"
            ],

        "medications":
            clinical_case[
                "medications"
            ],

        "allergies":
            clinical_case[
                "allergies"
            ],

        "missing_information":
            clinical_case[
                "missing_information"
            ],
    }


# ============================================================
# ARABIC DIGITS
# ============================================================

def convert_to_arabic_indic_digits(
    text: str,
) -> str:

    table = str.maketrans(
        "0123456789",
        "٠١٢٣٤٥٦٧٨٩",
    )

    return text.translate(
        table
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    clinical_case: dict,
    language_name: str,
) -> str:

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

A duration must apply ONLY to the symptom whose structured
target contains that duration.

Never coordinate a symptom that has a duration with another
symptom whose "duration" is null in wording such as:

"X and Y for 3 days"

because this linguistically assigns the duration to BOTH
symptoms.

Instead, phrase them separately so it is completely
unambiguous which symptom the duration belongs to.

Example:

Target:
- cough: duration = 3 days
- headache: duration = null

BAD:
"I have had a cough and headache for 3 days."

GOOD:
"I have had a cough for 3 days. I also have a headache."

When expressing duration naturally, use grammatically correct
singular, plural, or dual forms for the requested language.

The numeric duration value must remain unchanged.


============================================================
MEDICATIONS
============================================================

"unknown":
Do NOT mention medication at all.

"none":
Explicitly state that the patient is not taking medication.

"reported":
Mention exactly the SAME medication or medications listed in
"items".

Do not replace medication identities.
Do not add medications.

For Modern Standard Arabic, medication names must be written
in Arabic script.

For Hindi, medication names must be written in Devanagari.


============================================================
ALLERGIES
============================================================

"unknown":
Do NOT mention allergies at all.

"none":
Explicitly state that the patient reports no known allergies.

"reported":
Mention exactly the allergy or allergies listed in "items".

Do not add allergies.


============================================================
MISSING INFORMATION
============================================================

"missing_information" is target metadata.

Do NOT turn missing information into facts.

"medications" means medication information is NOT mentioned.
It does NOT mean the patient takes no medication.

"allergies" means allergy information is NOT mentioned.
It does NOT mean the patient has no allergies.

"duration" means the chief complaint duration is NOT stated.

Do not ask follow-up questions.


============================================================
LANGUAGE STYLE
============================================================

Use natural everyday patient language.

Do not write like a medical report or structured
questionnaire.

Natural wording may vary.

Clinical meaning may NOT vary.

Do not add severity, frequency, intensity, or other clinical
details that are not represented in the target.


============================================================
MODERN STANDARD ARABIC
============================================================

If the requested language is Modern Standard Arabic:

- Use natural simple Modern Standard Arabic.
- Use Arabic script throughout.
- Do not use dialect.
- Do not use Latin-script words.
- Use Arabic-Indic digits instead of Western digits.

IMPORTANT FOR THE ARABIC DUAL:

When the duration value is exactly 2, use the Arabic dual form
WITHOUT placing the digit ٢ before it.

Correct:
2 hours -> ساعتان / ساعتين
2 days -> يومان / يومين
2 weeks -> أسبوعان / أسبوعين

Incorrect:
٢ ساعتين
٢ يومين
٢ أسبوعين

The dual form itself already expresses the number two.


============================================================
HINDI
============================================================

If the requested language is Hindi:

- Use Devanagari script.
- Do not use Latin transliteration.
- Medication names should also use Devanagari script.


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
# GENERATION
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

            response = (
                client.responses.create(
                    model="gpt-5-mini",
                    input=prompt,
                )
            )

            utterance = (
                response.output_text
                .strip()
            )

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
                f"API error. "
                f"Retry {attempt}/"
                f"{MAX_RETRIES} "
                f"in {delay}s: "
                f"{error}"
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        f"Generation failed: "
        f"{last_error}"
    )


# ============================================================
# BUILD REPAIR KEYS
# ============================================================

def load_repair_keys():

    # --------------------------------------------------------
    # 1. 200 repaired clinical cases
    # --------------------------------------------------------

    if not CLINICAL_REPAIR_REPORT.exists():

        raise FileNotFoundError(
            f"Missing repair report:\n"
            f"{CLINICAL_REPAIR_REPORT}"
        )

    report = json.loads(
        CLINICAL_REPAIR_REPORT.read_text(
            encoding="utf-8"
        )
    )

    repaired_case_ids = set(
        report[
            "changed_case_ids"
        ]
    )

    # --------------------------------------------------------
    # 2. Existing semantic/language failures
    # --------------------------------------------------------

    validation_rows = load_jsonl(
        SEMANTIC_VALIDATION_FILE
    )

    qc_failure_keys = set()

    for row in validation_rows:

        if (
            not row[
                "semantic_passed"
            ]
            or not row[
                "language_passed"
            ]
        ):

            qc_failure_keys.add(
                (
                    row["case_id"],
                    row["split"],
                    row["language"],
                )
            )

    return (
        repaired_case_ids,
        qc_failure_keys,
    )


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(
    clinical_case: dict,
    language_code: str,
    language_name: str,
) -> dict:

    utterance = generate_utterance(
        clinical_case,
        language_name,
    )

    if language_code == "ar_msa":

        utterance = (
            convert_to_arabic_indic_digits(
                utterance
            )
        )

    return {
        "case_id":
            clinical_case["case_id"],

        "split":
            clinical_case["_split"],

        "language":
            language_code,

        "utterance":
            utterance,

        "target":
            build_target(
                clinical_case
            ),

        "source":
            "synthetic",

        "schema_version":
            "v2",
    }


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_final_dataset(
    samples: list[dict],
):

    if len(samples) != 7000:

        raise ValueError(
            f"Expected 7000 samples, "
            f"found {len(samples)}"
        )

    keys = [
        (
            sample["case_id"],
            sample["split"],
            sample["language"],
        )
        for sample in samples
    ]

    if len(keys) != len(set(keys)):

        raise ValueError(
            "Duplicate sample keys detected"
        )

    case_languages = {}

    for sample in samples:

        case_id = sample[
            "case_id"
        ]

        case_languages.setdefault(
            case_id,
            set(),
        ).add(
            sample["language"]
        )

    if len(case_languages) != 1000:

        raise ValueError(
            f"Expected 1000 case IDs, "
            f"found {len(case_languages)}"
        )

    expected_languages = set(
        LANGUAGES.keys()
    )

    for (
        case_id,
        languages,
    ) in case_languages.items():

        if languages != expected_languages:

            raise ValueError(
                f"{case_id}: "
                "incorrect language set: "
                f"{sorted(languages)}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "MULTILINGUAL DATASET REPAIR"
    )
    print("=" * 60)

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    old_samples = load_jsonl(
        DATASET_FILE
    )

    clinical_cases = load_cases()

    (
        repaired_case_ids,
        qc_failure_keys,
    ) = load_repair_keys()

    print(
        f"\nExisting samples: "
        f"{len(old_samples)}"
    )

    print(
        f"Repaired clinical cases: "
        f"{len(repaired_case_ids)}"
    )

    print(
        f"Existing QC failures: "
        f"{len(qc_failure_keys)}"
    )

    # --------------------------------------------------------
    # BUILD ALL REPAIR KEYS
    # --------------------------------------------------------

    repair_keys = set()

    for (
        case_id,
        split,
    ), clinical_case in (
        clinical_cases.items()
    ):

        if (
            case_id
            not in repaired_case_ids
        ):
            continue

        for language_code in LANGUAGES:

            repair_keys.add(
                (
                    case_id,
                    split,
                    language_code,
                )
            )

    repair_keys.update(
        qc_failure_keys
    )

    overlap = (
        len(repaired_case_ids) * 7
        + len(qc_failure_keys)
        - len(repair_keys)
    )

    print(
        f"\nRepair keys from "
        f"clinical changes: "
        f"{len(repaired_case_ids) * 7}"
    )

    print(
        f"QC overlap: "
        f"{overlap}"
    )

    print(
        f"Unique samples to regenerate: "
        f"{len(repair_keys)}"
    )

    # --------------------------------------------------------
    # VALIDATE KEYS
    # --------------------------------------------------------

    old_by_key = {
        (
            sample["case_id"],
            sample["split"],
            sample["language"],
        ): sample
        for sample in old_samples
    }

    if len(old_by_key) != len(
        old_samples
    ):

        raise ValueError(
            "Existing multilingual dataset "
            "contains duplicate keys"
        )

    missing_keys = (
        repair_keys
        - set(old_by_key.keys())
    )

    if missing_keys:

        raise ValueError(
            "Repair keys missing from "
            "existing dataset: "
            f"{sorted(missing_keys)[:5]}"
        )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    if not BACKUP_FILE.exists():

        shutil.copy2(
            DATASET_FILE,
            BACKUP_FILE,
        )

        print(
            f"\nBackup created:\n"
            f"{BACKUP_FILE}"
        )

    else:

        print(
            "\nBackup already exists."
        )

    # --------------------------------------------------------
    # BUILD JOBS
    # --------------------------------------------------------

    jobs = []

    for key in sorted(
        repair_keys
    ):

        (
            case_id,
            split,
            language_code,
        ) = key

        clinical_key = (
            case_id,
            split,
        )

        if clinical_key not in clinical_cases:

            raise ValueError(
                f"Clinical case missing: "
                f"{clinical_key}"
            )

        clinical_case = (
            clinical_cases[
                clinical_key
            ]
        )

        jobs.append(
            (
                key,
                clinical_case,
                language_code,
                LANGUAGES[
                    language_code
                ],
            )
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print(
        "\nStarting repair generation...\n"
    )

    generated = {}
    failed = []

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_key = {}

        for (
            key,
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

            future_to_key[
                future
            ] = key

        for future in as_completed(
            future_to_key
        ):

            key = future_to_key[
                future
            ]

            try:

                generated[key] = (
                    future.result()
                )

                completed = (
                    len(generated)
                    + len(failed)
                )

                print(
                    f"[{completed}/"
                    f"{len(jobs)}] "
                    f"{key[0]}/"
                    f"{key[2]} PASS"
                )

            except Exception as error:

                failed.append(
                    {
                        "key": key,
                        "error":
                            str(error),
                    }
                )

                completed = (
                    len(generated)
                    + len(failed)
                )

                print(
                    f"[{completed}/"
                    f"{len(jobs)}] "
                    f"{key[0]}/"
                    f"{key[2]} ERROR"
                )

                print(
                    f"    {error}"
                )

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT TOUCH DATASET IF ANY GENERATION FAILED
    # --------------------------------------------------------

    if failed:

        print(
            "\nRepair aborted."
        )

        print(
            f"Failed samples: "
            f"{len(failed)}"
        )

        print(
            "Original dataset was NOT "
            "modified."
        )

        return

    # --------------------------------------------------------
    # REPLACE IN ORIGINAL ORDER
    # --------------------------------------------------------

    final_samples = []

    replaced = 0

    for old_sample in old_samples:

        key = (
            old_sample["case_id"],
            old_sample["split"],
            old_sample["language"],
        )

        if key in generated:

            final_samples.append(
                generated[key]
            )

            replaced += 1

        else:

            final_samples.append(
                old_sample
            )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    validate_final_dataset(
        final_samples
    )

    if replaced != len(
        repair_keys
    ):

        raise ValueError(
            f"Expected to replace "
            f"{len(repair_keys)} samples, "
            f"replaced {replaced}"
        )

    # --------------------------------------------------------
    # SAVE ONLY AFTER EVERYTHING PASSES
    # --------------------------------------------------------

    save_jsonl(
        DATASET_FILE,
        final_samples,
    )

    elapsed = (
        time.time()
        - start_time
    )

    report = {
        "original_samples":
            len(old_samples),

        "repaired_clinical_cases":
            len(repaired_case_ids),

        "clinical_change_keys":
            len(repaired_case_ids) * 7,

        "previous_qc_failures":
            len(qc_failure_keys),

        "qc_overlap":
            overlap,

        "unique_regenerated_samples":
            len(repair_keys),

        "final_samples":
            len(final_samples),

        "runtime_minutes":
            elapsed / 60,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPAIR_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)

    print(
        f"\nSamples regenerated: "
        f"{len(generated)}"
    )

    print(
        f"Samples replaced: "
        f"{replaced}"
    )

    print(
        f"Final dataset size: "
        f"{len(final_samples)}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "\nFinal structural checks: "
        "PASSED"
    )

    print(
        f"\nDataset:\n"
        f"{DATASET_FILE}"
    )

    print(
        f"\nReport:\n"
        f"{REPAIR_REPORT_FILE}"
    )


if __name__ == "__main__":
    main()