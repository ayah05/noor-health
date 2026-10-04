import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import SYNTHETIC_DATA_DIR


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

INPUT_FILE = SYNTHETIC_DATA_DIR / "clinical_cases.jsonl"
OUTPUT_FILE = SYNTHETIC_DATA_DIR / "multilingual_cases.jsonl"


LANGUAGES = {
    "en": "English",
    "de": "German",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "sw": "Swahili",
}


# ============================================================
# LOAD CASES
# ============================================================

def load_cases() -> list[dict]:

    cases = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                cases.append(
                    json.loads(line)
                )

    return cases


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    clinical_case: dict,
    language_name: str,
) -> str:

    case_json = json.dumps(
        clinical_case,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are generating synthetic multilingual data for a research
project about offline clinical intake.

Generate ONE realistic patient utterance in {language_name}.

The patient utterance must express ONLY information contained
in the clinical case below.

Do not diagnose the patient.
Do not add medical facts.
Do not add symptoms that are not present in the case.
Do not add demographic information.

IMPORTANT:

Symptoms have one of three statuses:

present:
The patient clearly reports having the symptom.

absent:
The patient clearly says they do NOT have the symptom.

uncertain:
The patient expresses uncertainty about whether they have
the symptom.

The language should sound like a real patient speaking,
not like a medical report.

Use natural everyday language.

For Arabic, use natural conversational Arabic rather than
formal medical terminology where appropriate.

For all languages, avoid literal translations and use
natural phrasing.

Clinical case:

{case_json}

Return ONLY the patient utterance.
Do not return JSON.
Do not explain your answer.
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

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    utterance = response.output_text.strip()

    # Remove accidental surrounding quotes
    if (
        len(utterance) >= 2
        and utterance[0] == '"'
        and utterance[-1] == '"'
    ):
        utterance = utterance[1:-1]

    return utterance.strip()


# ============================================================
# SAVE
# ============================================================

def save_sample(
    sample: dict,
):

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
# EXISTING SAMPLES
# ============================================================

def load_existing_keys() -> set[tuple[str, str]]:

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

            sample = json.loads(line)

            existing.add(
                (
                    sample["case_id"],
                    sample["language"],
                )
            )

    return existing


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NOOR HEALTH - MULTILINGUAL DATA GENERATION")
    print("=" * 60)

    cases = load_cases()

    existing_keys = load_existing_keys()

    print(f"\nClinical cases: {len(cases)}")
    print(f"Languages: {len(LANGUAGES)}")
    print(
        f"Maximum samples: "
        f"{len(cases) * len(LANGUAGES)}"
    )

    generated = 0
    skipped = 0
    failed = 0

    for case_index, clinical_case in enumerate(
        cases,
        start=1,
    ):

        case_id = clinical_case["case_id"]

        print(
            f"\n[{case_index}/{len(cases)}] "
            f"{case_id}"
        )

        for language_code, language_name in LANGUAGES.items():

            key = (
                case_id,
                language_code,
            )

            # Allows script to resume safely
            if key in existing_keys:

                print(
                    f"  {language_code}: already exists"
                )

                skipped += 1
                continue

            try:

                utterance = generate_utterance(
                    clinical_case,
                    language_name,
                )

                sample = {
                    "case_id": case_id,
                    "language": language_code,
                    "utterance": utterance,

                    # Ground truth
                    "target": {
                        "chief_complaint":
                            clinical_case["chief_complaint"],

                        "symptoms":
                            clinical_case["symptoms"],

                        "duration":
                            clinical_case["duration"],

                        "medications":
                            clinical_case["medications"],

                        "allergies":
                            clinical_case["allergies"],
                    },

                    # Dataset provenance
                    "source": "synthetic",
                }

                save_sample(sample)

                existing_keys.add(key)

                generated += 1

                print(
                    f"  {language_code}: "
                    f"{utterance}"
                )

                # Small delay to avoid hammering API
                time.sleep(0.2)

            except Exception as error:

                failed += 1

                print(
                    f"  {language_code}: ERROR"
                )

                print(
                    f"    {error}"
                )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()