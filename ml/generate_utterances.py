import json
import os
import time

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

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

OUTPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "multilingual_cases_v2.jsonl"
)

# Keep this small while validating the dataset.
# Set to None later to process all cases.
MAX_CASES = 20


LANGUAGES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Modern Standard Arabic",
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
    Build the structured clinical output that Noor should
    learn to predict from the patient utterance.

    Generator-only metadata such as case_id and symptom_group
    is intentionally excluded.
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

    case_json = json.dumps(
        clinical_case,
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

Always use Arabic-Indic digits:

٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩

Examples:

4 hours -> ٤ ساعات
7 days -> ٧ أيام
2 weeks -> أسبوعان / أسبوعين as grammatically appropriate

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

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    utterance = response.output_text.strip()

    # Remove accidental surrounding quotation marks.
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
) -> None:

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
    """
    Allows the generation process to be safely resumed.

    A sample is uniquely identified by:
        (case_id, language)
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
    print(
        "NOOR HEALTH - "
        "MULTILINGUAL DATA GENERATION V2"
    )
    print("=" * 60)

    cases = load_cases()

    existing_keys = load_existing_keys()

    print(
        f"\nClinical cases: {len(cases)}"
    )

    print(
        f"Languages: {len(LANGUAGES)}"
    )

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

        target = build_target(
            clinical_case
        )

        for (
            language_code,
            language_name,
        ) in LANGUAGES.items():

            key = (
                case_id,
                language_code,
            )

            # Allows safe resume after interruption.
            if key in existing_keys:

                print(
                    f"  {language_code}: "
                    "already exists"
                )

                skipped += 1
                continue

            try:

                utterance = generate_utterance(
                    clinical_case,
                    language_name,
                )

                # Deterministically enforce Arabic-Indic
                # digits for MSA.
                if language_code == "ar_msa":
                    utterance = (
                        convert_to_arabic_indic_digits(
                            utterance
                        )
                    )

                sample = {
                    "case_id": case_id,
                    "language": language_code,
                    "utterance": utterance,

                    # Ground truth that Qwen will
                    # eventually learn to produce.
                    "target": target,

                    # Dataset provenance.
                    "source": "synthetic",
                    "schema_version": "v2",
                }

                save_sample(
                    sample
                )

                existing_keys.add(
                    key
                )

                generated += 1

                print(
                    f"  {language_code}: "
                    f"{utterance}"
                )

                time.sleep(0.2)

            except Exception as error:

                failed += 1

                print(
                    f"  {language_code}: ERROR"
                )

                print(
                    f"    {error}"
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
        f"Generated: {generated}"
    )

    print(
        f"Skipped:   {skipped}"
    )

    print(
        f"Failed:    {failed}"
    )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()