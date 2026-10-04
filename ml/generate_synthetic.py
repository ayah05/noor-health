import json
import random
from pathlib import Path

from config import SYNTHETIC_DATA_DIR, RANDOM_SEED


# ============================================================
# CONFIG
# ============================================================

NUM_CASES = 200

random.seed(RANDOM_SEED)


# ============================================================
# CLINICAL VOCABULARY
# ============================================================

SYMPTOMS = [
    "headache",
    "abdominal pain",
    "nausea",
    "dizziness",
    "cough",
    "sore throat",
    "fatigue",
    "fever",
    "vomiting",
    "diarrhea",
    "back pain",
    "shortness of breath",
]

MEDICATIONS = [
    "ibuprofen",
    "paracetamol",
    "aspirin",
    "amoxicillin",
]

ALLERGIES = [
    "penicillin",
    "aspirin",
    "ibuprofen",
]

DURATION_UNITS = [
    "hours",
    "days",
    "weeks",
]

STATUSES = [
    "present",
    "absent",
    "uncertain",
]


# ============================================================
# GENERATION
# ============================================================

def generate_duration():

    unit = random.choice(DURATION_UNITS)

    if unit == "hours":
        value = random.randint(1, 12)

    elif unit == "days":
        value = random.randint(1, 7)

    else:
        value = random.randint(1, 3)

    return {
        "value": value,
        "unit": unit,
    }


def generate_case(case_number: int):

    chief_complaint = random.choice(SYMPTOMS)

    other_symptoms = [
        symptom
        for symptom in SYMPTOMS
        if symptom != chief_complaint
    ]

    selected_symptoms = random.sample(
        other_symptoms,
        k=random.randint(0, 3),
    )

    symptoms = [
        {
            "name": chief_complaint,
            "status": "present",
        }
    ]

    for symptom in selected_symptoms:

        symptoms.append(
            {
                "name": symptom,
                "status": random.choice(STATUSES),
            }
        )

    medications = []

    if random.random() < 0.35:
        medications = random.sample(
            MEDICATIONS,
            k=1,
        )

    allergies = []

    if random.random() < 0.20:
        allergies = random.sample(
            ALLERGIES,
            k=1,
        )

    return {
        "case_id": f"case_{case_number:04d}",
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        "duration": generate_duration(),
        "medications": medications,
        "allergies": allergies,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    SYNTHETIC_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        SYNTHETIC_DATA_DIR
        / "clinical_cases.jsonl"
    )

    cases = [
        generate_case(i)
        for i in range(1, NUM_CASES + 1)
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for case in cases:

            file.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 60)
    print("NOOR HEALTH - SYNTHETIC CLINICAL CASES")
    print("=" * 60)

    print(f"\nGenerated: {len(cases)} cases")
    print(f"Saved to: {output_file}")

    print("\nExample:\n")

    print(
        json.dumps(
            cases[0],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()