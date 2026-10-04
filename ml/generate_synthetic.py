import json
import random

from config import SYNTHETIC_DATA_DIR, RANDOM_SEED


# ============================================================
# CONFIG
# ============================================================

NUM_CASES = 20

random.seed(RANDOM_SEED)


# ============================================================
# CLINICAL VOCABULARY
# ============================================================

SYMPTOM_GROUPS = {
    "respiratory": {
        "chief_complaints": [
            "cough",
            "sore throat",
            "shortness of breath",
        ],
        "related_symptoms": [
            "fever",
            "fatigue",
            "headache",
        ],
    },

    "gastrointestinal": {
        "chief_complaints": [
            "abdominal pain",
            "nausea",
            "vomiting",
            "diarrhea",
        ],
        "related_symptoms": [
            "fever",
            "fatigue",
            "dizziness",
        ],
    },

    "general": {
        "chief_complaints": [
            "headache",
            "dizziness",
            "fatigue",
            "fever",
        ],
        "related_symptoms": [
            "nausea",
        ],
    },

    "pain": {
        "chief_complaints": [
            "back pain",
            "headache",
            "abdominal pain",
        ],
        "related_symptoms": [
            "fatigue",
            "dizziness",
        ],
    },
}


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


SYMPTOM_STATUSES = [
    "present",
    "absent",
    "uncertain",
]


# ============================================================
# DURATION
# ============================================================

def generate_duration():

    unit = random.choice(
        [
            "hours",
            "days",
            "weeks",
        ]
    )

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


def generate_optional_duration(
    probability=0.80,
):

    if random.random() < probability:
        return generate_duration()

    return None


# ============================================================
# MEDICATIONS
# ============================================================

def generate_medications():

    status = random.choices(
        population=[
            "unknown",
            "none",
            "reported",
        ],
        weights=[
            0.45,
            0.25,
            0.30,
        ],
        k=1,
    )[0]

    if status == "reported":

        items = random.sample(
            MEDICATIONS,
            k=1,
        )

    else:
        items = []

    return {
        "status": status,
        "items": items,
    }


# ============================================================
# ALLERGIES
# ============================================================

def generate_allergies():

    status = random.choices(
        population=[
            "unknown",
            "none",
            "reported",
        ],
        weights=[
            0.50,
            0.30,
            0.20,
        ],
        k=1,
    )[0]

    if status == "reported":

        items = random.sample(
            ALLERGIES,
            k=1,
        )

    else:
        items = []

    return {
        "status": status,
        "items": items,
    }


# ============================================================
# SYMPTOMS
# ============================================================

def generate_symptoms():

    group_name = random.choice(
        list(SYMPTOM_GROUPS.keys())
    )

    group = SYMPTOM_GROUPS[group_name]

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    symptoms = [
        {
            "name": chief_complaint,
            "status": "present",
            "duration": generate_optional_duration(
                probability=0.80
            ),
        }
    ]

    # Additional symptoms can come from both the other
    # chief complaints and related symptoms.
    possible_additional = (
        [
            symptom
            for symptom in group["chief_complaints"]
            if symptom != chief_complaint
        ]
        + group["related_symptoms"]
    )

    # Remove possible duplicates while preserving order.
    possible_additional = list(
        dict.fromkeys(
            possible_additional
        )
    )

    number_additional = random.randint(
        0,
        min(
            3,
            len(possible_additional),
        ),
    )

    selected_symptoms = random.sample(
        possible_additional,
        k=number_additional,
    )

    for symptom_name in selected_symptoms:

        status = random.choices(
            population=[
                "present",
                "absent",
                "uncertain",
            ],
            weights=[
                0.55,
                0.25,
                0.20,
            ],
            k=1,
        )[0]

        duration = None

        # Only a symptom that is actually present
        # may receive a duration.
        if (
            status == "present"
            and random.random() < 0.30
        ):
            duration = generate_duration()

        symptoms.append(
            {
                "name": symptom_name,
                "status": status,
                "duration": duration,
            }
        )

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# MISSING INFORMATION
# ============================================================

def determine_missing_information(
    symptoms,
    chief_complaint,
    medications,
    allergies,
):

    missing = []

    chief = next(
        symptom
        for symptom in symptoms
        if symptom["name"] == chief_complaint
    )

    if chief["duration"] is None:
        missing.append("duration")

    if medications["status"] == "unknown":
        missing.append("medications")

    if allergies["status"] == "unknown":
        missing.append("allergies")

    return missing


# ============================================================
# CASE GENERATION
# ============================================================

def generate_case(case_number: int):

    (
        symptom_group,
        chief_complaint,
        symptoms,
    ) = generate_symptoms()

    medications = generate_medications()
    allergies = generate_allergies()

    missing_information = (
        determine_missing_information(
            symptoms=symptoms,
            chief_complaint=chief_complaint,
            medications=medications,
            allergies=allergies,
        )
    )

    return {
        "case_id": (
            f"case_{case_number:04d}"
        ),
        "symptom_group": symptom_group,
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        "medications": medications,
        "allergies": allergies,
        "missing_information": (
            missing_information
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_case(case):

    symptoms = case["symptoms"]

    # --------------------------------------------------------
    # Chief complaint
    # --------------------------------------------------------

    chief_matches = [
        symptom
        for symptom in symptoms
        if (
            symptom["name"]
            == case["chief_complaint"]
        )
    ]

    if len(chief_matches) != 1:
        raise ValueError(
            f"{case['case_id']}: "
            "chief complaint missing "
            "or duplicated"
        )

    chief = chief_matches[0]

    if chief["status"] != "present":
        raise ValueError(
            f"{case['case_id']}: "
            "chief complaint must be present"
        )

    # --------------------------------------------------------
    # Symptom validation
    # --------------------------------------------------------

    symptom_names = [
        symptom["name"]
        for symptom in symptoms
    ]

    if len(symptom_names) != len(
        set(symptom_names)
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "duplicate symptoms"
        )

    for symptom in symptoms:

        if symptom["status"] not in {
            "present",
            "absent",
            "uncertain",
        }:
            raise ValueError(
                f"{case['case_id']}: "
                f"invalid status for "
                f"{symptom['name']}"
            )

        if (
            symptom["status"]
            in {"absent", "uncertain"}
            and symptom["duration"] is not None
        ):
            raise ValueError(
                f"{case['case_id']}: "
                f"{symptom['name']} "
                "must not have duration"
            )

    # --------------------------------------------------------
    # Medication validation
    # --------------------------------------------------------

    medications = case["medications"]

    if (
        medications["status"] == "reported"
        and not medications["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "reported medication "
            "requires an item"
        )

    if (
        medications["status"] != "reported"
        and medications["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "medication items must be empty"
        )

    # --------------------------------------------------------
    # Allergy validation
    # --------------------------------------------------------

    allergies = case["allergies"]

    if (
        allergies["status"] == "reported"
        and not allergies["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "reported allergy "
            "requires an item"
        )

    if (
        allergies["status"] != "reported"
        and allergies["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "allergy items must be empty"
        )

    # --------------------------------------------------------
    # Missing information validation
    # --------------------------------------------------------

    expected_missing = (
        determine_missing_information(
            symptoms=symptoms,
            chief_complaint=(
                case["chief_complaint"]
            ),
            medications=medications,
            allergies=allergies,
        )
    )

    if (
        case["missing_information"]
        != expected_missing
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "incorrect missing_information"
        )


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
        / "clinical_cases_v2.jsonl"
    )

    cases = []

    for i in range(
        1,
        NUM_CASES + 1,
    ):

        case = generate_case(i)

        validate_case(case)

        cases.append(case)

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
    print(
        "NOOR HEALTH - "
        "SYNTHETIC CLINICAL CASES V2"
    )
    print("=" * 60)

    print(
        f"\nGenerated: {len(cases)} cases"
    )

    print(
        f"Saved to: {output_file}"
    )

    print("\nExamples:\n")

    for case in cases[:3]:

        print(
            json.dumps(
                case,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("-" * 60)


if __name__ == "__main__":
    main()