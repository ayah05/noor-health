import json
import random
from collections import Counter

from config import SYNTHETIC_DATA_DIR, RANDOM_SEED


# ============================================================
# CONFIG
# ============================================================

NUM_CASES = 1000

OUTPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "clinical_cases_v2.jsonl"
)

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
            "runny nose",
        ],
        "related_symptoms": [
            "fever",
            "fatigue",
            "headache",
            "chest pain",
        ],
    },

    "gastrointestinal": {
        "chief_complaints": [
            "abdominal pain",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
        ],
        "related_symptoms": [
            "fever",
            "fatigue",
            "dizziness",
            "loss of appetite",
        ],
    },

    "general": {
        "chief_complaints": [
            "headache",
            "dizziness",
            "fatigue",
            "fever",
            "weakness",
        ],
        "related_symptoms": [
            "nausea",
            "loss of appetite",
        ],
    },

    "pain": {
        "chief_complaints": [
            "back pain",
            "headache",
            "abdominal pain",
            "chest pain",
            "joint pain",
        ],
        "related_symptoms": [
            "fatigue",
            "dizziness",
        ],
    },

    "urinary": {
        "chief_complaints": [
            "painful urination",
            "frequent urination",
        ],
        "related_symptoms": [
            "fever",
            "abdominal pain",
            "back pain",
        ],
    },

    "skin": {
        "chief_complaints": [
            "rash",
            "itching",
        ],
        "related_symptoms": [
            "fever",
            "swelling",
            "pain",
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


# ============================================================
# CASE PATTERNS
# ============================================================

# These patterns deliberately create the phenomena that the
# extraction model needs to learn.
#
# Counts add up to exactly 1000 when NUM_CASES == 1000.

CASE_PATTERN_WEIGHTS = {
    "simple": 0.15,
    "negation": 0.15,
    "uncertainty": 0.15,
    "mixed": 0.15,
    "missing_duration": 0.10,
    "missing_medications": 0.10,
    "missing_allergies": 0.10,
    "multiple_present": 0.10,
}


# ============================================================
# DURATION
# ============================================================

def generate_duration():
    """
    Generate a short duration suitable for primary-care
    symptom descriptions.
    """

    unit = random.choice(
        [
            "hours",
            "days",
            "weeks",
        ]
    )

    if unit == "hours":
        value = random.randint(
            1,
            12,
        )

    elif unit == "days":
        value = random.randint(
            1,
            7,
        )

    else:
        value = random.randint(
            1,
            3,
        )

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

def generate_medications(
    force_status=None,
):
    """
    Medication semantics:

    unknown
        Medication information is not mentioned.

    none
        Patient explicitly reports no medication.

    reported
        One medication is explicitly reported.
    """

    if force_status is not None:
        status = force_status

    else:
        status = random.choices(
            population=[
                "unknown",
                "none",
                "reported",
            ],
            weights=[
                0.35,
                0.30,
                0.35,
            ],
            k=1,
        )[0]

    if status == "reported":
        items = [
            random.choice(
                MEDICATIONS
            )
        ]

    else:
        items = []

    return {
        "status": status,
        "items": items,
    }


# ============================================================
# ALLERGIES
# ============================================================

def generate_allergies(
    medications,
    force_status=None,
):
    """
    Generate allergy information.

    A reported medication is excluded from the allergy
    candidates to avoid accidentally creating medication /
    allergy conflicts in ordinary synthetic cases.
    """

    if force_status is not None:
        status = force_status

    else:
        status = random.choices(
            population=[
                "unknown",
                "none",
                "reported",
            ],
            weights=[
                0.35,
                0.35,
                0.30,
            ],
            k=1,
        )[0]

    if status != "reported":
        return {
            "status": status,
            "items": [],
        }

    medication_items = set(
        medications["items"]
    )

    possible_allergies = [
        allergy
        for allergy in ALLERGIES
        if allergy not in medication_items
    ]

    if not possible_allergies:
        return {
            "status": "none",
            "items": [],
        }

    return {
        "status": "reported",
        "items": [
            random.choice(
                possible_allergies
            )
        ],
    }


# ============================================================
# SYMPTOM HELPERS
# ============================================================

def choose_group():
    group_name = random.choice(
        list(
            SYMPTOM_GROUPS.keys()
        )
    )

    return (
        group_name,
        SYMPTOM_GROUPS[group_name],
    )


def get_additional_symptom_pool(
    group,
    chief_complaint,
):
    """
    Additional symptoms may include other chief complaints
    from the same group plus related symptoms.
    """

    symptoms = (
        [
            symptom
            for symptom
            in group["chief_complaints"]
            if symptom != chief_complaint
        ]
        + group["related_symptoms"]
    )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(symptoms)
    )


def create_symptom(
    name,
    status,
    duration=None,
):
    if status != "present":
        duration = None

    return {
        "name": name,
        "status": status,
        "duration": duration,
    }


# ============================================================
# PATTERN: SIMPLE
# ============================================================

def generate_simple_case():
    """
    One chief complaint only.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=generate_optional_duration(
                probability=0.85
            ),
        )
    ]

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# PATTERN: NEGATION
# ============================================================

def generate_negation_case():
    """
    Chief complaint plus at least one explicitly absent
    symptom.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    pool = get_additional_symptom_pool(
        group,
        chief_complaint,
    )

    negative_symptom = random.choice(
        pool
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=generate_optional_duration(
                probability=0.85
            ),
        ),
        create_symptom(
            name=negative_symptom,
            status="absent",
        ),
    ]

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# PATTERN: UNCERTAINTY
# ============================================================

def generate_uncertainty_case():
    """
    Chief complaint plus at least one explicitly uncertain
    symptom.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    pool = get_additional_symptom_pool(
        group,
        chief_complaint,
    )

    uncertain_symptom = random.choice(
        pool
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=generate_optional_duration(
                probability=0.85
            ),
        ),
        create_symptom(
            name=uncertain_symptom,
            status="uncertain",
        ),
    ]

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# PATTERN: MIXED
# ============================================================

def generate_mixed_case():
    """
    Combination of present, absent and uncertain symptoms.

    This is one of the most important patterns for testing
    whether the extraction model understands semantic status.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    pool = get_additional_symptom_pool(
        group,
        chief_complaint,
    )

    number_additional = min(
        3,
        len(pool),
    )

    selected = random.sample(
        pool,
        k=number_additional,
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=generate_optional_duration(
                probability=0.85
            ),
        )
    ]

    statuses = [
        "present",
        "absent",
        "uncertain",
    ]

    random.shuffle(
        statuses
    )

    for symptom_name, status in zip(
        selected,
        statuses,
    ):
        duration = None

        if (
            status == "present"
            and random.random() < 0.40
        ):
            duration = generate_duration()

        symptoms.append(
            create_symptom(
                name=symptom_name,
                status=status,
                duration=duration,
            )
        )

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# PATTERN: MISSING DURATION
# ============================================================

def generate_missing_duration_case():
    """
    Chief complaint is present but duration is intentionally
    unknown.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=None,
        )
    ]

    # Sometimes include one additional symptom to prevent
    # all missing-duration examples from looking identical.
    if random.random() < 0.50:

        pool = get_additional_symptom_pool(
            group,
            chief_complaint,
        )

        symptom_name = random.choice(
            pool
        )

        status = random.choice(
            [
                "present",
                "absent",
                "uncertain",
            ]
        )

        duration = None

        if (
            status == "present"
            and random.random() < 0.30
        ):
            duration = generate_duration()

        symptoms.append(
            create_symptom(
                name=symptom_name,
                status=status,
                duration=duration,
            )
        )

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# PATTERN: MULTIPLE PRESENT
# ============================================================

def generate_multiple_present_case():
    """
    Multiple simultaneously present symptoms.

    Some additional present symptoms receive their own
    duration. This helps teach the model not to transfer
    durations between symptoms.
    """

    group_name, group = (
        choose_group()
    )

    chief_complaint = random.choice(
        group["chief_complaints"]
    )

    pool = get_additional_symptom_pool(
        group,
        chief_complaint,
    )

    number_additional = min(
        random.randint(
            1,
            3,
        ),
        len(pool),
    )

    selected = random.sample(
        pool,
        k=number_additional,
    )

    symptoms = [
        create_symptom(
            name=chief_complaint,
            status="present",
            duration=generate_optional_duration(
                probability=0.90
            ),
        )
    ]

    for symptom_name in selected:

        duration = (
            generate_duration()
            if random.random() < 0.50
            else None
        )

        symptoms.append(
            create_symptom(
                name=symptom_name,
                status="present",
                duration=duration,
            )
        )

    return (
        group_name,
        chief_complaint,
        symptoms,
    )


# ============================================================
# GENERATE SYMPTOMS BY PATTERN
# ============================================================

def generate_symptoms(
    case_pattern,
):
    if case_pattern == "simple":
        return generate_simple_case()

    if case_pattern == "negation":
        return generate_negation_case()

    if case_pattern == "uncertainty":
        return generate_uncertainty_case()

    if case_pattern == "mixed":
        return generate_mixed_case()

    if case_pattern in {
        "missing_duration",
        "missing_medications",
        "missing_allergies",
    }:
        return generate_missing_duration_case()

    if case_pattern == "multiple_present":
        return generate_multiple_present_case()

    raise ValueError(
        f"Unknown case pattern: "
        f"{case_pattern}"
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
        if symptom["name"]
        == chief_complaint
    )

    if chief["duration"] is None:
        missing.append(
            "duration"
        )

    if (
        medications["status"]
        == "unknown"
    ):
        missing.append(
            "medications"
        )

    if (
        allergies["status"]
        == "unknown"
    ):
        missing.append(
            "allergies"
        )

    return missing


# ============================================================
# PATTERN SCHEDULE
# ============================================================

def build_pattern_schedule(
    num_cases,
):
    """
    Build a deterministic approximately balanced schedule.

    For NUM_CASES=1000 this produces exactly:
        simple               150
        negation             150
        uncertainty          150
        mixed                150
        missing_duration     100
        missing_medications  100
        missing_allergies    100
        multiple_present     100
    """

    patterns = []

    remaining = num_cases

    items = list(
        CASE_PATTERN_WEIGHTS.items()
    )

    for index, (
        pattern,
        weight,
    ) in enumerate(items):

        if index == len(items) - 1:
            count = remaining

        else:
            count = round(
                num_cases * weight
            )

            remaining -= count

        patterns.extend(
            [pattern] * count
        )

    random.shuffle(
        patterns
    )

    return patterns


# ============================================================
# CASE GENERATION
# ============================================================

def generate_case(
    case_number,
    case_pattern,
):
    (
        symptom_group,
        chief_complaint,
        symptoms,
    ) = generate_symptoms(
        case_pattern
    )

    # --------------------------------------------------------
    # MEDICATION STATUS
    # --------------------------------------------------------

    if (
        case_pattern
        == "missing_medications"
    ):
        medications = (
            generate_medications(
                force_status="unknown"
            )
        )

    else:
        medications = (
            generate_medications()
        )

    # --------------------------------------------------------
    # ALLERGY STATUS
    # --------------------------------------------------------

    if (
        case_pattern
        == "missing_allergies"
    ):
        allergies = (
            generate_allergies(
                medications=medications,
                force_status="unknown",
            )
        )

    else:
        allergies = (
            generate_allergies(
                medications=medications,
            )
        )

    missing_information = (
        determine_missing_information(
            symptoms=symptoms,
            chief_complaint=(
                chief_complaint
            ),
            medications=medications,
            allergies=allergies,
        )
    )

    return {
        "case_id": (
            f"case_{case_number:04d}"
        ),

        # Generator metadata.
        # Do NOT include this in the Qwen target.
        "case_pattern":
            case_pattern,

        "symptom_group":
            symptom_group,

        "chief_complaint":
            chief_complaint,

        "symptoms":
            symptoms,

        "medications":
            medications,

        "allergies":
            allergies,

        "missing_information":
            missing_information,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_case(
    case,
):
    symptoms = case["symptoms"]

    # --------------------------------------------------------
    # CHIEF COMPLAINT
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

    if (
        chief["status"]
        != "present"
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "chief complaint must "
            "be present"
        )

    # --------------------------------------------------------
    # DUPLICATE SYMPTOMS
    # --------------------------------------------------------

    symptom_names = [
        symptom["name"]
        for symptom in symptoms
    ]

    if (
        len(symptom_names)
        != len(set(symptom_names))
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "duplicate symptoms"
        )

    # --------------------------------------------------------
    # SYMPTOM STATUS + DURATION
    # --------------------------------------------------------

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
            and symptom["duration"]
            is not None
        ):
            raise ValueError(
                f"{case['case_id']}: "
                f"{symptom['name']} "
                "must not have duration"
            )

        duration = (
            symptom["duration"]
        )

        if duration is not None:

            if not isinstance(
                duration,
                dict,
            ):
                raise ValueError(
                    f"{case['case_id']}: "
                    "duration must be "
                    "dict or null"
                )

            value = duration.get(
                "value"
            )

            unit = duration.get(
                "unit"
            )

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"{case['case_id']}: "
                    "invalid duration value"
                )

            if unit not in {
                "hours",
                "days",
                "weeks",
            }:
                raise ValueError(
                    f"{case['case_id']}: "
                    "invalid duration unit"
                )

    # --------------------------------------------------------
    # MEDICATIONS
    # --------------------------------------------------------

    medications = (
        case["medications"]
    )

    if medications["status"] not in {
        "unknown",
        "none",
        "reported",
    }:
        raise ValueError(
            f"{case['case_id']}: "
            "invalid medication status"
        )

    if (
        medications["status"]
        == "reported"
        and not medications["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "reported medication "
            "requires an item"
        )

    if (
        medications["status"]
        != "reported"
        and medications["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "medication items must "
            "be empty"
        )

    # --------------------------------------------------------
    # ALLERGIES
    # --------------------------------------------------------

    allergies = (
        case["allergies"]
    )

    if allergies["status"] not in {
        "unknown",
        "none",
        "reported",
    }:
        raise ValueError(
            f"{case['case_id']}: "
            "invalid allergy status"
        )

    if (
        allergies["status"]
        == "reported"
        and not allergies["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "reported allergy "
            "requires an item"
        )

    if (
        allergies["status"]
        != "reported"
        and allergies["items"]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "allergy items must "
            "be empty"
        )

    # --------------------------------------------------------
    # MEDICATION / ALLERGY CONFLICT
    # --------------------------------------------------------

    medication_items = set(
        medications["items"]
    )

    allergy_items = set(
        allergies["items"]
    )

    conflicts = (
        medication_items
        & allergy_items
    )

    if conflicts:
        raise ValueError(
            f"{case['case_id']}: "
            f"medication/allergy conflict: "
            f"{sorted(conflicts)}"
        )

    # --------------------------------------------------------
    # MISSING INFORMATION
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
            "incorrect "
            "missing_information"
        )

    # --------------------------------------------------------
    # PATTERN-SPECIFIC CHECKS
    # --------------------------------------------------------

    pattern = case[
        "case_pattern"
    ]

    statuses = [
        symptom["status"]
        for symptom in symptoms
    ]

    if (
        pattern == "negation"
        and "absent" not in statuses
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "negation pattern requires "
            "an absent symptom"
        )

    if (
        pattern == "uncertainty"
        and "uncertain"
        not in statuses
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "uncertainty pattern requires "
            "an uncertain symptom"
        )

    if pattern == "mixed":

        required = {
            "present",
            "absent",
            "uncertain",
        }

        if not required.issubset(
            set(statuses)
        ):
            raise ValueError(
                f"{case['case_id']}: "
                "mixed pattern must contain "
                "present, absent and uncertain"
            )

    if (
        pattern == "missing_duration"
        and "duration"
        not in case[
            "missing_information"
        ]
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "missing_duration pattern "
            "requires missing duration"
        )

    if (
        pattern == "missing_medications"
        and medications["status"]
        != "unknown"
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "missing_medications pattern "
            "requires unknown medications"
        )

    if (
        pattern == "missing_allergies"
        and allergies["status"]
        != "unknown"
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "missing_allergies pattern "
            "requires unknown allergies"
        )

    if (
        pattern == "multiple_present"
        and statuses.count("present") < 2
    ):
        raise ValueError(
            f"{case['case_id']}: "
            "multiple_present pattern "
            "requires at least two "
            "present symptoms"
        )


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(
    cases,
):
    case_ids = [
        case["case_id"]
        for case in cases
    ]

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise ValueError(
            "Duplicate case IDs detected"
        )

    for case in cases:
        validate_case(
            case
        )


# ============================================================
# REPORT
# ============================================================

def print_distribution_report(
    cases,
):
    pattern_counts = Counter(
        case["case_pattern"]
        for case in cases
    )

    group_counts = Counter(
        case["symptom_group"]
        for case in cases
    )

    medication_counts = Counter(
        case["medications"]["status"]
        for case in cases
    )

    allergy_counts = Counter(
        case["allergies"]["status"]
        for case in cases
    )

    symptom_status_counts = Counter()

    missing_counts = Counter()

    for case in cases:

        for symptom in case[
            "symptoms"
        ]:
            symptom_status_counts[
                symptom["status"]
            ] += 1

        for missing in case[
            "missing_information"
        ]:
            missing_counts[
                missing
            ] += 1

    print(
        "\nCASE PATTERNS"
    )

    for pattern in sorted(
        pattern_counts
    ):
        print(
            f"  {pattern:<22} "
            f"{pattern_counts[pattern]}"
        )

    print(
        "\nSYMPTOM GROUPS"
    )

    for group in sorted(
        group_counts
    ):
        print(
            f"  {group:<22} "
            f"{group_counts[group]}"
        )

    print(
        "\nSYMPTOM STATUSES"
    )

    for status in [
        "present",
        "absent",
        "uncertain",
    ]:
        print(
            f"  {status:<22} "
            f"{symptom_status_counts[status]}"
        )

    print(
        "\nMEDICATION STATUS"
    )

    for status in [
        "unknown",
        "none",
        "reported",
    ]:
        print(
            f"  {status:<22} "
            f"{medication_counts[status]}"
        )

    print(
        "\nALLERGY STATUS"
    )

    for status in [
        "unknown",
        "none",
        "reported",
    ]:
        print(
            f"  {status:<22} "
            f"{allergy_counts[status]}"
        )

    print(
        "\nMISSING INFORMATION"
    )

    for field in [
        "duration",
        "medications",
        "allergies",
    ]:
        print(
            f"  {field:<22} "
            f"{missing_counts[field]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "NOOR HEALTH - "
        "SYNTHETIC CLINICAL CASES V2"
    )
    print("=" * 60)

    SYNTHETIC_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern_schedule = (
        build_pattern_schedule(
            NUM_CASES
        )
    )

    cases = []

    for case_number, case_pattern in enumerate(
        pattern_schedule,
        start=1,
    ):
        case = generate_case(
            case_number=case_number,
            case_pattern=case_pattern,
        )

        validate_case(
            case
        )

        cases.append(
            case
        )

    # Validate the entire collection once more.
    validate_dataset(
        cases
    )

    # IMPORTANT:
    # This overwrites clinical_cases_v2.jsonl.
    with OUTPUT_FILE.open(
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

    print(
        f"\nGenerated: "
        f"{len(cases)} cases"
    )

    print(
        f"Validation: PASSED"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print_distribution_report(
        cases
    )

    print(
        "\nEXAMPLES\n"
    )

    for case in cases[:3]:

        print(
            json.dumps(
                case,
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "-" * 60
        )


if __name__ == "__main__":
    main()