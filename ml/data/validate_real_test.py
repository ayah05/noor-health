"""
Validate the handwritten real-world test set.

Checks every record against the same vocabulary and schema rules
as the synthetic generator, so that the real test set and the
training data use identical label conventions.

missing_information is never typed by hand. It is derived
deterministically. With --fix the derived value is written back.

Usage (from the repository root):
    python ml/data/validate_real_test.py
    python ml/data/validate_real_test.py --fix
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR  # noqa: E402
from generate_synthetic import (  # noqa: E402
    ALLERGIES,
    MEDICATIONS,
    SYMPTOM_GROUPS,
    determine_missing_information,
)


REAL_TEST_FILE = DATA_DIR / "real_test" / "real_test_v1.jsonl"

LANGUAGES = {"en", "de", "ar_msa", "ar_eg"}
VARIETIES = {"standard", "colloquial", "msa", "egyptian"}
INPUT_MODES = {"typed", "asr_style"}

SYMPTOMS = {
    name
    for group in SYMPTOM_GROUPS.values()
    for key in ("chief_complaints", "related_symptoms")
    for name in group[key]
}

STATUSES = {"present", "absent", "uncertain"}
UNITS = {"hours", "days", "weeks"}
INFO_STATUSES = {"unknown", "none", "reported"}

REQUIRED_FIELDS = [
    "case_id", "split", "language", "variety",
    "input_mode", "author", "utterance", "target",
]


def check_info_block(block, allowed_items, label):
    errors = []
    if block.get("status") not in INFO_STATUSES:
        errors.append(f"{label}: invalid status {block.get('status')!r}")
    items = block.get("items", [])
    if block.get("status") == "reported" and not items:
        errors.append(f"{label}: 'reported' requires at least one item")
    if block.get("status") != "reported" and items:
        errors.append(f"{label}: items must be empty unless 'reported'")
    for item in items:
        if item not in allowed_items:
            errors.append(
                f"{label}: {item!r} not in vocabulary "
                f"(use the active ingredient, e.g. Panadol -> paracetamol)"
            )
    return errors


def validate_record(record):
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field {field!r}")
    if errors:
        return errors

    if record["language"] not in LANGUAGES:
        errors.append(f"unknown language {record['language']!r}")
    if record["variety"] not in VARIETIES:
        errors.append(f"unknown variety {record['variety']!r}")
    if record["input_mode"] not in INPUT_MODES:
        errors.append(f"unknown input_mode {record['input_mode']!r}")
    if not record["utterance"].strip():
        errors.append("empty utterance")

    target = record["target"]
    symptoms = target.get("symptoms", [])

    if not symptoms:
        errors.append("no symptoms")
        return errors

    names = [s.get("name") for s in symptoms]
    if len(names) != len(set(names)):
        errors.append("duplicate symptoms")

    for symptom in symptoms:
        name = symptom.get("name")
        if name not in SYMPTOMS:
            errors.append(f"symptom {name!r} not in vocabulary")
        if symptom.get("status") not in STATUSES:
            errors.append(f"{name}: invalid status {symptom.get('status')!r}")
        duration = symptom.get("duration")
        if duration is not None:
            if symptom.get("status") != "present":
                errors.append(f"{name}: only present symptoms may have a duration")
            value = duration.get("value")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{name}: duration value must be a positive integer")
            if duration.get("unit") not in UNITS:
                errors.append(f"{name}: invalid duration unit {duration.get('unit')!r}")

    chief = target.get("chief_complaint")
    if chief != names[0]:
        errors.append("chief_complaint must be the first symptom (training convention)")
    if symptoms[0].get("status") != "present":
        errors.append("chief_complaint must have status 'present'")

    errors += check_info_block(target.get("medications", {}), MEDICATIONS, "medications")
    errors += check_info_block(target.get("allergies", {}), ALLERGIES, "allergies")

    return errors


def derive_missing(target):
    return determine_missing_information(
        symptoms=target["symptoms"],
        chief_complaint=target["chief_complaint"],
        medications=target["medications"],
        allergies=target["allergies"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=REAL_TEST_FILE)
    parser.add_argument("--fix", action="store_true",
                        help="write derived missing_information back to the file")
    args = parser.parse_args()

    records = []
    with args.file.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                records.append((line_number, json.loads(line)))

    ids = Counter(r["case_id"] for _, r in records)
    duplicate_ids = [i for i, n in ids.items() if n > 1]

    n_errors = 0
    n_fixed = 0
    for line_number, record in records:
        errors = validate_record(record)
        if not errors:
            expected = derive_missing(record["target"])
            if record["target"].get("missing_information") != expected:
                if args.fix:
                    record["target"]["missing_information"] = expected
                    n_fixed += 1
                else:
                    errors.append(
                        f"missing_information should be {expected} "
                        f"(run with --fix)"
                    )
        for error in errors:
            print(f"line {line_number} [{record.get('case_id')}]: {error}")
        n_errors += len(errors)

    for case_id in duplicate_ids:
        print(f"duplicate case_id: {case_id}")
        n_errors += 1

    if args.fix and n_fixed:
        with args.file.open("w", encoding="utf-8") as f:
            for _, record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"fixed missing_information in {n_fixed} record(s)")

    print(f"\n{len(records)} records, {n_errors} error(s)")
    print("by language:  ", dict(Counter(r["language"] for _, r in records)))
    print("by variety:   ", dict(Counter(r["variety"] for _, r in records)))
    print("by input_mode:", dict(Counter(r["input_mode"] for _, r in records)))

    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
