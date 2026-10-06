"""
Noor Health lexicon: loading, validation and deterministic checks.

The lexicon (data/lexicon/lexicon_v1.json) has two jobs:

1. Generation: "phrases" give the utterance generator varied,
   colloquial wording per symptom and language.
2. Checking: "match" patterns let us verify WITHOUT an LLM that
   every symptom in a target is actually mentioned in the
   utterance. This replaces the circular GPT-checks-GPT step.

Usage (from the repository root):
    python ml/data/lexicon.py --validate
    python ml/data/lexicon.py --check data/processed/train_sft.jsonl
    python ml/data/lexicon.py --check data/processed/train_sft.jsonl --languages en de --show 10
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR  # noqa: E402
from generate_synthetic import ALLERGIES, MEDICATIONS, SYMPTOM_GROUPS  # noqa: E402


LEXICON_FILE = DATA_DIR / "lexicon" / "lexicon_v1.json"

VOCAB_SYMPTOMS = {
    name
    for group in SYMPTOM_GROUPS.values()
    for key in ("chief_complaints", "related_symptoms")
    for name in group[key]
}

# Generic symptoms whose patterns overlap with specific ones.
WEAK_SYMPTOMS = {"pain"}


# ============================================================
# NORMALIZATION
# ============================================================

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")
_ARABIC_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه",
    "ى": "ي",
})


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes and Arabic letter variants."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("\u2019", "'").replace("`", "'")
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_ARABIC_MAP)
    return text


# ============================================================
# LOAD + VALIDATE
# ============================================================

def load_lexicon(path: Path = LEXICON_FILE) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_lexicon(lex: dict) -> list[str]:
    errors = []
    languages = lex["languages"]

    missing_symptoms = VOCAB_SYMPTOMS - set(lex["symptoms"])
    extra_symptoms = set(lex["symptoms"]) - VOCAB_SYMPTOMS
    for name in sorted(missing_symptoms):
        errors.append(f"symptom {name!r} is in the vocabulary but not in the lexicon")
    for name in sorted(extra_symptoms):
        errors.append(f"symptom {name!r} is in the lexicon but not in the vocabulary")

    for name, per_lang in lex["symptoms"].items():
        for lang in languages:
            entry = per_lang.get(lang)
            if not entry:
                errors.append(f"{name}: missing language {lang}")
                continue
            if not entry.get("phrases"):
                errors.append(f"{name}/{lang}: no phrases")
            if not entry.get("match"):
                errors.append(f"{name}/{lang}: no match patterns")
            for pattern in entry.get("match", []):
                if pattern != normalize(pattern):
                    errors.append(
                        f"{name}/{lang}: match pattern {pattern!r} is not normalized "
                        f"(expected {normalize(pattern)!r})"
                    )

    # Self-consistency: every generation phrase must be found by its own
    # match patterns, otherwise generated sentences fail the check.
    for name, per_lang in lex["symptoms"].items():
        for lang in languages:
            entry = per_lang.get(lang) or {}
            for phrase in entry.get("phrases", []):
                if not any(p in normalize(phrase) for p in entry.get("match", [])):
                    errors.append(f"{name}/{lang}: phrase {phrase!r} is not matched by its own match patterns")

    for drug in set(MEDICATIONS) | set(ALLERGIES):
        if drug not in lex["drug_names"]:
            errors.append(f"drug {drug!r} missing from drug_names")
        else:
            for lang in languages:
                if not lex["drug_names"][drug].get(lang):
                    errors.append(f"drug {drug}: missing language {lang}")

    for lang in languages:
        for item in lex["time_expressions"].get(lang, []):
            kind = item.get("kind")
            if kind not in {"relative", "approximate", "vague"}:
                errors.append(f"time/{lang}: invalid kind {kind!r} for {item.get('text')!r}")
            if kind == "vague" and (item["value"] is not None or item["unit"] is not None):
                errors.append(f"time/{lang}: vague expression {item['text']!r} must have null value and unit")
            if kind != "vague" and (not isinstance(item["value"], int) or item["unit"] not in {"hours", "days", "weeks"}):
                errors.append(f"time/{lang}: {item['text']!r} needs an integer value and a valid unit")
        if not lex["time_expressions"].get(lang):
            errors.append(f"time_expressions: missing language {lang}")
        if not lex["chief_complaint_cues"].get(lang):
            errors.append(f"chief_complaint_cues: missing language {lang}")

    return errors


# ============================================================
# DETERMINISTIC CHECKS
# ============================================================

_LATIN = re.compile(r"[a-z]")


def contains_term(text: str, term: str) -> bool:
    """
    Substring match on normalized text. Latin-script terms must match as
    whole words, so that e.g. 'ASS' (aspirin) is not found in 'Wasserlassen'.
    Arabic terms match as substrings, because of attached prefixes (ال، ب، و).
    """
    text, term = normalize(text), normalize(term)
    if _LATIN.search(term):
        return re.search(rf"(?<![a-zäöüß]){re.escape(term)}(?![a-zäöüß])", text) is not None
    return term in text


def symptom_mentioned(utterance: str, symptom: str, language: str, lex: dict) -> bool:
    text = normalize(utterance)
    patterns = lex["symptoms"][symptom][language]["match"]
    return any(p in text for p in patterns)


def drug_mentioned(utterance: str, drug: str, language: str, lex: dict) -> bool:
    # Accept the drug name from any language: code-switching is common.
    names = {
        n
        for names_per_lang in lex["drug_names"][drug].values()
        for n in names_per_lang
    }
    return any(contains_term(utterance, n) for n in names)


def check_sample(sample: dict, lex: dict) -> list[str]:
    """
    Return a list of problems. An empty list means every symptom and
    drug in the target is detectably mentioned in the utterance.

    This checks presence only, not status (present/absent/uncertain).
    """
    language = sample["language"]
    if language not in lex["languages"]:
        return []

    utterance = sample["utterance"]
    target = sample["target"]
    problems = []

    for symptom in target["symptoms"]:
        name = symptom["name"]
        if name in WEAK_SYMPTOMS:
            continue
        if not symptom_mentioned(utterance, name, language, lex):
            problems.append(f"symptom not found: {name} ({symptom['status']})")

    for block in ("medications", "allergies"):
        for drug in target[block]["items"]:
            if not drug_mentioned(utterance, drug, language, lex):
                problems.append(f"{block[:-1]} not found: {drug}")

    return problems


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check", type=Path, help="JSONL file with language/utterance/target")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--show", type=int, default=5, help="examples to show per language")
    args = parser.parse_args()

    lex = load_lexicon()

    if args.validate or not args.check:
        errors = validate_lexicon(lex)
        for error in errors:
            print("ERROR:", error)
        print(f"lexicon {lex['version']}: {len(errors)} error(s)")
        if errors:
            sys.exit(1)

    if args.check:
        languages = set(args.languages or lex["languages"])
        totals = Counter()
        failures = defaultdict(list)
        problem_counts = defaultdict(Counter)

        with args.check.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                lang = sample["language"]
                if lang not in languages:
                    continue
                totals[lang] += 1
                problems = check_sample(sample, lex)
                if problems:
                    failures[lang].append((sample, problems))
                    for p in problems:
                        problem_counts[lang][p] += 1

        print(f"\nCHECK {args.check}")
        for lang in sorted(totals):
            n, bad = totals[lang], len(failures[lang])
            print(f"\n[{lang}] {n - bad}/{n} pass ({(n - bad) / n:.1%}), {bad} flagged")
            for problem, count in problem_counts[lang].most_common(8):
                print(f"   {count:>4}  {problem}")
            for sample, problems in failures[lang][: args.show]:
                print(f"   - {sample['utterance']}")
                print(f"     -> {'; '.join(problems)}")


if __name__ == "__main__":
    main()