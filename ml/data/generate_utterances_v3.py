"""
Noor Health - utterance generation v3.

What changed compared to v2 (generate_utterances.py):

1. Rendering plan per (case, language). Before calling the LLM we decide
   deterministically HOW the case is said: mention order, wording from the
   lexicon, relative or vague time expressions, brand names, register,
   length, speaker gender, typed vs speech-like. The LLM only writes the
   sentence; it no longer decides the variation itself.
2. Coordinated durations ("cough and fever for 3 days") are allowed when
   both symptoms really share the duration.
3. Deterministic validation with the lexicon instead of GPT-checks-GPT.
   Failed samples are regenerated (max 3 attempts), then logged as rejected.
4. Deterministic augmentation after validation: speech-like transcripts
   (lowercase, no punctuation) and occasional typos.

Languages: en, de, ar_msa, ar_eg.

Usage (from the repository root):
    python ml/data/generate_utterances_v3.py --dry-run --limit 3
    python ml/data/generate_utterances_v3.py --limit 20
    python ml/data/generate_utterances_v3.py
"""

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR  # noqa: E402
from generate_synthetic import determine_missing_information  # noqa: E402
from lexicon import check_sample, contains_term, load_lexicon, normalize  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

MODEL = "gpt-5-mini"
MAX_WORKERS = 10
MAX_API_RETRIES = 5
MAX_ATTEMPTS = 3          # regenerations when validation fails
SEED = "noor-v3"

LANGUAGES = {
    "en": "English",
    "de": "German",
    "ar_msa": "Modern Standard Arabic",
    "ar_eg": "Egyptian Arabic",
}

SPLIT_FILES = {
    "train": PROCESSED_DATA_DIR / "train_cases.jsonl",
    "validation": PROCESSED_DATA_DIR / "validation_cases.jsonl",
    "test": PROCESSED_DATA_DIR / "test_cases.jsonl",
}

OUTPUT_FILE = SYNTHETIC_DATA_DIR / "multilingual_cases_v3.jsonl"
REJECTED_FILE = SYNTHETIC_DATA_DIR / "multilingual_cases_v3_rejected.jsonl"

# Probabilities of the rendering plan. Change here, not in the code.
P = {
    "coordinate_duration": 0.25,   # per case, if two present symptoms exist
    "chief_not_first": 0.40,
    "relative_time": 0.45,         # if the duration has a relative expression
    "vague_time": 0.35,            # present symptom without duration
    "brand_name": 0.40,
    "asr_style": 0.35,
    "typo": 0.15,                  # typed Latin-script utterances only
    "code_switch": 0.15,           # de, ar_eg: one English word
}

REGISTERS = (["colloquial", "neutral", "slightly formal"], [0.50, 0.35, 0.15])
# MSA is not spoken colloquially, so no "colloquial" register and no
# code-switching there.
REGISTERS_BY_LANGUAGE = {
    "ar_msa": (["neutral", "slightly formal"], [0.70, 0.30]),
}
CODE_SWITCH_LANGUAGES = {"de", "ar_eg"}

LENGTHS = (["short and direct", "normal", "rambling with filler words"], [0.35, 0.45, 0.20])

LANGUAGE_NOTES = {
    "en": "Write natural everyday English.",
    "de": "Schreibe natürliches Alltagsdeutsch, wie man mit einer Pflegekraft spricht.",
    "ar_msa": (
        "Write simple Modern Standard Arabic in Arabic script. No dialect. "
        "Use Arabic-Indic digits if you write digits. For a value of exactly 2 "
        "use the dual form without a digit (يومين، ساعتين، أسبوعين)."
    ),
    "ar_eg": (
        "Write Egyptian Arabic (عامية مصرية) in Arabic script, the way a patient in "
        "Egypt really talks to a nurse. Do NOT use Modern Standard Arabic. "
        "Durations may be written as words (يومين، تلت أيام، أسبوع) or digits."
    ),
}

write_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def rng_for(*parts: str) -> random.Random:
    """Deterministic RNG: same inputs give the same plan on every run."""
    digest = hashlib.sha256("|".join((SEED,) + parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_cases() -> list[dict]:
    cases = []
    for split, path in SPLIT_FILES.items():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    case["_split"] = split
                    cases.append(case)
    return cases


def build_target(case: dict) -> dict:
    return {
        "chief_complaint": case["chief_complaint"],
        "symptoms": case["symptoms"],
        "medications": case["medications"],
        "allergies": case["allergies"],
        "missing_information": case["missing_information"],
    }


# ============================================================
# CASE-LEVEL CHANGE: COORDINATED DURATIONS
# ============================================================

def apply_coordination(case: dict) -> tuple[dict, list[str] | None]:
    """
    For some cases, give a second present symptom the SAME duration as the
    chief complaint, so the utterance may say "X and Y for 3 days".

    Decided per case (not per language) so that every language shares the
    same target.
    """
    case = copy.deepcopy(case)
    rng = rng_for(case["case_id"], "coordination")
    symptoms = case["symptoms"]
    chief = symptoms[0]

    candidates = [s for s in symptoms[1:] if s["status"] == "present"]
    if chief["duration"] is None or not candidates or rng.random() >= P["coordinate_duration"]:
        return case, None

    partner = rng.choice(candidates)
    partner["duration"] = dict(chief["duration"])
    case["missing_information"] = determine_missing_information(
        symptoms=symptoms,
        chief_complaint=case["chief_complaint"],
        medications=case["medications"],
        allergies=case["allergies"],
    )
    return case, [chief["name"], partner["name"]]


# ============================================================
# RENDERING PLAN
# ============================================================

def time_rendering(symptom: dict, language: str, lex: dict, rng: random.Random) -> dict:
    duration = symptom["duration"]

    if symptom["status"] != "present":
        return {"mode": "none"}

    if duration is None:
        vague = [t for t in lex["time_expressions"][language] if t["kind"] == "vague"]
        if vague and rng.random() < P["vague_time"]:
            return {"mode": "vague", "text": rng.choice(vague)["text"]}
        return {"mode": "none"}

    relative = [
        t for t in lex["time_expressions"][language]
        if t["kind"] in {"relative", "approximate"}
        and t["value"] == duration["value"] and t["unit"] == duration["unit"]
    ]
    if relative and rng.random() < P["relative_time"]:
        return {"mode": "relative", "text": rng.choice(relative)["text"]}

    return {"mode": "exact", "value": duration["value"], "unit": duration["unit"]}


def build_plan(case: dict, coordinated: list[str] | None, language: str, lex: dict) -> dict:
    rng = rng_for(case["case_id"], language, "plan")
    symptoms = case["symptoms"]
    chief = case["chief_complaint"]

    # Mention order
    order = [s["name"] for s in symptoms]
    chief_cue = None
    if len(order) > 1 and rng.random() < P["chief_not_first"]:
        rest = order[1:]
        rng.shuffle(rest)
        position = rng.randint(1, len(rest))
        order = rest[:position] + [chief] + rest[position:]
        chief_cue = rng.choice(lex["chief_complaint_cues"][language])

    by_name = {s["name"]: s for s in symptoms}
    planned = []
    for name in order:
        s = by_name[name]
        phrases = lex["symptoms"][name][language]["phrases"]
        # Varied wording only for present symptoms. Absent / uncertain symptoms
        # get the first (base) phrase, because sentence-like phrases such as
        # "I can't go to the toilet" are awkward or ambiguous when negated.
        wording = rng.choice(phrases) if s["status"] == "present" else phrases[0]
        entry = {
            "name": name,
            "status": s["status"],
            "wording": wording,
            "time": time_rendering(s, language, lex, rng),
        }
        if s["status"] == "uncertain":
            entry["uncertainty_cue"] = rng.choice(lex["uncertainty_cues"][language])
        planned.append(entry)

    # Coordinated pair shares ONE time expression
    if coordinated:
        shared = next(p["time"] for p in planned if p["name"] == coordinated[0])
        for p in planned:
            if p["name"] in coordinated:
                p["time"] = shared

    def drug_surface(drug: str) -> str:
        names = lex["drug_names"][drug][language]
        if len(names) > 1 and rng.random() < P["brand_name"]:
            return rng.choice(names[1:])
        return names[0]

    meds = case["medications"]
    allergies = case["allergies"]

    code_switch = None
    if language in CODE_SWITCH_LANGUAGES and rng.random() < P["code_switch"]:
        code_switch = rng.choice(planned)["name"]

    return {
        "order": order,
        "chief_complaint": chief,
        "chief_cue": chief_cue,
        "symptoms": planned,
        "coordinated": coordinated,
        "medications": {
            "status": meds["status"],
            "surface": [drug_surface(d) for d in meds["items"]],
        },
        "allergies": {
            "status": allergies["status"],
            "surface": [drug_surface(d) for d in allergies["items"]],
        },
        "register": rng.choices(*REGISTERS_BY_LANGUAGE.get(language, REGISTERS))[0],
        "length": rng.choices(*LENGTHS)[0],
        "speaker_gender": rng.choice(["female", "male"]),
        "input_mode": "asr_style" if rng.random() < P["asr_style"] else "typed",
        "code_switch_symptom": code_switch,
    }


# ============================================================
# PROMPT
# ============================================================

STATUS_TEXT = {
    "present": "the patient says they have this",
    "absent": "the patient says they do NOT have this",
    "uncertain": "the patient says they are not sure whether they have this",
}

SINGULAR = {"hours": "hour", "days": "day", "weeks": "week"}


def describe_time(t: dict) -> str:
    if t["mode"] == "exact":
        unit = SINGULAR[t["unit"]] if t["value"] == 1 else t["unit"]
        return f'duration: {t["value"]} {unit} (say it naturally and grammatically in the target language)'
    if t["mode"] == "relative":
        return f'duration: say "{t["text"]}" (or a very close equivalent)'
    if t["mode"] == "vague":
        return f'time: say "{t["text"]}" and NO number'
    return "no time information at all"


def build_prompt(plan: dict, language: str) -> str:
    lines = []
    for i, s in enumerate(plan["symptoms"], start=1):
        line = (
            f'{i}. [meaning: {s["name"]}] {STATUS_TEXT[s["status"]]}. '
            f'Suggested wording: "{s["wording"]}". '
        )
        line += describe_time(s["time"]) + "."
        if s.get("uncertainty_cue"):
            line += f' Express the doubt e.g. with "{s["uncertainty_cue"]}".'
        if plan["code_switch_symptom"] == s["name"]:
            line += (
                " Code-switching: for THIS symptom use the English term INSTEAD of the "
                "native wording, inside the sentence (do not append it at the end)."
            )
        lines.append(line)

    if plan["coordinated"]:
        a, b = plan["coordinated"]
        lines.append(
            f'Mention "{a}" and "{b}" TOGETHER with one shared time expression, '
            f'e.g. "{a} and {b} for ..." in the target language.'
        )
    else:
        lines.append(
            "Never coordinate symptoms under one time expression: each time "
            "expression must clearly belong to exactly one symptom."
        )

    def info_line(label: str, block: dict, unknown_rule: str, none_rule: str) -> str:
        if block["status"] == "unknown":
            return f"{label}: {unknown_rule}"
        if block["status"] == "none":
            return f"{label}: {none_rule}"
        return f'{label}: mention exactly these names as written: {", ".join(block["surface"])}.'

    meds = info_line(
        "Medications", plan["medications"],
        "do NOT mention medications at all.",
        "the patient says they take no medication (vary the wording).",
    )
    allergies = info_line(
        "Allergies", plan["allergies"],
        "do NOT mention allergies at all.",
        "the patient says they have no allergies (vary the wording, not always 'no known allergies').",
    )

    if plan["chief_cue"]:
        chief_rule = (
            f'The MAIN complaint is "{plan["chief_complaint"]}". It is NOT the first thing '
            f'mentioned. Mark it as the main reason for the visit, e.g. with "{plan["chief_cue"]}".'
        )
    else:
        chief_rule = (
            f'The MAIN complaint is "{plan["chief_complaint"]}"; mention it first.'
        )

    if plan["input_mode"] == "asr_style":
        mode_rule = (
            "Write it as a transcript of spoken speech: no punctuation, numbers as words, "
            "small hesitations or filler words are fine."
        )
    else:
        mode_rule = "Write it as a typed chat message."

    return f"""You write ONE synthetic patient statement for training an offline clinical intake model.

Language: {LANGUAGES[language]}
{LANGUAGE_NOTES[language]}

Speaker: a {plan["speaker_gender"]} adult patient talking to a health worker.
Use grammatically correct {plan["speaker_gender"]} forms where the language requires it.
Register: {plan["register"]}. Length: {plan["length"]}.
{mode_rule}

Mention the symptoms in EXACTLY this order:
{chr(10).join(lines)}

{chief_rule}
{meds}
{allergies}

Hard rules:
- Write ONLY what the patient says. Never copy the instructions, the [meaning: ...] labels,
  or words like "has", "present", "absent", "main complaint" from this prompt.
- Mention every symptom above, no other symptoms, no diagnoses, no advice.
- Keep each status exactly as described above.
- Do not add severity words unless natural filler ("a bit", "really"); never add new facts.
- Use only the durations and time expressions given. No other numbers.
- The suggested wordings are suggestions: you may inflect them, but keep the meaning.

Return ONLY the patient statement. No quotes, no explanation."""


# ============================================================
# VALIDATION
# ============================================================

DIGITS = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]")


ARABIC = {"ar_msa", "ar_eg"}

# A reported drug must not be preceded by a block-specific negation
# ("not allergic to X", "I don't take X") or followed by doubt ("X, not sure").
# The phrases are specific on purpose: generic words like "no" would also
# match the neighbouring clause ("I take X. I have no allergies.").
NEGATION_PHRASES = {
    "allergies": {
        "en": ["not allergic", "no allergy", "no allergies", "not allergic to"],
        "de": ["nicht allergisch", "keine allergie", "keine allergien"],
        "ar_msa": ["لا اعاني من حساسيه", "ليس لدي حساسيه", "لا توجد لدي حساسيه", "ليست لدي حساسيه"],
        "ar_eg": ["معنديش حساسيه", "ما عنديش حساسيه", "ماعنديش حساسيه", "مفيش حساسيه", "مليش حساسيه"],
    },
    "medications": {
        "en": ["don't take", "dont take", "not taking", "do not take", "not on"],
        "de": ["nehme kein", "nehme nicht", "nehm kein", "nehm nicht"],
        "ar_msa": ["لا اتناول", "لم اتناول", "لا اخذ"],
        "ar_eg": ["مش باخد", "ما باخدش", "مباخدش", "مش باخده"],
    },
}
DOUBT_PHRASES = {
    "en": ["not sure", "unsure", "don't know"],
    "de": ["nicht sicher", "weiß nicht", "weiss nicht"],
    "ar_msa": ["لست متاكد", "لا اعرف"],
    "ar_eg": ["مش متاكد", "مش عارف"],
}


def polarity_problems(text: str, plan: dict, language: str, before: int = 5, after: int = 4) -> list[str]:
    tokens = re.findall(r"[\w']+", normalize(text))
    problems = []
    for block in ("medications", "allergies"):
        if plan[block]["status"] != "reported":
            continue
        negations = [normalize(p) for p in NEGATION_PHRASES[block].get(language, [])]
        doubts = [normalize(p) for p in DOUBT_PHRASES.get(language, [])]
        for surface in plan[block]["surface"]:
            first = normalize(surface).split()[0]
            for i, tok in enumerate(tokens):
                if first not in tok:
                    continue
                left = " ".join(tokens[max(0, i - before): i])
                right = " ".join(tokens[i + 1: i + 1 + after])
                hits = [p for p in negations if p in left] + [p for p in doubts if p in right]
                if hits:
                    problems.append(f"possible negated {block[:-1]} '{surface}' (near: {hits})")
                break
    return problems


def validate(sample: dict, plan: dict, lex: dict) -> list[str]:
    problems = check_sample(sample, lex)
    text = sample["utterance"]
    language = sample["language"]

    # A code-switched symptom may be said in English instead.
    cs = plan.get("code_switch_symptom")
    if cs:
        english = lex["symptoms"][cs]["en"]["match"]
        if any(p in normalize(text) for p in english):
            problems = [p for p in problems if not p.startswith(f"symptom not found: {cs} ")]

    exact_planned = any(s["time"]["mode"] == "exact" for s in plan["symptoms"])
    if not exact_planned and DIGITS.search(text):
        problems.append("digits found although no exact duration was planned")

    for block in ("medications", "allergies"):
        if plan[block]["status"] == "unknown":
            # Unknown medications AND allergies: no drug name may appear at all.
            other = "allergies" if block == "medications" else "medications"
            if plan[other]["status"] == "unknown":
                for names in lex["drug_names"].values():
                    for name in (n for per_lang in names.values() for n in per_lang):
                        if contains_term(text, name):
                            problems.append(f"drug name '{name}' mentioned although {block} unknown")
                            break

    # Prompt artifacts copied into the utterance
    if re.search(r"(—|-|:)\s*has\b|\bhas\s*[.)]|\(has|\bhas\s*$|\[meaning|main complaint:", text, re.IGNORECASE):
        problems.append("prompt artifact in utterance")
    if language != "en" and re.search(r"\bhas\b", text, re.IGNORECASE):
        problems.append("prompt artifact in utterance")

    problems += polarity_problems(text, plan, language)

    # "1 hours", "1 days", "1 weeks"
    if re.search(r"(?<![0-9])1\s+(hours|days|weeks)\b", text):
        problems.append("ungrammatical singular unit")

    # Arabic: Latin script only for the planned code-switched symptom or Latin drug names
    if language in ARABIC:
        allowed = set()
        if plan.get("code_switch_symptom"):
            english = lex["symptoms"][plan["code_switch_symptom"]]["en"]
            for term in english["match"] + english["phrases"]:
                allowed.update(normalize(term).replace("'", " ").split())
        for block in ("medications", "allergies"):
            for surface in plan[block]["surface"]:
                allowed.update(normalize(surface).split())
        latin_words = re.findall(r"[a-z]+", normalize(text))
        stray = [w for w in latin_words if not any(w.startswith(a) or a.startswith(w) for a in allowed)]
        if stray:
            problems.append(f"unexpected Latin words in Arabic text: {stray[:3]}")

    if len(text) < 4:
        problems.append("utterance too short")
    if text.strip().startswith("{"):
        problems.append("looks like JSON")

    return sorted(set(problems))


# ============================================================
# AUGMENTATION (after validation)
# ============================================================

PUNCTUATION = re.compile(r"[.,;:!?¿¡\"()\[\]…،؛؟«»]")
LATIN = {"en", "de"}


def add_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randint(1, len(w) - 2)
    kind = rng.choice(["drop", "swap", "double"])
    if kind == "drop":
        w = w[:j] + w[j + 1:]
    elif kind == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def augment(text: str, plan: dict, case_id: str, language: str) -> str:
    rng = rng_for(case_id, language, "augment")
    if plan["input_mode"] == "asr_style":
        text = PUNCTUATION.sub(" ", text)
        if language in LATIN:
            text = text.lower()
        return re.sub(r"\s+", " ", text).strip()
    if language in LATIN and rng.random() < P["typo"]:
        return add_typo(text, rng)
    return text


# ============================================================
# LLM CALL
# ============================================================

_client = None


def get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_llm(prompt: str) -> str:
    last_error = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = get_client().responses.create(model=MODEL, input=prompt)
            text = response.output_text.strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
                text = text[1:-1].strip()
            return text
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < MAX_API_RETRIES:
                time.sleep(2 * 2 ** (attempt - 1))
    raise last_error


# ============================================================
# WORKER
# ============================================================

def process(case: dict, coordinated, language: str, lex: dict, generate=call_llm) -> dict:
    plan = build_plan(case, coordinated, language, lex)
    prompt = build_prompt(plan, language)
    target = build_target(case)

    attempts = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        utterance = generate(prompt).strip()
        sample = {"language": language, "utterance": utterance, "target": target}
        problems = validate(sample, plan, lex)
        if not problems:
            return {
                "ok": True,
                "record": {
                    "case_id": case["case_id"],
                    "split": case["_split"],
                    "language": language,
                    "utterance": augment(utterance, plan, case["case_id"], language),
                    "utterance_clean": utterance,
                    "target": target,
                    "plan": plan,
                    "attempts": attempt,
                    "source": "synthetic_v3",
                    "generator_model": MODEL,
                    "schema_version": "v2",
                },
            }
        attempts.append({"utterance": utterance, "problems": problems})

    return {
        "ok": False,
        "record": {
            "case_id": case["case_id"],
            "split": case["_split"],
            "language": language,
            "target": target,
            "plan": plan,
            "failed_attempts": attempts,
        },
    }


def append_jsonl(path: Path, record: dict):
    with write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["case_id"], r["language"]))
    return keys


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print plans and prompts, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases")
    parser.add_argument("--languages", nargs="*", default=list(LANGUAGES))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    lex = load_lexicon()
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    prepared = [apply_coordination(c) for c in cases]
    jobs = [(c, coord, lang) for c, coord in prepared for lang in args.languages]

    if args.dry_run:
        for c, coord, lang in jobs:
            plan = build_plan(c, coord, lang, lex)
            print("=" * 70)
            print(f"{c['case_id']} [{lang}] split={c['_split']} coordinated={coord}")
            print(json.dumps(build_target(c), ensure_ascii=False))
            print("-" * 70)
            print(build_prompt(plan, lang))
        print(f"\n{len(jobs)} prompts (dry run, nothing generated)")
        return

    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    done = existing_keys(OUTPUT_FILE) | existing_keys(REJECTED_FILE)
    todo = [j for j in jobs if (j[0]["case_id"], j[2]) not in done]
    print(f"{len(jobs)} samples total, {len(jobs) - len(todo)} already done, {len(todo)} to generate")

    stats = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, c, coord, lang, lex): (c["case_id"], lang) for c, coord, lang in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                stats["api_error"] += 1
                print(f"  API error for {key}: {error}")
                continue
            if result["ok"]:
                append_jsonl(OUTPUT_FILE, result["record"])
                stats["ok"] += 1
                stats[f"attempts_{result['record']['attempts']}"] += 1
            else:
                append_jsonl(REJECTED_FILE, result["record"])
                stats["rejected"] += 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={stats['ok']} rejected={stats['rejected']} errors={stats['api_error']}")

    print("\nDone.", dict(stats))
    print(f"Output:   {OUTPUT_FILE}")
    print(f"Rejected: {REJECTED_FILE}")


if __name__ == "__main__":
    main()