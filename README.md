# Noor Health

> An offline-first, multilingual AI assistant for structuring patient-reported information in resource-constrained healthcare settings.

Noor Health is a lightweight clinical intake assistant designed to help frontline and community health workers transform free-form patient statements into structured, clinician-reviewable information.

The project was originally created as part of the **Global AI Hackathon 2026**, inspired by the challenge of building small, practical AI systems for development and healthcare contexts.

Noor Health is designed around a simple principle:

**AI should help structure information — not replace clinical judgment.**

---

## The Problem

Clinical intake often begins with unstructured information.

A patient might say:

> "I've had stomach pain for two days and started vomiting about six hours ago. I don't have diarrhea, but I'm not sure if I have a fever. I'm not taking any medication. I'm allergic to penicillin."

For a healthcare worker, this information needs to be separated into clinically useful fields:

- What symptoms are present?
- What symptoms are explicitly absent?
- What is uncertain?
- How long has each symptom been present?
- Is the patient taking medication?
- Are allergies reported?
- What information is still missing?

This becomes more challenging in multilingual and resource-constrained environments, where connectivity and access to large cloud-based AI systems cannot always be assumed.

Noor Health explores whether a **small language model** can perform this structured extraction task across multiple languages.

---

## What Noor Health Does

Noor Health converts a patient statement into structured JSON.

### Example

**Patient statement**

```text
I've had stomach pain for two days and started vomiting about six hours ago.
I don't have diarrhea, but I'm not sure if I have a fever.
I'm not taking any medication.
I'm allergic to penicillin.
```

**Structured output**

```json
{
  "chief_complaint": "stomach pain",
  "symptoms": [
    {
      "name": "stomach pain",
      "status": "present",
      "duration": {
        "value": 2,
        "unit": "days"
      }
    },
    {
      "name": "vomiting",
      "status": "present",
      "duration": {
        "value": 6,
        "unit": "hours"
      }
    },
    {
      "name": "diarrhea",
      "status": "absent",
      "duration": null
    },
    {
      "name": "fever",
      "status": "uncertain",
      "duration": null
    }
  ],
  "medications": {
    "status": "none",
    "items": []
  },
  "allergies": {
    "status": "reported",
    "items": [
      "penicillin"
    ]
  },
  "missing_information": []
}
```

The distinction between **absent**, **uncertain**, and **not reported** information is intentionally preserved.

---

## Supported Languages

The current system is trained and evaluated on seven languages:

| Language | Code |
|---|---|
| English | `en` |
| German | `de` |
| Modern Standard Arabic | `ar_msa` |
| French | `fr` |
| Spanish | `es` |
| Hindi | `hi` |
| Swahili | `sw` |

The canonical structured output is represented in English to provide a consistent downstream schema.

---

## Model

Noor Health currently uses:

**Qwen3-0.6B**

The model was adapted using **LoRA (Low-Rank Adaptation)** for multilingual structured clinical information extraction.

### Training configuration

- Base model: Qwen3-0.6B
- Fine-tuning: LoRA
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Epochs: 3
- Learning rate: `2e-4`
- Effective batch size: 16
- Maximum sequence length: 512
- Precision: BF16
- Assistant-only loss masking
- Deterministic inference

The training set contained:

- **5,589 training samples**
- **699 validation samples**
- **700 held-out test samples**

Cases were split by `case_id` rather than individual language variants to prevent cross-language leakage between train and test sets.

---

## Dataset

The current experimental dataset contains synthetic multilingual clinical intake cases covering categories such as:

- respiratory symptoms
- gastrointestinal symptoms
- general symptoms
- pain
- urinary symptoms
- skin-related symptoms

Each underlying clinical case is represented across the seven supported languages.

The structured schema captures:

- chief complaint
- symptoms
- symptom status
- symptom-specific duration
- medications
- allergies
- missing information

### Data Quality Checks

Automated semantic consistency checks were applied to verify that multilingual utterances remained consistent with their structured targets.

After quality control:

- 7,000 multilingual records were generated
- 6,988 passed the final automated quality checks
- 12 samples were rejected
- all 700 held-out test samples passed the final dataset filtering

These checks measure **dataset consistency**, not clinical correctness or clinical validation.

---

## Evaluation

The base model and LoRA model were evaluated on the **same 700 held-out samples**, using:

- the same test set
- the same prompt
- the same decoding strategy
- the same base model
- the same evaluation logic

The only model-level difference was the LoRA adapter.

### Base vs. LoRA

| Metric | Qwen3-0.6B Base | Qwen3-0.6B + LoRA |
|---|---:|---:|
| Valid JSON | 97.86% | **99.29%** |
| Schema valid | 80.29% | **97.71%** |
| Core exact match | 1.71% | **77.57%** |
| Chief complaint | 14.57% | **97.57%** |
| Symptom extraction | 7.29% | **84.29%** |
| Symptom status | 7.29% | **82.57%** |
| Duration | 4.86% | **83.14%** |
| Medication | 39.43% | **97.14%** |
| Allergy | 51.57% | **94.71%** |

`Core exact match` requires all core structured extraction components for a sample to be correct.

### Sample-Level Comparison

For core exact match:

```text
Base correct:        12 / 700
LoRA correct:       543 / 700

LoRA improvements:  531
LoRA regressions:      0
Both wrong:          157
```

No sample that was core-exact correct with the base model became core-exact incorrect after LoRA fine-tuning.

---

## Performance by Language

Core exact match after LoRA fine-tuning:

| Language | Base | LoRA |
|---|---:|---:|
| German | 0% | **93%** |
| English | 11% | **89%** |
| Spanish | 0% | **89%** |
| French | 1% | **88%** |
| Arabic | 0% | **84%** |
| Swahili | 0% | **52%** |
| Hindi | 0% | **48%** |

The results reveal a substantial remaining multilingual performance gap.

Hindi and Swahili currently represent the largest sources of error and are therefore a major focus of the ongoing evaluation.

---

## Failure Analysis

The 157 samples that remained incorrect after LoRA fine-tuning were analyzed separately.

### Remaining failures by language

| Language | Failed samples |
|---|---:|
| Hindi | 52 |
| Swahili | 48 |
| Arabic | 16 |
| French | 12 |
| English | 11 |
| Spanish | 11 |
| German | 7 |

The most common remaining error was a **missing symptom**.

Additional failure patterns differed by language.

For example:

- Hindi showed comparatively frequent extra symptom generation and medication errors.
- Swahili showed frequent missing symptoms, symptom-status errors, and allergy errors.
- English, German, French, and Spanish failures were dominated by incomplete symptom extraction.

Cross-language analysis also showed:

- 29 / 100 test cases were correct in all seven languages.
- 22 cases failed only in Hindi.
- 13 cases failed only in Swahili.
- 10 cases failed in Hindi and Swahili while succeeding in the other languages.
- 2 cases failed across all seven languages.

This suggests that a substantial part of the remaining error is language-specific rather than purely case-specific.

---

## Deterministic Business Logic

One important finding was that some output fields should not necessarily be generated by the language model.

For example, `missing_information` can be derived deterministically from the extracted structure.

Raw model accuracy for this field after LoRA fine-tuning was only:

**2.14%**

When derived deterministically from the structured extraction:

**96.14%**

Noor Health therefore separates tasks that benefit from language understanding from tasks that can be handled reliably using deterministic application logic.

---

## Architecture

The current development architecture is:

```text
Patient statement
       │
       ▼
React Frontend
       │
       ▼
FastAPI Backend
       │
       ▼
Qwen3-0.6B + LoRA
       │
       ▼
Structured JSON
       │
       ▼
Schema Validation
       │
       ▼
Deterministic Post-processing
       │
       ▼
Clinician-reviewable Intake
```

The current development/demo inference environment uses **Modal with an NVIDIA L4 GPU**.

The long-term design goal is local or edge inference with a small, quantized model. Therefore, the current cloud-hosted development setup should not be interpreted as an already deployed offline system.

---

## Repository Structure

```text
noor-health/
├── backend/
│
├── frontend/
│
├── data/
│   └── processed/
│
├── ml/
│   ├── data/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── analysis/
│   └── archive/
│
├── results/
│   └── evaluation/
│       ├── lora/
│       ├── comparison/
│       └── failure_analysis/
│
└── README.md
```

The ML pipeline includes dataset generation, multilingual quality control, training, inference, evaluation, base-vs-LoRA comparison, and failure analysis.

---

## Design Principles

Noor Health follows several principles:

**Human in the loop**  
The generated structure is intended for review by a healthcare professional.

**Preserve uncertainty**  
"I don't have a fever" and "I'm not sure if I have a fever" must not become the same representation.

**Not reported ≠ none**  
Missing information must not be interpreted as a negative answer.

**Small models first**  
The project investigates how far a compact model can be pushed before requiring substantially larger infrastructure.

**Offline-first direction**  
The architecture is being developed with future local and edge deployment in mind.

**Deterministic where possible**  
Information that can be derived reliably in code should not unnecessarily depend on probabilistic model generation.

---

## Current Limitations

Noor Health is currently an experimental prototype.

Important limitations include:

- The current training and test data are synthetic.
- Evaluation measures structured extraction performance, not clinical outcomes.
- The system has not been clinically validated.
- Hindi and Swahili performance remains substantially below the other evaluated languages.
- Real-world multilingual evaluation is still required.
- Current development inference uses cloud GPU infrastructure.
- The system must not be used as an autonomous diagnostic or treatment system.

The reported results demonstrate performance on the project's held-out synthetic evaluation set and should not be interpreted as real-world clinical performance.

---

## Next Steps

Current development priorities include:

1. Detailed qualitative analysis of the remaining multilingual failures.
2. Independent evaluation on more realistic patient language.
3. Improving Hindi and Swahili extraction robustness.
4. Evaluating targeted multilingual data augmentation.
5. Comparing small-model sizes and fine-tuning strategies.
6. Model quantization.
7. Local / edge inference experiments.
8. Voice-based patient intake.
9. Improved clinician review workflows.

---

## Safety

Noor Health is **not a diagnostic system**.

It does not provide diagnoses, treatment recommendations, or autonomous clinical decisions.

Its purpose is to structure patient-reported information so that it can be reviewed by a qualified healthcare professional.

**Clinical review is required.**

---

## Hackathon Origin

Noor Health originated during the **Global AI Hackathon 2026** as an exploration of how small AI models could support healthcare workflows in settings where compute resources, connectivity, and access to large cloud models may be limited.

Development has continued beyond the initial hackathon scope, with a stronger focus on reproducible evaluation, multilingual failure analysis, model efficiency, and responsible clinical AI design.

---

## Status

🚧 **Research / prototype stage**

Noor Health is under active development and is not intended for clinical deployment.
