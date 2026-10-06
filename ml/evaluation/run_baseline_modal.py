import json
import time
from pathlib import Path

import modal


# ============================================================
# MODAL CONFIG
# ============================================================

app = modal.App(
    "noor-health-final-baseline-evaluation"
)

image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "safetensors",
    )
)

model_volume = modal.Volume.from_name(
    "noor-health-models",
    create_if_missing=True,
)

# Important:
# This is a path INSIDE the Modal container / volume.
# It is not a local Windows project path.
MODEL_DIR = "/models/qwen3-0.6b"

BATCH_SIZE = 32


# ============================================================
# SYSTEM PROMPT
# ============================================================

# Keep this identical to the LoRA evaluation prompt so that
# the only meaningful model difference is the LoRA adapter.

SYSTEM_PROMPT = """
You are Noor Health, a multilingual clinical intake assistant.

Extract structured clinical information from the patient's statement.

You are NOT diagnosing the patient.

Return ONLY valid JSON.

Use exactly this schema:

{
  "chief_complaint": string,
  "symptoms": [
    {
      "name": string,
      "status": "present" | "absent" | "uncertain",
      "duration": {
        "value": integer,
        "unit": "hours" | "days" | "weeks"
      } | null
    }
  ],
  "medications": {
    "status": "unknown" | "none" | "reported",
    "items": []
  },
  "allergies": {
    "status": "unknown" | "none" | "reported",
    "items": []
  },
  "missing_information": []
}

Rules:

- Use English canonical clinical terms.
- Do not invent information.
- A symptom not mentioned must not be added.
- Preserve present, absent and uncertain status.
- Durations belong only to the symptom they describe.
- If medication information is not mentioned, use status "unknown".
- If the patient explicitly takes no medication, use status "none".
- If allergy information is not mentioned, use status "unknown".
- If the patient explicitly has no known allergies, use status "none".
- missing_information may contain only:
  "duration", "medications", "allergies".
- If the chief complaint duration is missing, include "duration".
- If medications are unknown, include "medications".
- If allergies are unknown, include "allergies".
""".strip()


# ============================================================
# BASE MODEL INFERENCE
# ============================================================

@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/models": model_volume,
    },
    timeout=3600,
)
def run_baseline(
    test_samples: list[dict],
) -> list[dict]:

    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    print("=" * 60)
    print(
        "NOOR HEALTH - QWEN3 BASE MODEL EVALUATION"
    )
    print("=" * 60)

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Samples: "
        f"{len(test_samples)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ========================================================
    # BASE MODEL
    # ========================================================

    print(
        f"\nLoading base model from:"
        f"\n{MODEL_DIR}"
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    )

    model.eval()

    print("\nBase model loaded.")

    # ========================================================
    # BUILD PROMPTS
    # ========================================================

    print("\nBuilding prompts...")

    prompts = []

    for sample in test_samples:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Language: "
                    f"{sample['language']}\n\n"
                    f"Patient statement:\n"
                    f"{sample['utterance']}"
                ),
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompts.append(prompt)

    # ========================================================
    # BATCHED INFERENCE
    # ========================================================

    results = []

    start_time = time.time()

    print("\nStarting inference...\n")

    for batch_start in range(
        0,
        len(test_samples),
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            len(test_samples),
        )

        batch_samples = test_samples[
            batch_start:batch_end
        ]

        batch_prompts = prompts[
            batch_start:batch_end
        ]

        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        inputs = {
            key: value.to("cuda")
            for key, value
            in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        # ----------------------------------------------------
        # REMOVE INPUT TOKENS
        # ----------------------------------------------------

        generated_tokens = outputs[
            :,
            input_length:
        ]

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # ----------------------------------------------------
        # SAVE BATCH RESULTS
        # ----------------------------------------------------

        for sample, prediction in zip(
            batch_samples,
            decoded,
        ):

            results.append(
                {
                    "case_id":
                        sample["case_id"],

                    "split":
                        sample["split"],

                    "language":
                        sample["language"],

                    "utterance":
                        sample["utterance"],

                    "target":
                        sample["target"],

                    "prediction":
                        prediction.strip(),
                }
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        completed = batch_end

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            completed / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"{completed}/"
            f"{len(test_samples)} "
            f"| {rate:.2f} samples/sec"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print("\n" + "=" * 60)

    print(
        "BASE MODEL INFERENCE COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nSamples: "
        f"{len(results)}"
    )

    print(
        f"Runtime: "
        f"{elapsed:.1f} seconds"
    )

    if elapsed > 0:

        print(
            f"Average: "
            f"{len(results) / elapsed:.2f} "
            f"samples/sec"
        )

    return results


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():

    """
    This function runs locally.

    Local filesystem paths are intentionally defined here
    instead of globally.

    Modal imports this file remotely as something like:

        /root/run_baseline_modal.py

    Therefore project paths based on __file__ must NOT be
    evaluated globally.
    """

    # ========================================================
    # LOCAL PROJECT PATHS
    # ========================================================

    # Current file:
    #
    # noor-health/
    # └── ml/
    #     └── evaluation/
    #         └── run_baseline_modal.py
    #
    # parents[0] -> evaluation
    # parents[1] -> ml
    # parents[2] -> noor-health

    root_dir = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    local_dataset = (
        root_dir
        / "data"
        / "processed"
        / "test_sft.jsonl"
    )

    output_file = (
        root_dir
        / "results"
        / "synthetic_test"
        / "base"
        / "predictions.jsonl"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 60)

    print(
        "NOOR HEALTH - FINAL BASE MODEL TEST SET"
    )

    print("=" * 60)

    print(
        f"\nProject root:"
        f"\n{root_dir}"
    )

    print(
        f"\nDataset:"
        f"\n{local_dataset}"
    )

    # ========================================================
    # LOAD FINAL TEST DATASET
    # ========================================================

    if not local_dataset.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{local_dataset}"
        )

    samples = []

    with local_dataset.open(
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

                sample = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSON in test dataset "
                    f"at line {line_number}."
                ) from exc

            samples.append(
                sample
            )

    print(
        f"\nTest samples: "
        f"{len(samples)}"
    )

    # ========================================================
    # DATASET SAFETY CHECKS
    # ========================================================

    if len(samples) != 700:

        raise ValueError(
            f"Expected 700 test samples, "
            f"found {len(samples)}."
        )

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    invalid_splits = [
        sample.get(
            "case_id",
            "<missing case_id>",
        )
        for sample in samples
        if sample.get("split") != "test"
    ]

    if invalid_splits:

        raise ValueError(
            "Found samples that are not "
            "marked as test."
        )

    # --------------------------------------------------------
    # Verify required fields
    # --------------------------------------------------------

    required_fields = {
        "case_id",
        "split",
        "language",
        "utterance",
        "target",
    }

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        missing_fields = (
            required_fields
            - set(sample.keys())
        )

        if missing_fields:

            raise ValueError(
                f"Sample {index} is missing "
                f"required fields: "
                f"{sorted(missing_fields)}"
            )

    # --------------------------------------------------------
    # Verify case count
    # --------------------------------------------------------

    case_ids = {
        sample["case_id"]
        for sample in samples
    }

    if len(case_ids) != 100:

        raise ValueError(
            f"Expected 100 unique test cases, "
            f"found {len(case_ids)}."
        )

    print(
        f"Unique test cases: "
        f"{len(case_ids)}"
    )

    # --------------------------------------------------------
    # Language distribution
    # --------------------------------------------------------

    language_counts = {}

    for sample in samples:

        language = sample[
            "language"
        ]

        language_counts[
            language
        ] = (
            language_counts.get(
                language,
                0,
            )
            + 1
        )

    print(
        "\nLanguage distribution:"
    )

    for language in sorted(
        language_counts
    ):

        print(
            f"  {language}: "
            f"{language_counts[language]}"
        )

    expected_languages = {
        "en",
        "de",
        "ar_msa",
        "fr",
        "es",
        "hi",
        "sw",
    }

    actual_languages = set(
        language_counts.keys()
    )

    if (
        actual_languages
        != expected_languages
    ):

        raise ValueError(
            "Unexpected language set.\n"
            f"Expected: "
            f"{sorted(expected_languages)}\n"
            f"Found: "
            f"{sorted(actual_languages)}"
        )

    # ========================================================
    # RUN BASE MODEL ON MODAL
    # ========================================================

    print(
        "\nStarting base model GPU inference..."
    )

    results = run_baseline.remote(
        samples
    )

    # ========================================================
    # RESULT SAFETY CHECKS
    # ========================================================

    if len(results) != len(samples):

        raise RuntimeError(
            f"Expected {len(samples)} "
            f"predictions, "
            f"received {len(results)}."
        )

    # Verify that result ordering / identity matches
    # the original test set.

    for index, (
        sample,
        result,
    ) in enumerate(
        zip(
            samples,
            results,
        ),
        start=1,
    ):

        if (
            sample["case_id"]
            != result["case_id"]
        ):

            raise RuntimeError(
                "Case ID mismatch at "
                f"position {index}: "
                f"{sample['case_id']} != "
                f"{result['case_id']}"
            )

        if (
            sample["language"]
            != result["language"]
        ):

            raise RuntimeError(
                "Language mismatch at "
                f"position {index}: "
                f"{sample['language']} != "
                f"{result['language']}"
            )

    # ========================================================
    # SAVE LOCALLY
    # ========================================================

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "BASE MODEL EVALUATION INFERENCE FINISHED"
    )

    print("=" * 60)

    print(
        f"\nPredictions saved to:"
        f"\n{output_file}"
    )

    print(
        f"\nPredictions: "
        f"{len(results)}"
    )