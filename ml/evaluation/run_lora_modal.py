import json
import time
from pathlib import Path

import modal


# ============================================================
# LOCAL PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"

LOCAL_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "test_sft.jsonl"
)


# ============================================================
# MODAL CONFIG
# ============================================================

app = modal.App("noor-health-lora-evaluation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "safetensors",
        "peft",
    )
)

model_volume = modal.Volume.from_name(
    "noor-health-models",
    create_if_missing=True,
)

MODEL_DIR = "/models/qwen3-0.6b"

ADAPTER_DIR = (
    "/models/noor-health-qwen3-lora/final-adapter"
)

BATCH_SIZE = 32


# ============================================================
# SYSTEM PROMPT
# ============================================================

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
# LORA INFERENCE
# ============================================================

@app.function(
    image=image,
    gpu="L4",
    volumes={"/models": model_volume},
    timeout=3600,
)
def run_lora(
    test_samples: list[dict],
):

    import torch

    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    print("=" * 60)
    print("NOOR HEALTH - QWEN3 + LORA EVALUATION")
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
        MODEL_DIR,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # BASE MODEL
    # ========================================================

    print(
        f"\nLoading base model from:"
        f"\n{MODEL_DIR}"
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    )

    # ========================================================
    # LORA ADAPTER
    # ========================================================

    print(
        f"\nLoading LoRA adapter from:"
        f"\n{ADAPTER_DIR}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.eval()

    print("\nFine-tuned model loaded.")

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

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

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

        generated_tokens = outputs[
            :,
            input_length:
        ]

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

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
    print("LORA INFERENCE COMPLETE")
    print("=" * 60)

    print(
        f"\nSamples: "
        f"{len(results)}"
    )

    print(
        f"Runtime: "
        f"{elapsed:.1f} seconds"
    )

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

    print("=" * 60)
    print("NOOR HEALTH - LORA TEST SET")
    print("=" * 60)

    # ========================================================
    # LOAD FINAL TEST DATASET
    # ========================================================

    if not LOCAL_DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{LOCAL_DATASET}"
        )

    samples = []

    with LOCAL_DATASET.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            sample = json.loads(line)

            samples.append(sample)

    print(
        f"\nDataset:"
        f"\n{LOCAL_DATASET}"
    )

    print(
        f"\nTest samples: "
        f"{len(samples)}"
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if len(samples) != 700:

        raise ValueError(
            f"Expected 700 test samples, "
            f"found {len(samples)}"
        )

    invalid_splits = [
        sample["case_id"]
        for sample in samples
        if sample.get("split") != "test"
    ]

    if invalid_splits:

        raise ValueError(
            "Found samples that are not "
            "marked as test."
        )

    case_ids = {
        sample["case_id"]
        for sample in samples
    }

    print(
        f"Unique test cases: "
        f"{len(case_ids)}"
    )

    # ========================================================
    # RUN LORA INFERENCE
    # ========================================================

    print(
        "\nStarting LoRA GPU inference..."
    )

    results = run_lora.remote(
        samples
    )

    # ========================================================
    # VERIFY RESULT COUNT
    # ========================================================

    if len(results) != len(samples):

        raise RuntimeError(
            f"Expected {len(samples)} "
            f"predictions, "
            f"received {len(results)}."
        )

    # ========================================================
    # SAVE LOCALLY
    # ========================================================

    output_file = (
        RESULTS_DIR
        / "lora_predictions.jsonl"
    )

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

    print("\n" + "=" * 60)
    print("EVALUATION INFERENCE FINISHED")
    print("=" * 60)

    print(
        f"\nPredictions saved to:"
        f"\n{output_file}"
    )

    print(
        f"\nPredictions: "
        f"{len(results)}"
    )