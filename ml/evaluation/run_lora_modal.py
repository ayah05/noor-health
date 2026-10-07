import json
import sys
import time
from pathlib import Path

import modal


# ============================================================
# LOCAL PATHS
# ============================================================

# Locally this file lives at ml/evaluation/run_lora_modal.py,
# so the repository root is two levels up.
#
# Modal also imports this module INSIDE the container, where
# it is copied to /root/run_lora_modal.py and has no parents[2].
# The paths below are only used by the local entrypoint, so in
# the container a harmless fallback is enough.
_THIS_FILE = Path(__file__).resolve()

ROOT_DIR = (
    _THIS_FILE.parents[2]
    if len(_THIS_FILE.parents) > 2
    else _THIS_FILE.parent
)

RESULTS_DIR = ROOT_DIR / "results"

# ============================================================
# EVALUATION SET
# ============================================================
#
# Select with the environment variable NOOR_EVAL_SET.
# Default is the synthetic held-out test set, so existing
# behaviour is unchanged.
#
# Predictions are written to
#   results/<results_dir>/<NOOR_RUN_NAME>/predictions.jsonl
# NOOR_RUN_NAME identifies the model version (default: lora_v2).
#
# PowerShell:
#   $env:NOOR_EVAL_SET = "real_test"
#   $env:NOOR_RUN_NAME = "lora_v2"
#   modal run ml/evaluation/run_lora_modal.py

import os

EVAL_SETS = {
    "synthetic_test": {
        "dataset": "test_sft.jsonl",
        "split": "test",
        "expected_samples": 700,
        "results_dir": "synthetic_test",
    },
    "synthetic_test_v3": {
        "dataset": "v3/test_sft.jsonl",
        "split": "test",
        "expected_samples": 400,
        "results_dir": "synthetic_test_v3",
    },
    "real_test": {
        "dataset": "real_test_sft.jsonl",
        "split": "real_test",
        "expected_samples": None,
        "results_dir": "real_test_v1",
    },
}

EVAL_SET_NAME = os.environ.get(
    "NOOR_EVAL_SET",
    "synthetic_test",
)

if EVAL_SET_NAME not in EVAL_SETS:
    raise ValueError(
        f"Unknown NOOR_EVAL_SET {EVAL_SET_NAME!r}. "
        f"Choose one of {sorted(EVAL_SETS)}."
    )

EVAL_SET = EVAL_SETS[EVAL_SET_NAME]

RUN_NAME = os.environ.get(
    "NOOR_RUN_NAME",
    "lora_v3",
)

# Which adapter, which system prompt (key in ml/prompts.py) and which
# language-code mapping each run uses. The prompt MUST be the one the
# adapter was trained with.
RUNS = {
    # Reproduces the earlier v2 numbers (evaluated with a prompt that
    # differed from the training prompt).
    "lora_v2": {
        "adapter": "/models/noor-health-qwen3-lora/final-adapter",
        "prompt": "v2_eval",
        "language_map": {"ar_eg": "ar_msa"},
    },
    # Fair v2 baseline: same adapter, evaluated with its TRAINING prompt.
    "lora_v2_trainprompt": {
        "adapter": "/models/noor-health-qwen3-lora/final-adapter",
        "prompt": "v2",
        "language_map": {"ar_eg": "ar_msa"},
    },
    "lora_v3": {
        "adapter": "/models/noor-health-qwen3-lora_v3/final-adapter",
        "prompt": "v3",
        "language_map": {},
    },
}

if RUN_NAME not in RUNS:
    raise ValueError(f"Unknown NOOR_RUN_NAME {RUN_NAME!r}. Choose one of {sorted(RUNS)}.")

RUN = RUNS[RUN_NAME]

LOCAL_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / EVAL_SET["dataset"]
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

ADAPTER_DIR = RUN["adapter"]

BATCH_SIZE = 32


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
    system_prompt: str,
    adapter_dir: str,
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
        f"\n{adapter_dir}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
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
                "content": system_prompt,
            },
            {
                "role": "user",
                # built locally with prompts.build_user_prompt
                "content": sample["user_prompt"],
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
                        sample.get("language_original") or sample["language"],

                    "utterance":
                        sample["utterance"],

                    "target":
                        sample["target"],

                    "prediction":
                        prediction.strip(),

                    # Optional metadata (real test set only).
                    "language_original":
                        sample.get("language_original"),

                    "input_mode":
                        sample.get("input_mode"),
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
    print(f"NOOR HEALTH - LORA EVALUATION ({EVAL_SET_NAME})")
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

    expected = EVAL_SET["expected_samples"]

    if (
        expected is not None
        and len(samples) != expected
    ):

        raise ValueError(
            f"Expected {expected} samples "
            f"for {EVAL_SET_NAME}, "
            f"found {len(samples)}"
        )

    invalid_splits = [
        sample["case_id"]
        for sample in samples
        if sample.get("split") != EVAL_SET["split"]
    ]

    if invalid_splits:

        raise ValueError(
            f"Found samples that are not "
            f"marked as {EVAL_SET['split']}."
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

    # Local-only import: the container never needs prompts.py
    sys.path.insert(0, str(ROOT_DIR / "ml"))
    from prompts import SYSTEM_PROMPTS, build_user_prompt

    for sample in samples:
        original = sample.get("language_original", sample["language"])
        prompt_language = RUN["language_map"].get(original, original)
        sample["user_prompt"] = build_user_prompt(prompt_language, sample["utterance"])

    print(f"Run: {RUN_NAME} | adapter: {RUN['adapter']} | prompt: {RUN['prompt']}")

    results = run_lora.remote(
        samples,
        SYSTEM_PROMPTS[RUN["prompt"]],
        RUN["adapter"],
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
        / EVAL_SET["results_dir"]
        / RUN_NAME
        / "predictions.jsonl"
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

    print(
        "\nNext step:\n"
        f"python ml/evaluation/evaluate_predictions.py "
        f"--predictions {output_file.relative_to(ROOT_DIR).as_posix()} "
        f"--name qwen3-0.6b_{RUN_NAME}"
    )