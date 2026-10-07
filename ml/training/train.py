"""
LoRA fine-tuning of Qwen3-0.6B on Modal.

The local entrypoint reads the SFT data from data/processed/<data_version>/
and the system prompt from ml/prompts.py, and passes both to the GPU
function. No manual upload to a Modal volume is needed, and training and
evaluation are guaranteed to use the same prompt.

PowerShell:
    $env:NOOR_RUN_NAME = "lora_v3"        # adapter name on the model volume
    $env:NOOR_DATA_VERSION = "v3"         # data/processed/v3/
    $env:NOOR_PROMPT_VERSION = "v3"       # key in ml/prompts.py
    modal run ml/training/train.py
"""

import json
import os
import sys
from pathlib import Path

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "noor-health-training"

BASE_MODEL = "Qwen/Qwen3-0.6B"

RUN_NAME = os.environ.get("NOOR_RUN_NAME", "lora_v3")
DATA_VERSION = os.environ.get("NOOR_DATA_VERSION", "v3")
PROMPT_VERSION = os.environ.get("NOOR_PROMPT_VERSION", "v3")

# Expected (train, validation) sizes per data version. A mismatch means the
# wrong or an incomplete data folder; checked locally before the GPU starts.
EXPECTED_SAMPLES = {
    "v2": (5589, 699),
    "v3": (3200, 400),
}

# Model volume layout: /outputs/noor-health-qwen3-<run_name>/final-adapter
# (the v2 adapter keeps its original folder noor-health-qwen3-lora/)
VOLUME_ROOT = "/outputs"

# Raised from 512: the v3 system prompt is longer. Records that would still
# be truncated abort the training instead of silently losing target tokens.
MAX_LENGTH = 1536

NUM_EPOCHS = 3

LEARNING_RATE = 2e-4

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ============================================================
# MODAL
# ============================================================

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "safetensors",
    )
)

model_volume = modal.Volume.from_name(
    "noor-health-models",
    create_if_missing=True,
)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def build_messages(record, system_prompt):
    # record["user_prompt"] is built locally with prompts.build_user_prompt
    user_prompt = record["user_prompt"]

    target = json.dumps(
        record["target"],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
        {
            "role": "assistant",
            "content": target,
        },
    ]


# ============================================================
# TRAINING
# ============================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 3,
    volumes={
        VOLUME_ROOT: model_volume,
    },
)
def train(
    train_records: list[dict],
    validation_records: list[dict],
    system_prompt: str,
    output_dir: str,
    run_info: dict,
):
    import torch

    from datasets import Dataset

    from peft import (
        LoraConfig,
        get_peft_model,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    print("=" * 60)
    print("NOOR HEALTH - QWEN3 LoRA TRAINING")
    print("=" * 60)

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(f"Run: {run_info}")

    print(f"\nTrain samples:      {len(train_records)}")
    print(f"Validation samples: {len(validation_records)}")

    # Sample counts are checked in the local entrypoint (EXPECTED_SAMPLES),
    # before any GPU time is used.

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nTrainable parameters:")

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # TOKENIZATION
    #
    # Important:
    # Only assistant tokens contribute to the loss.
    # System + user tokens receive label -100.
    # --------------------------------------------------------

    def tokenize_record(record):
        messages = build_messages(record, system_prompt)

        prompt_messages = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        untruncated_length = len(tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"])

        if untruncated_length > MAX_LENGTH:
            raise ValueError(
                f"Record {record.get('case_id')} has {untruncated_length} tokens "
                f"(> MAX_LENGTH={MAX_LENGTH}); target tokens would be lost."
            )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        labels = full_tokens.copy()

        prompt_length = min(
            len(prompt_tokens),
            len(labels),
        )

        for index in range(prompt_length):
            labels[index] = -100

        attention_mask = [1] * len(full_tokens)

        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    print("\nTokenizing training data...")

    train_dataset = Dataset.from_list(
        train_records
    )

    validation_dataset = Dataset.from_list(
        validation_records
    )

    train_dataset = train_dataset.map(
        tokenize_record,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    validation_dataset = validation_dataset.map(
        tokenize_record,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing validation",
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print("\nRunning tokenization sanity check...")

    example = train_dataset[0]

    supervised_tokens = sum(
        label != -100
        for label in example["labels"]
    )

    print(
        f"Sequence length: "
        f"{len(example['input_ids'])}"
    )

    print(
        f"Supervised assistant tokens: "
        f"{supervised_tokens}"
    )

    if supervised_tokens == 0:
        raise ValueError(
            "Training example contains no "
            "supervised assistant tokens."
        )

    # --------------------------------------------------------
    # COLLATOR
    # --------------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_steps=0.05,

        weight_decay=0.01,

        logging_steps=20,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        report_to="none",

        remove_unused_columns=False,

        seed=42,
        data_seed=42,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting training...\n")

    result = trainer.train()

    print("\nTraining complete.")

    print(
        f"Training loss: "
        f"{result.training_loss:.4f}"
    )

    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------

    print("\nRunning final validation...")

    metrics = trainer.evaluate()

    print("\nValidation metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # SAVE ADAPTER
    # --------------------------------------------------------

    final_adapter_dir = (
        f"{output_dir}/final-adapter"
    )

    print(
        f"\nSaving LoRA adapter to:\n"
        f"{final_adapter_dir}"
    )

    trainer.save_model(
        final_adapter_dir
    )

    tokenizer.save_pretrained(
        final_adapter_dir
    )

    # --------------------------------------------------------
    # SAVE TRAINING METRICS
    # --------------------------------------------------------

    metrics_file = (
        f"{output_dir}/training_metrics.json"
    )

    final_metrics = {
        "base_model": BASE_MODEL,
        **run_info,
        "max_length": MAX_LENGTH,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "training_loss": result.training_loss,
        "validation_metrics": metrics,
    }

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    model_volume.commit()

    print("\nModel saved successfully.")

    return final_metrics


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():
    # Local-only imports: the container never needs prompts.py
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml"))
    from prompts import SYSTEM_PROMPTS, build_user_prompt

    data_dir = repo_root / "data" / "processed" / DATA_VERSION
    if DATA_VERSION == "v2":
        data_dir = repo_root / "data" / "processed"

    def load_split(name):
        records = load_jsonl(data_dir / f"{name}_sft.jsonl")
        for r in records:
            if r["split"] != name:
                raise ValueError(f"{r['case_id']} has split {r['split']!r} in {name}_sft.jsonl")
            r["user_prompt"] = build_user_prompt(r["language"], r["utterance"])
        return records

    train_records = load_split("train")
    validation_records = load_split("validation")

    expected = EXPECTED_SAMPLES.get(DATA_VERSION)
    if expected and (len(train_records), len(validation_records)) != expected:
        raise ValueError(
            f"Expected {expected[0]} / {expected[1]} train / validation samples for "
            f"{DATA_VERSION}, found {len(train_records)} / {len(validation_records)}"
        )

    overlap = {r["case_id"] for r in train_records} & {r["case_id"] for r in validation_records}
    if overlap:
        raise ValueError(f"case_ids in train AND validation: {sorted(overlap)[:5]}")

    folder = "noor-health-qwen3-lora" if RUN_NAME == "lora_v2" else f"noor-health-qwen3-{RUN_NAME}"
    output_dir = f"{VOLUME_ROOT}/{folder}"

    run_info = {
        "run_name": RUN_NAME,
        "data_version": DATA_VERSION,
        "prompt_version": PROMPT_VERSION,
    }

    print("=" * 60)
    print(f"Run name:       {RUN_NAME}")
    print(f"Data:           {data_dir}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Train / val:    {len(train_records)} / {len(validation_records)}")
    print(f"Output:         {output_dir} (Modal volume noor-health-models)")
    print("=" * 60)

    metrics = train.remote(
        train_records,
        validation_records,
        SYSTEM_PROMPTS[PROMPT_VERSION],
        output_dir,
        run_info,
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )