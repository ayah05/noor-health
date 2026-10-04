import json
from pathlib import Path

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "noor-health-training"

BASE_MODEL = "Qwen/Qwen3-0.6B"

REMOTE_DATA_DIR = "../data"
REMOTE_OUTPUT_DIR = "/outputs/noor-health-qwen3-lora"

TRAIN_FILE = f"{REMOTE_DATA_DIR}/train_sft.jsonl"
VALIDATION_FILE = f"{REMOTE_DATA_DIR}/validation_sft.jsonl"

MAX_LENGTH = 512

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

training_data_volume = modal.Volume.from_name(
    "noor-health-training-data",
    create_if_missing=True,
)

model_volume = modal.Volume.from_name(
    "noor-health-models",
    create_if_missing=True,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Noor Health, a clinical intake structuring assistant.

Your task is to convert a patient's statement into structured JSON.

You do not diagnose.
You do not recommend treatment.
You do not infer information that the patient did not provide.

Return JSON only.

The JSON must have exactly this structure:

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

- The chief complaint must be explicitly supported by the patient statement.
- Do not invent symptoms.
- Do not invent durations.
- Do not assign one symptom's duration to another symptom.
- Negated symptoms must have status "absent".
- Uncertain symptoms must have status "uncertain".
- Absent or uncertain symptoms must have duration null.

Medication rules:
- If medications are not mentioned, use status "unknown".
- If the patient explicitly reports taking no medication, use status "none".
- If medications are reported, use status "reported" and list them.

Allergy rules:
- If allergies are not mentioned, use status "unknown".
- If the patient explicitly reports no allergies, use status "none".
- If allergies are reported, use status "reported" and list them.

Missing information rules:
- Add "duration" when the chief complaint has no reported duration.
- Add "medications" when medication status is "unknown".
- Add "allergies" when allergy status is "unknown".

Preserve uncertainty.
Never convert missing information into negative information.
""".strip()


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


def build_messages(record):
    language = record["language"]
    utterance = record["utterance"]

    user_prompt = (
        f"Language: {language}\n\n"
        f"Patient statement:\n"
        f"{utterance}"
    )

    target = json.dumps(
        record["target"],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
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
        REMOTE_DATA_DIR: training_data_volume,
        "/outputs": model_volume,
    },
)
def train():
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

    train_records = load_jsonl(TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)

    print(f"\nTrain samples:      {len(train_records)}")
    print(f"Validation samples: {len(validation_records)}")

    if len(train_records) != 5589:
        raise ValueError(
            f"Expected 5589 training samples, "
            f"found {len(train_records)}"
        )

    if len(validation_records) != 699:
        raise ValueError(
            f"Expected 699 validation samples, "
            f"found {len(validation_records)}"
        )

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
        messages = build_messages(record)

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
        output_dir=REMOTE_OUTPUT_DIR,

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
        f"{REMOTE_OUTPUT_DIR}/final-adapter"
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
        f"{REMOTE_OUTPUT_DIR}/training_metrics.json"
    )

    final_metrics = {
        "base_model": BASE_MODEL,
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
    metrics = train.remote()

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )