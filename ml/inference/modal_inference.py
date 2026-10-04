import json

import modal


# ============================================================
# MODAL CONFIG
# ============================================================

app = modal.App("noor-health-inference")

image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
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

ADAPTER_DIR = "/models/noor-health-qwen3-lora/final-adapter"


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
# MODEL
# ============================================================

@app.cls(
    image=image,
    gpu="L4",
    volumes={
        "/models": model_volume,
    },
    timeout=600,
    scaledown_window=300,
)
class NoorModel:

    @modal.enter()
    def load_model(self):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        print("Loading Noor Health model...")

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_DIR
            )
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        print(
            "Loading Noor Health LoRA adapter from:",
            ADAPTER_DIR,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
        )

        self.model.eval()

        print("Model ready.")


    @modal.method()
    def predict(
        self,
        text: str,
        language: str,
    ) -> dict:

        import torch

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Language: {language}\n\n"
                    f"Patient statement:\n{text}"
                ),
            },
        ]

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = outputs[
            :,
            input_length:
        ]

        prediction = (
            self.tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        print(
            f"Raw prediction:\n{prediction}"
        )

        try:
            result = json.loads(
                prediction
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Model returned invalid JSON."
            ) from exc

        return result