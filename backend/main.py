import modal
import json
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Noor Health API",
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST
# ============================================================

class IntakeRequest(BaseModel):
    text: str
    language: str


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class Duration(BaseModel):
    value: int
    unit: Literal[
        "hours",
        "days",
        "weeks",
    ]


class Symptom(BaseModel):
    name: str
    status: Literal[
        "present",
        "absent",
        "uncertain",
    ]
    duration: Duration | None


class InformationField(BaseModel):
    status: Literal[
        "unknown",
        "none",
        "reported",
    ]
    items: list[str]


class ClinicalIntake(BaseModel):
    chief_complaint: str
    symptoms: list[Symptom]
    medications: InformationField
    allergies: InformationField
    missing_information: list[str]


# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "noor-health",
    }


@app.post(
    "/api/intake",
    response_model=ClinicalIntake,
)
def structure_intake(
    request: IntakeRequest,
):
    NoorModel = modal.Cls.from_name(
        "noor-health-inference",
        "NoorModel",
    )

    noor_model = NoorModel()

    result = noor_model.predict.remote(
        text=request.text,
        language=request.language,
    )

    print("\n===== MODAL PREDICTION =====")
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("============================\n")

    validated_result = ClinicalIntake.model_validate(
        result
    )

    return validated_result