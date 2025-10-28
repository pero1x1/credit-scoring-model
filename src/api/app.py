"""FastAPI приложение для инференса кредитной скоринговой модели."""

from __future__ import annotations

from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.inference import (
    ModelNotFoundError,
    batch_to_json,
    ensure_dataframe,
    get_model_path,
    load_model,
    predict_proba,
)
from src.models.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

app = FastAPI(title="Credit Default Prediction API", version="1.0.0")

ALL_FEATURES = [
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]


class ClientFeatures(BaseModel):
    """Описание входных признаков для PD-модели."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 20000,
                "AGE": 30,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 0,
                "BILL_AMT5": 0,
                "BILL_AMT6": 0,
                "PAY_AMT1": 0,
                "PAY_AMT2": 689,
                "PAY_AMT3": 0,
                "PAY_AMT4": 0,
                "PAY_AMT5": 0,
                "PAY_AMT6": 0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
            }
        }
    )

    LIMIT_BAL: float = Field(..., ge=1, description="Кредитный лимит")
    AGE: int = Field(..., ge=18, le=100)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)
    SEX: int = Field(..., ge=1, le=2)
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    PAY_0: int = Field(..., ge=-2, le=9)
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    default_prediction: int = Field(..., ge=0, le=1)
    model_path: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


def _model_dependency():
    try:
        path = get_model_path()
        model = load_model(path)
    except ModelNotFoundError as exc:  # pragma: no cover - ручное тестирование
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return model


@app.get("/")
def root() -> dict:
    return {
        "message": "Credit Default Prediction API",
        "model_path": str(get_model_path()),
        "expected_features": ALL_FEATURES,
    }


@app.post("/predict", response_model=BatchPredictionResponse)
def predict(
    data: List[ClientFeatures], model=Depends(_model_dependency)
) -> BatchPredictionResponse:
    """Возвращает предсказание дефолта для батча клиентов."""

    df = ensure_dataframe([item.model_dump() for item in data])
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Отсутствуют признаки: {sorted(missing)}"
        )

    probs = model.predict_proba(df[ALL_FEATURES])[:, 1]
    preds = (probs >= 0.5).astype(int)
    items = [
        PredictionResponse(
            default_probability=float(prob),
            default_prediction=int(pred),
            model_path=str(get_model_path()),
        )
        for prob, pred in zip(probs, preds)
    ]
    return BatchPredictionResponse(predictions=items)


@app.post("/predict:raw")
def predict_raw(data: List[ClientFeatures]) -> dict:
    """Отладочный endpoint: возвращает JSON с исходными признаками и вероятностями."""

    frame = pd.DataFrame([item.model_dump() for item in data])
    probs = predict_proba(frame)
    return {
        "model_path": str(get_model_path()),
        "payload": batch_to_json(frame.to_dict(orient="records"), probs),
    }


@app.get("/health")
def healthcheck(model=Depends(_model_dependency)) -> dict:
    return {
        "status": "ok",
        "model_path": str(get_model_path()),
        "features": ALL_FEATURES,
    }
