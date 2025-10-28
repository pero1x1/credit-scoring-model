"""Вспомогательные функции для инференса модели PD."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/credit_default_model.pkl")
MODEL_ENV_VAR = "CREDIT_MODEL_PATH"


class ModelNotFoundError(FileNotFoundError):
    """Ошибка, возникающая при отсутствии сохранённой модели."""


def get_model_path() -> Path:
    """Возвращает путь к модели из переменной окружения или значение по умолчанию."""

    env_path = os.getenv(MODEL_ENV_VAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_MODEL_PATH


@lru_cache(maxsize=2)
def load_model(model_path: str | Path | None = None):
    """Загружает модель из joblib-файла с кэшированием."""

    path = Path(model_path) if model_path else get_model_path()
    if not path.exists():
        raise ModelNotFoundError(f"Модель не найдена по пути {path}")
    return joblib.load(path)


def reset_model_cache() -> None:
    """Сбрасывает кэш загруженной модели (удобно для тестов)."""

    load_model.cache_clear()  # type: ignore[attr-defined]


def ensure_dataframe(data: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Преобразует входные данные в DataFrame с сохранением порядка колонок."""

    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(list(data))


def predict_proba(
    records: Iterable[dict] | pd.DataFrame, model_path: str | Path | None = None
) -> np.ndarray:
    """Возвращает вероятности дефолта для набора записей."""

    df = ensure_dataframe(records)
    model = load_model(model_path)
    return model.predict_proba(df)[:, 1]


def batch_to_json(records: Iterable[dict], probs: Iterable[float]) -> str:
    """Готовит JSON отчёт для мониторинга."""

    joined = []
    for row, prob in zip(records, probs):
        payload = dict(row)
        payload["pd_probability"] = float(prob)
        joined.append(payload)
    return json.dumps(joined, ensure_ascii=False, indent=2)
