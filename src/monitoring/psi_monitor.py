"""Скрипт расчёта PSI и имитации потоковой проверки модели."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests

from src.api.inference import predict_proba

DEFAULT_OUT = Path("reports/psi_report.json")


def calculate_psi(
    expected: Sequence[float], actual: Sequence[float], bins: int = 10
) -> float:
    """Вычисляет Population Stability Index (PSI)."""

    expected = np.asarray(expected)
    actual = np.asarray(actual)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_perc = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def send_to_api(records: Iterable[dict], api_url: str) -> np.ndarray:
    response = requests.post(api_url, json=[dict(r) for r in records], timeout=10)
    response.raise_for_status()
    payload = response.json()
    preds = [item["default_probability"] for item in payload["predictions"]]
    return np.asarray(preds)


def run_monitoring(
    train_path: Path,
    new_path: Path,
    model_path: Path | None,
    out_path: Path,
    sample_size: int,
    api_url: str | None,
) -> dict:
    train = pd.read_csv(train_path)
    new = pd.read_csv(new_path)
    train_probs = train.get("pd_probability")

    if train_probs is None:
        if model_path is None and api_url is None:
            raise SystemExit("Не указан путь к модели или API для расчёта вероятностей")
        # Используем обученные данные для получения вероятностей
        train_probs = predict_proba(train.drop(columns=["target"]), model_path)
    else:
        train_probs = train_probs.to_numpy()

    new_records = new.sample(min(sample_size, len(new)), random_state=42)
    features = new_records.drop(columns=["target"], errors="ignore")
    if api_url:
        probs = send_to_api(features.to_dict(orient="records"), api_url)
    else:
        probs = predict_proba(features, model_path)

    psi_value = calculate_psi(train_probs, probs)
    report = {
        "psi": psi_value,
        "sample_size": int(len(features)),
        "api_url": api_url,
        "model_path": str(model_path) if model_path else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PSI report saved to {out_path} :: psi={psi_value:.4f}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Расчёт PSI для новых данных")
    parser.add_argument("--train", type=Path, required=True, help="Путь до train.csv")
    parser.add_argument(
        "--new", type=Path, required=True, help="Новые данные для мониторинга"
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="Путь до pickle с моделью"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sample-size", type=int, default=200, help="Размер батча новых данных"
    )
    parser.add_argument(
        "--api-url", type=str, default=None, help="Endpoint /predict FastAPI"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monitoring(
        args.train, args.new, args.model, args.out, args.sample_size, args.api_url
    )


if __name__ == "__main__":
    main()
