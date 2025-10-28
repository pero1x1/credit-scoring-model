"""Запуск серии экспериментов с логированием в MLflow."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import Bunch

from src.models.pipeline import create_pipeline


def load_dataset(data_dir: Path) -> Bunch:
    train = pd.read_csv(data_dir / "train.csv")
    X = train.drop(columns=["target"])
    y = train["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return Bunch(X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)


def evaluate(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    proba = model.predict_proba(X_valid)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "valid_auc": roc_auc_score(y_valid, proba),
        "valid_precision": precision_score(y_valid, preds),
        "valid_recall": recall_score(y_valid, preds),
        "valid_f1": f1_score(y_valid, preds),
    }


def _preprocessor_template():
    pipeline, _, _ = create_pipeline()
    return clone(pipeline.named_steps["preprocessor"])


def run_gradient_boosting() -> Iterable[tuple[str, Pipeline, dict]]:
    base_pipeline, _, _ = create_pipeline()
    for lr, depth in product([0.03, 0.05, 0.1], [2, 3, 4]):
        pipe = clone(base_pipeline)
        pipe.set_params(
            clf__learning_rate=lr,
            clf__max_depth=depth,
            clf__subsample=0.9,
            clf__random_state=42,
        )
        yield f"gb_lr{lr}_depth{depth}", pipe, {
            "learning_rate": lr,
            "max_depth": depth,
            "subsample": 0.9,
        }


def run_random_forest() -> Iterable[tuple[str, Pipeline, dict]]:
    for depth in [4, 6]:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=depth,
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline([("preprocessor", _preprocessor_template()), ("clf", clf)])
        yield f"rf_depth{depth}", pipe, {"max_depth": depth}


def run_logreg() -> Iterable[tuple[str, Pipeline, dict]]:
    pipe = Pipeline(
        [
            ("preprocessor", _preprocessor_template()),
            ("clf", LogisticRegression(max_iter=200, solver="lbfgs")),
        ]
    )
    yield "logreg_lbfgs", pipe, {"solver": "lbfgs"}


def run_experiments(data_dir: Path, mlruns_dir: Path) -> None:
    dataset = load_dataset(data_dir)
    mlflow.set_tracking_uri(mlruns_dir.absolute().as_uri())
    mlflow.set_experiment("Credit_Default_Prediction")

    experiments = []
    experiments.extend(run_gradient_boosting())
    experiments.extend(run_random_forest())
    experiments.extend(run_logreg())

    for name, pipe, params in experiments:
        with mlflow.start_run(run_name=name):
            pipe.fit(dataset.X_train, dataset.y_train)
            metrics = evaluate(pipe, dataset.X_valid, dataset.y_valid)
            mlflow.log_params(
                {"model": pipe.named_steps["clf"].__class__.__name__, **params}
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            print(
                f"[{name}] AUC={metrics['valid_auc']:.4f} F1={metrics['valid_f1']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed", type=Path)
    parser.add_argument("--mlruns_dir", default="mlruns", type=Path)
    args = parser.parse_args()

    run_experiments(args.data_dir, args.mlruns_dir)


if __name__ == "__main__":
    main()
