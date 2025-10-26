import json

import argparse
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

try:
    from .pipeline import create_pipeline
except ImportError:
    from src.models.pipeline import create_pipeline

def load_split(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test  = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test,  y_test  = test.drop(columns=['target']),  test['target']
    return X_train, X_test, y_train, y_test

def tune_and_fit(pipe, X_train, y_train):
    # Простейший RandomizedSearch для GBDT
    param_dist = {
        'clf__n_estimators': randint(60, 200),
        'clf__max_depth': randint(2, 5),
        'clf__learning_rate': [0.02, 0.05, 0.1, 0.2],
        'clf__subsample': [0.7, 0.9, 1.0]
    }
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=15, cv=3,
        n_jobs=-1, random_state=42, scoring='roc_auc', verbose=0
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def plot_roc(y_true, y_prob, out):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main(data_dir: str, models_dir: str, mlruns_dir: str):
    data_dir = Path(data_dir); models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(data_dir)

    mlflow.set_tracking_uri(Path(mlruns_dir).absolute().as_uri())
    mlflow.set_experiment("Credit_Default_Prediction")

    with mlflow.start_run():
        pipe, num, cat = create_pipeline()
        best_pipe, best_params = tune_and_fit(pipe, X_train, y_train)

        # Метрики
        y_prob = best_pipe.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Логи MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)

        # ROC-кривая как артефакт
        roc_path = models_dir / "roc_curve.png"
        plot_roc(y_test, y_prob, roc_path)
        mlflow.log_artifact(str(roc_path))

        # Сохраняем метрики для DVC
        metrics = {
            'test_auc': float(auc),
            'test_precision': float(prec),
            'test_recall': float(rec),
            'test_f1': float(f1)
        }
        metrics_path = models_dir.parent / 'data' / 'processed' / 'metrics.json'
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

        # Сохраняем модель локально и в MLflow
        model_path = models_dir / "credit_default_model.pkl"
        joblib.dump(best_pipe, model_path)
        mlflow.sklearn.log_model(best_pipe, "model", registered_model_name="CreditDefaultModel")

        print(f"OK. AUC={auc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
        print(f"Model -> {model_path}")
        print(f"ROC -> {roc_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--mlruns_dir", default="mlruns")  # локально в проекте
    args = p.parse_args()
    main(args.data_dir, args.models_dir, args.mlruns_dir)
