import joblib
import numpy as np
import pandas as pd

from src.api import inference
from src.api.app import BatchPredictionResponse, ClientFeatures, predict
from src.models.pipeline import create_pipeline


PAYLOAD = {
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


def build_model(tmp_path):
    X = pd.DataFrame([PAYLOAD, {**PAYLOAD, "SEX": 1, "PAY_0": 1}])
    y = np.array([0, 1])
    pipe, _, _ = create_pipeline()
    pipe.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(pipe, model_path)
    return model_path


class DummyModel:
    def predict_proba(self, df):
        probs = np.full(shape=(len(df),), fill_value=0.65)
        return np.vstack([1 - probs, probs]).T


def test_predict_function_returns_batch_response():
    payload = ClientFeatures(**PAYLOAD)
    result = predict([payload], model=DummyModel())
    assert isinstance(result, BatchPredictionResponse)
    pred = result.predictions[0]
    assert pred.default_prediction in {0, 1}
    assert pred.default_probability == 0.65


def test_inference_predict_proba_loads_model(tmp_path, monkeypatch):
    model_path = build_model(tmp_path)
    monkeypatch.setenv(inference.MODEL_ENV_VAR, str(model_path))
    inference.reset_model_cache()
    probs = inference.predict_proba([PAYLOAD])
    assert probs.shape == (1,)
    assert 0 <= probs[0] <= 1
