import numpy as np
import pandas as pd

from src.models.pipeline import create_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_fake_dataset():
    data = {
        "LIMIT_BAL": [20000, 50000, 100000, 150000],
        "AGE": [25, 32, 45, 52],
        "BILL_AMT1": [3913, 2682, 8617, 12000],
        "BILL_AMT2": [3102, 1725, 7890, 11000],
        "BILL_AMT3": [689, 2682, 9050, 10800],
        "BILL_AMT4": [0, 3272, 8000, 9500],
        "BILL_AMT5": [0, 3455, 7800, 9000],
        "BILL_AMT6": [0, 3261, 7500, 8700],
        "PAY_AMT1": [0, 0, 2000, 3000],
        "PAY_AMT2": [689, 1000, 2100, 2900],
        "PAY_AMT3": [0, 1000, 2200, 2800],
        "PAY_AMT4": [0, 1000, 2300, 2700],
        "PAY_AMT5": [0, 0, 2400, 2600],
        "PAY_AMT6": [0, 2000, 2500, 2500],
        "SEX": [1, 2, 2, 1],
        "EDUCATION": [2, 2, 1, 3],
        "MARRIAGE": [1, 2, 1, 3],
        "PAY_0": [0, -1, 0, 2],
        "PAY_2": [0, 2, 0, 2],
        "PAY_3": [0, 0, 0, 2],
        "PAY_4": [0, 0, 0, 2],
        "PAY_5": [0, 0, 0, 2],
        "PAY_6": [0, 0, 0, 2],
    }
    X = pd.DataFrame(data)
    y = np.array([0, 1, 0, 1])
    return X, y


def test_pipeline_fit_predict():
    X, y = build_fake_dataset()
    pipe, _, _ = create_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_feature_lists_cover_columns():
    X, _ = build_fake_dataset()
    expected_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert expected_cols.issubset(set(X.columns))
