
import pandas as pd
import pandera as pa
from src.data.validation import CreditSchema

def test_credit_schema_valid_minimal():
    row = {
        "ID": 1, "LIMIT_BAL": 20000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 0.0, "BILL_AMT2": 0.0, "BILL_AMT3": 0.0,
        "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
        "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
        "target": 0
    }
    df = pd.DataFrame([row])
    # не должно бросать исключений
    CreditSchema.validate(df)
