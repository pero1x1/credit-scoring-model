import argparse
import pandera as pa
import pandera.typing as pt
import pandas as pd
from pathlib import Path


class CreditSchema(pa.SchemaModel):
    ID: pt.Series[int] = pa.Field(coerce=True, nullable=False)
    LIMIT_BAL: pt.Series[float] = pa.Field(gt=0, le=1_000_000, coerce=True)
    SEX: pt.Series[int] = pa.Field(isin=[1, 2], coerce=True)
    EDUCATION: pt.Series[int] = pa.Field(
        isin=[0, 1, 2, 3, 4, 5, 6], coerce=True
    )  # встречаются 0/5/6
    MARRIAGE: pt.Series[int] = pa.Field(isin=[0, 1, 2, 3], coerce=True)  # иногда 0
    AGE: pt.Series[int] = pa.Field(ge=18, le=100, coerce=True)

    PAY_0: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)
    PAY_2: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)
    PAY_3: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)
    PAY_4: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)
    PAY_5: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)
    PAY_6: pt.Series[int] = pa.Field(ge=-2, le=9, coerce=True)

    BILL_AMT1: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)
    BILL_AMT2: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)
    BILL_AMT3: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)
    BILL_AMT4: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)
    BILL_AMT5: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)
    BILL_AMT6: pt.Series[float] = pa.Field(ge=-1_000_000, le=10_000_000, coerce=True)

    PAY_AMT1: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)
    PAY_AMT2: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)
    PAY_AMT3: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)
    PAY_AMT4: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)
    PAY_AMT5: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)
    PAY_AMT6: pt.Series[float] = pa.Field(ge=0, le=10_000_000, coerce=True)

    target: pt.Series[int] = pa.Field(isin=[0, 1], nullable=False, coerce=True)

    class Config:
        strict = True  # лишние колонки — ошибка
        coerce = True  # приводим типы


def validate_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    try:
        CreditSchema.validate(df, lazy=True)
        return {"ok": True, "errors": []}
    except pa.errors.SchemaErrors as err:
        # возвращаем компактный отчёт
        report = (
            err.failure_cases[["column", "check", "failure_case"]]
            .head(20)
            .to_dict(orient="records")
        )
        return {"ok": False, "errors": report}


def main(
    train_csv: str,
    test_csv: str,
    out_json: str = "data/processed/validation_report.json",
):
    report = {
        "train": validate_csv(Path(train_csv)),
        "test": validate_csv(Path(test_csv)),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Validation report -> {out_json}")
    if (not report["train"]["ok"]) or (not report["test"]["ok"]):
        raise SystemExit("❌ Data validation failed. See report.")


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="data/processed/validation_report.json")
    args = p.parse_args()
    main(args.train, args.test, args.out)
