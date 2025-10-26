import argparse
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, Check


# Схема с проверками значений
schema = pa.DataFrameSchema(
    {
        "ID": Column(int, Check.ge(1)),
        "LIMIT_BAL": Column(float, Check.gt(0)),
        "SEX": Column(int, Check.isin([1, 2])),
        "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6])),
        "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3])),
        "AGE": Column(int, [Check.ge(18), Check.le(100)]),
        "PAY_0": Column(int, [Check.ge(-2), Check.le(9)]),
        "PAY_2": Column(int, [Check.ge(-2), Check.le(9)]),
        "PAY_3": Column(int, [Check.ge(-2), Check.le(9)]),
        "PAY_4": Column(int, [Check.ge(-2), Check.le(9)]),
        "PAY_5": Column(int, [Check.ge(-2), Check.le(9)]),
        "PAY_6": Column(int, [Check.ge(-2), Check.le(9)]),
        "BILL_AMT1": Column(float),
        "BILL_AMT2": Column(float),
        "BILL_AMT3": Column(float),
        "BILL_AMT4": Column(float),
        "BILL_AMT5": Column(float),
        "BILL_AMT6": Column(float),
        "PAY_AMT1": Column(float, Check.ge(0)),
        "PAY_AMT2": Column(float, Check.ge(0)),
        "PAY_AMT3": Column(float, Check.ge(0)),
        "PAY_AMT4": Column(float, Check.ge(0)),
        "PAY_AMT5": Column(float, Check.ge(0)),
        "PAY_AMT6": Column(float, Check.ge(0)),
        "target": Column(int, Check.isin([0, 1])),
    },
    strict=False,  # допускаем лишние колонки (если появятся)
)


def validate_file(path: Path) -> None:
    df = pd.read_csv(path)
    schema.validate(df, lazy=True)  # собираем все ошибки разом


def main(train_csv: str, test_csv: str, out_json: str | None = None) -> None:
    # базовая проверка: просто валидируем файлы, ошибки отдаст pandera
    validate_file(Path(train_csv))
    validate_file(Path(test_csv))
    # лёгкий отчёт (без ошибок = ок)
    if out_json:
        Path(out_json).write_text('{"ok": true}', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    main(args.train, args.test, args.out or None)
