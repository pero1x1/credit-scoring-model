import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "default.payment.next.month"


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Не найдена целевая колонка '{TARGET_COL}'")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Переименуем таргет в "target"
    df = df.rename(columns={TARGET_COL: "target"})

    # Таргет должен быть только 0/1
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("В целевой не только 0/1")

    # Возраст разумный
    if "AGE" in df.columns:
        df = df[(df["AGE"] >= 18) & (df["AGE"] <= 100)]

    # Уберём дубли строк
    df = df.drop_duplicates()

    return df


def split_and_save(
    df: pd.DataFrame,
    out_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    # train/test split с стратификацией по целевой
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train = X_train.copy()
    train["target"] = y_train.values

    test = X_test.copy()
    test["target"] = y_test.values

    # создаём директорию
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    # сохраняем
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # лог
    print(
        f"Saved train: {train_path} ({train.shape}), "
        f"test: {test_path} ({test.shape})"
    )


def main(raw_path: str, out_dir: str) -> None:
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)

    df = load_raw(raw_path)
    df = basic_clean(df)
    split_and_save(df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()

    main(args.raw, args.out)
