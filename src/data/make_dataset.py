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
    df = df.rename(columns={TARGET_COL: "target"})
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("В целевой не только 0/1")
    if "AGE" in df.columns:
        df = df[(df["AGE"] >= 18) & (df["AGE"] <= 100)]
    df = df.drop_duplicates()
    return df


def split_and_save(
    df: pd.DataFrame, out_dir: Path, test_size: float = 0.2, seed: int = 42
):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train = X_train.copy()
    train["target"] = y_train.values
    test = X_test.copy()
    test["target"] = y_test.values
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(
        f"Saved: {out_dir/'train.csv'} ({train.shape}), {out_dir/'test.csv'} ({test.shape})"
    )


def main(raw_path: str, out_dir: str):
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
