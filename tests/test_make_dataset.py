import pandas as pd

from src.data.make_dataset import TARGET_COL, basic_clean, split_and_save


def test_basic_clean_renames_target(tmp_path):
    df = pd.DataFrame(
        {
            TARGET_COL: [0, 1, 0],
            "AGE": [20, 30, 40],
            "ID": [1, 1, 2],  # дубликат по ID
        }
    )
    cleaned = basic_clean(df)
    assert "target" in cleaned.columns
    assert TARGET_COL not in cleaned.columns
    assert cleaned["target"].isin([0, 1]).all()
    assert cleaned["AGE"].between(18, 100).all()
    assert cleaned["ID"].is_unique


def test_split_and_save_stratified(tmp_path):
    df = pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0] * 8 + [1, 1],
        }
    )
    split_and_save(df, tmp_path, test_size=0.3, seed=0)
    train = pd.read_csv(tmp_path / "train.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert len(train) == 7
    assert len(test) == 3
    assert train["target"].isin([0, 1]).all()
    assert test["target"].isin([0, 1]).all()
