
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore

def build_report(train_path: Path, test_path: Path) -> dict:
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    report = {}
    report["shapes"] = {"train": train.shape, "test": test.shape}
    report["target"] = {
        "train_mean": float(train["target"].mean()),
        "test_mean": float(test["target"].mean()),
        "train_counts": train["target"].value_counts(dropna=False).to_dict(),
        "test_counts":  test["target"].value_counts(dropna=False).to_dict(),
    }
    report["duplicates"] = {
        "train_rows_duplicated": int(train.duplicated().sum()),
        "test_rows_duplicated": int(test.duplicated().sum()),
        "train_id_duplicated": int(train["ID"].duplicated().sum()),
        "test_id_duplicated": int(test["ID"].duplicated().sum()),
    }
    inter_id = set(train["ID"]).intersection(set(test["ID"]))
    report["id_leakage_between_splits"] = {"count": len(inter_id), "examples": list(sorted(inter_id))[:5]}

    def nulls_consts(df):
        nulls = df.isna().mean().round(6).to_dict()
        const = {c:int(df[c].nunique()==1) for c in df.columns}
        return {"nulls_share": nulls, "const_flags": const}
    report["nulls_consts"] = {"train": nulls_consts(train), "test": nulls_consts(test)}

    bad = []
    def check_range(col, ge=None, le=None):
        s = pd.concat([train[col], test[col]])
        if ge is not None and (s < ge).any():
            bad.append({"column": col, "issue": f"<{ge}", "count": int((s<ge).sum())})
        if le is not None and (s > le).any():
            bad.append({"column": col, "issue": f">{le}", "count": int((s>le).sum())})
    for col in ["LIMIT_BAL"]: check_range(col, ge=1, le=1_000_000)
    for col in ["AGE"]: check_range(col, ge=18, le=100)
    for col in [f"PAY_{k}" for k in [0,2,3,4,5,6]]: check_range(col, ge=-2, le=9)
    for col in [f"BILL_AMT{k}" for k in range(1,7)]: check_range(col, ge=-1_000_000, le=10_000_000)
    for col in [f"PAY_AMT{k}" for k in range(1,7)]: check_range(col, ge=0, le=10_000_000)

    def unexpected_values(col, allowed):
        s = pd.concat([train[col], test[col]])
        diff = sorted(set(s.unique()) - set(allowed))
        if diff:
            bad.append({"column": col, "issue": "unexpected_categories", "values": diff})
    unexpected_values("SEX", [1,2])
    unexpected_values("EDUCATION", [0,1,2,3,4,5,6])
    unexpected_values("MARRIAGE", [0,1,2,3])
    report["range_category_issues"] = bad

    num_cols = train.drop(columns=["target"]).select_dtypes(include=["number"]).columns
    z = np.abs(train[num_cols].apply(lambda s: zscore(s, nan_policy="omit")))
    outliers = (z > 6).sum().sort_values(ascending=False)
    report["outliers_train_z>6_per_column"] = outliers[outliers>0].to_dict()

    drift_simple = {}
    for c in num_cols:
        mu_tr, mu_te = float(train[c].mean()), float(test[c].mean())
        rel = abs(mu_tr - mu_te) / (abs(mu_tr) + 1e-9)
        drift_simple[c] = {"mean_train": mu_tr, "mean_test": mu_te, "relative_diff": rel}
    report["mean_diff_num"] = {k:v for k,v in drift_simple.items() if v["relative_diff"]>0.2 and abs(v["mean_train"])>1e-9}

    fail_reasons = []
    if report["duplicates"]["train_rows_duplicated"]>0 or report["duplicates"]["test_rows_duplicated"]>0:
        fail_reasons.append("Row duplicates found")
    if report["duplicates"]["train_id_duplicated"]>0 or report["duplicates"]["test_id_duplicated"]>0:
        fail_reasons.append("ID duplicates found")
    if report["id_leakage_between_splits"]["count"]>0:
        fail_reasons.append("ID leakage between train/test")
    if any(v>0.1 for v in report["nulls_consts"]["train"]["nulls_share"].values()):
        fail_reasons.append("Too many NaNs (>10%) in train")
    if report["range_category_issues"]:
        fail_reasons.append("Range/category violations")
    report["fail_reasons"] = fail_reasons
    return report

def main(train_csv: str, test_csv: str, out_json: str):
    rep = build_report(Path(train_csv), Path(test_csv))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Data quality report -> {out_json}")
    if rep['fail_reasons']:
        raise SystemExit('❌ FAIL: ' + ', '.join(rep['fail_reasons']))
    print('✅ Data quality checks passed (basic).')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--test",  default="data/processed/test.csv")
    p.add_argument("--out",   default="data/processed/data_quality_report.json")
    args = p.parse_args()
    main(args.train, args.test, args.out)
