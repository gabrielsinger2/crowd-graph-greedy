from pathlib import Path
import pandas as pd

DATA = Path("data")

SPECTRAL = ["bird", "dog", "rte", "trec", "web"]

def read_txt_no_header(path):
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")

def prepare_spectral_dataset(name):
    folder = DATA / name
    ann_path = folder / "annotations.txt"
    truth_path = folder / "truth.txt"

    if not ann_path.exists():
        print(f"[skip] {name}: missing {ann_path}")
        return

    ann = read_txt_no_header(ann_path)
    print(f"[{name}] annotations shape:", ann.shape)

    if ann.shape[1] < 3:
        raise ValueError(f"{ann_path} should have at least 3 columns: item worker label")

    ann = ann.iloc[:, :3].copy()
    ann.columns = ["item", "worker", "label"]

    if truth_path.exists():
        truth = read_txt_no_header(truth_path)
        print(f"[{name}] truth shape:", truth.shape)

        if truth.shape[1] < 2:
            raise ValueError(f"{truth_path} should have at least 2 columns: item truth")

        truth = truth.iloc[:, :2].copy()
        truth.columns = ["item", "truth"]

        out = ann.merge(truth, on="item", how="left")
    else:
        print(f"[warning] {name}: no truth file found")
        out = ann

    out_path = folder / "annotations.csv"
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")

if __name__ == "__main__":
    for name in SPECTRAL:
        prepare_spectral_dataset(name)