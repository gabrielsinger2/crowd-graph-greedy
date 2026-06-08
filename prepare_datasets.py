from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "data"

SPECTRAL = {
    "bird": ("bluebird_crowd.txt", "bluebird_truth.txt"),
    "dog": ("dog_crowd.txt", "dog_truth.txt"),
    "rte": ("rte_crowd.txt", "rte_truth.txt"),
    "trec": ("trec_crowd.txt", "trec_truth.txt"),
    "web": ("web_crowd.txt", "web_truth.txt"),
}

ACT = {
    "CF": "CF.csv",
    "CF_amt": "CF_amt.csv",
    "MS": "MS.csv",
    "SP": "SP.csv",
    "SP_amt": "SP_amt.csv",
    "ZCall": "ZenCrowd_all.csv",
    "ZCin": "ZenCrowd_in.csv",
    "ZCus": "ZenCrowd_us.csv",
}


def prepare_spectral():
    src = ROOT / "external_data" / "SpectralMethodsMeetEM" / "src"

    for name, (crowd_file, truth_file) in SPECTRAL.items():
        crowd_path = src / crowd_file
        truth_path = src / truth_file

        if not crowd_path.exists():
            print(f"[skip spectral] {name}: missing {crowd_path}")
            continue

        crowd = pd.read_csv(crowd_path, sep=r"\s+", header=None, engine="python").iloc[:, :3]
        crowd.columns = ["item", "worker", "label"]

        truth = pd.read_csv(truth_path, sep=r"\s+", header=None, engine="python").iloc[:, :2]
        truth.columns = ["item", "truth"]

        out = crowd.merge(truth, on="item", how="left")

        out_dir = DATA / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "annotations.csv", index=False)

        print(f"[ok spectral] {name}: {out.shape}")


def prepare_active_crowd_toolkit():
    src = ROOT / "external_data" / "active-crowd-toolkit" / "Data"

    for name, filename in ACT.items():
        path = src / filename

        if not path.exists():
            print(f"[skip act] {name}: missing {path}")
            continue

        # Active Crowd Toolkit files are headerless:
        # WorkerID, TaskID, WorkerLabel, GoldLabel
        raw = pd.read_csv(path, header=None)

        if raw.shape[1] < 3:
            print(f"[skip act] {name}: not enough columns")
            continue

        out = pd.DataFrame({
            "worker": raw.iloc[:, 0],
            "item": raw.iloc[:, 1],
            "label": raw.iloc[:, 2],
        })

        if raw.shape[1] >= 4:
            out["truth"] = raw.iloc[:, 3]

        out_dir = DATA / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "annotations.csv", index=False)

        print(f"[ok act] {name}: {out.shape}")


if __name__ == "__main__":
    prepare_spectral()
    prepare_active_crowd_toolkit()
