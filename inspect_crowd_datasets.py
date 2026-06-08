#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit crowdsourcing datasets converted to data/<dataset>/annotations.csv.

Expected columns:
    item, worker, label, truth

This script studies:
- number of items/workers/votes/classes
- annotation matrix density
- votes per item and items per worker
- majority-vote accuracy
- coverage/accuracy of top-B workers by accuracy
- coverage/accuracy of greedy max-coverage workers
- pairwise worker overlap

Usage:
    python inspect_crowd_datasets.py --data_root ./data --out_dir ./dataset_audit

Selected datasets:
    python inspect_crowd_datasets.py --datasets bird dog rte trec web CF MS SP
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Optional, Sequence, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DATASETS = [
    "CF", "CF_amt", "MS", "SP", "SP_amt", "ZCall", "ZCin", "ZCus",
    "bird", "dog", "rte", "trec", "web", "prod", "senti", "face", "adult",
]

ALIASES = {"CF*": "CF_amt", "SP*": "SP_amt"}


def resolve(name: str) -> str:
    return ALIASES.get(name, name)


def load_dataset(data_root: Path, name: str) -> Optional[pd.DataFrame]:
    name = resolve(name)
    path = data_root / name / "annotations.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    needed = {"item", "worker", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {needed}; got {df.columns.tolist()}")
    for c in ["item", "worker", "label"]:
        df[c] = df[c].astype(str)
    if "truth" in df.columns:
        df["truth"] = df["truth"].astype(str)
        df.loc[df["truth"].str.lower().isin(["nan", "none", ""]), "truth"] = np.nan
    df["dataset"] = name
    return df


def majority_vote(labels: Sequence[str]) -> Optional[str]:
    labels = [str(x) for x in labels if pd.notna(x)]
    if not labels:
        return None
    vc = pd.Series(labels).value_counts()
    top = vc.max()
    ties = sorted(vc[vc == top].index.astype(str).tolist())
    return ties[0]


def majority_vote_accuracy(df: pd.DataFrame) -> float:
    if "truth" not in df.columns or df["truth"].isna().all():
        return np.nan
    ok = []
    for _, g in df.groupby("item"):
        truth = g["truth"].dropna()
        if len(truth) == 0:
            continue
        pred = majority_vote(g["label"].tolist())
        if pred is not None:
            ok.append(pred == str(truth.iloc[0]))
    return float(np.mean(ok)) if ok else np.nan


def worker_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if "truth" not in df.columns or df["truth"].isna().all():
        out = df.groupby("worker").size().reset_index(name="n_votes")
        out["accuracy"] = np.nan
        return out
    tmp = df.dropna(subset=["truth"]).copy()
    tmp["correct"] = (tmp["label"].astype(str) == tmp["truth"].astype(str)).astype(float)
    return tmp.groupby("worker").agg(
        n_votes=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()


def predict_selected(df: pd.DataFrame, workers: Sequence[str]) -> pd.DataFrame:
    W = set(map(str, workers))
    rows = []
    for item, g in df.groupby("item"):
        gs = g[g["worker"].isin(W)]
        pred = majority_vote(gs["label"].tolist()) if len(gs) else None
        truth = None
        if "truth" in g.columns and g["truth"].notna().any():
            truth = str(g["truth"].dropna().iloc[0])
        rows.append({"item": item, "prediction": pred, "truth": truth, "covered": pred is not None})
    return pd.DataFrame(rows)


def eval_selected(df: pd.DataFrame, workers: Sequence[str]) -> Dict[str, float]:
    pred = predict_selected(df, workers)
    coverage = float(pred["covered"].mean()) if len(pred) else np.nan
    covered = pred[pred["covered"]]
    if len(covered) == 0 or "truth" not in df.columns or df["truth"].isna().all():
        return {"coverage": coverage, "accuracy_on_covered": np.nan, "n_covered_items": int(len(covered))}
    acc = (covered["prediction"].astype(str) == covered["truth"].astype(str)).mean()
    return {"coverage": coverage, "accuracy_on_covered": float(acc), "n_covered_items": int(len(covered))}


def top_accuracy_workers(df: pd.DataFrame, B: int) -> List[str]:
    acc = worker_accuracy(df)
    if acc["accuracy"].notna().any():
        acc = acc.sort_values(["accuracy", "n_votes"], ascending=[False, False])
    else:
        acc = acc.sort_values("n_votes", ascending=False)
    return acc["worker"].astype(str).head(B).tolist()


def greedy_max_coverage_workers(df: pd.DataFrame, B: int) -> List[str]:
    items_by_worker = {w: set(g["item"].unique()) for w, g in df.groupby("worker")}
    chosen, covered = [], set()
    for _ in range(min(B, len(items_by_worker))):
        best, gain = None, -1
        for w, items in items_by_worker.items():
            if w in chosen:
                continue
            g = len(items - covered)
            if g > gain:
                best, gain = w, g
        if best is None:
            break
        chosen.append(best)
        covered |= items_by_worker[best]
    return chosen


def pairwise_overlap_summary(df: pd.DataFrame) -> Dict[str, float]:
    items_by_worker = {w: set(g["item"].unique()) for w, g in df.groupby("worker")}
    workers = sorted(items_by_worker)
    overlaps = []
    jaccards = []
    for a, b in itertools.combinations(workers, 2):
        ia, ib = items_by_worker[a], items_by_worker[b]
        inter = len(ia & ib)
        union = len(ia | ib)
        overlaps.append(inter)
        jaccards.append(inter / union if union else 0.0)
    if not overlaps:
        return {
            "pair_overlap_mean": np.nan,
            "pair_overlap_median": np.nan,
            "pair_overlap_max": 0,
            "share_pairs_overlap_ge_1": np.nan,
            "share_pairs_overlap_ge_5": np.nan,
            "jaccard_mean": np.nan,
        }
    ov = np.array(overlaps)
    ja = np.array(jaccards)
    return {
        "pair_overlap_mean": float(np.mean(ov)),
        "pair_overlap_median": float(np.median(ov)),
        "pair_overlap_max": int(np.max(ov)),
        "share_pairs_overlap_ge_1": float(np.mean(ov >= 1)),
        "share_pairs_overlap_ge_5": float(np.mean(ov >= 5)),
        "jaccard_mean": float(np.mean(ja)),
    }


def save_hist(values, title, xlabel, path: Path):
    vals = pd.Series(values).dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=min(40, max(5, len(np.unique(vals)))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def inspect_dataset(df: pd.DataFrame, name: str, budgets: Sequence[int], out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_items = df["item"].nunique()
    n_workers = df["worker"].nunique()
    n_votes = len(df.drop_duplicates(["item", "worker"]))
    density = n_votes / (n_items * n_workers) if n_items and n_workers else np.nan

    votes_per_item = df.groupby("item")["worker"].nunique()
    items_per_worker = df.groupby("worker")["item"].nunique()

    summary = {
        "dataset": name,
        "n_items": int(n_items),
        "n_workers": int(n_workers),
        "n_votes": int(n_votes),
        "n_labels": int(df["label"].nunique()),
        "density": float(density),
        "votes_per_item_mean": float(votes_per_item.mean()),
        "votes_per_item_median": float(votes_per_item.median()),
        "votes_per_item_min": int(votes_per_item.min()),
        "votes_per_item_max": int(votes_per_item.max()),
        "items_per_worker_mean": float(items_per_worker.mean()),
        "items_per_worker_median": float(items_per_worker.median()),
        "items_per_worker_min": int(items_per_worker.min()),
        "items_per_worker_max": int(items_per_worker.max()),
        "majority_vote_accuracy": majority_vote_accuracy(df),
    }
    summary.update(pairwise_overlap_summary(df))

    save_hist(votes_per_item, f"{name}: votes per item", "Votes per item", plot_dir / f"votes_per_item_{name}.png")
    save_hist(items_per_worker, f"{name}: items per worker", "Items per worker", plot_dir / f"items_per_worker_{name}.png")

    budget_rows = []
    for B0 in budgets:
        B = min(int(B0), n_workers)
        if B <= 0:
            continue
        for strategy, workers in [
            ("top_accuracy", top_accuracy_workers(df, B)),
            ("max_coverage_greedy", greedy_max_coverage_workers(df, B)),
        ]:
            ev = eval_selected(df, workers)
            budget_rows.append({
                "dataset": name,
                "budget": B,
                "strategy": strategy,
                "selected_workers": "|".join(workers),
                **ev,
            })

    bdf = pd.DataFrame(budget_rows)
    if not bdf.empty:
        plt.figure(figsize=(8, 4))
        for strategy, g in bdf.groupby("strategy"):
            g = g.sort_values("budget")
            plt.plot(g["budget"], g["coverage"], marker="o", label=strategy)
        plt.title(f"{name}: coverage vs budget")
        plt.xlabel("Budget")
        plt.ylabel("Coverage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"coverage_vs_budget_{name}.png", dpi=200)
        plt.close()

    return summary, budget_rows


def global_plots(summary: pd.DataFrame, budget: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if len(summary):
        for col, title, fname in [
            ("density", "Annotation matrix density", "summary_density.png"),
            ("votes_per_item_mean", "Mean votes per item", "summary_votes_per_item.png"),
            ("items_per_worker_mean", "Mean items per worker", "summary_items_per_worker.png"),
        ]:
            plt.figure(figsize=(11, 5))
            s = summary.sort_values(col)
            plt.bar(s["dataset"], s[col])
            plt.xticks(rotation=60, ha="right")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(plot_dir / fname, dpi=200)
            plt.close()

    if len(budget):
        maxB = budget["budget"].max()
        b = budget[budget["budget"] == maxB]
        plt.figure(figsize=(11, 5))
        for strategy, g in b.groupby("strategy"):
            g = g.sort_values("dataset")
            plt.plot(g["dataset"], g["coverage"], marker="o", linestyle="", label=strategy)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Coverage")
        plt.title(f"Coverage at budget={maxB}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "summary_coverage_at_max_budget.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="./dataset_audit")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--budgets", nargs="*", type=int, default=[1, 3, 5, 10, 20, 50])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    budget_rows = []

    for raw in args.datasets:
        name = resolve(raw)
        df = load_dataset(data_root, name)
        if df is None:
            print(f"[skip] {name}: missing {data_root / name / 'annotations.csv'}")
            continue
        print(f"[inspect] {name}: {df.shape}")
        s, rows = inspect_dataset(df, name, args.budgets, out_dir)
        summaries.append(s)
        budget_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    budget_df = pd.DataFrame(budget_rows)

    summary_df.to_csv(out_dir / "dataset_summary.csv", index=False)
    budget_df.to_csv(out_dir / "budget_coverage_summary.csv", index=False)
    global_plots(summary_df, budget_df, out_dir)

    print("\nSaved:")
    print(out_dir / "dataset_summary.csv")
    print(out_dir / "budget_coverage_summary.csv")
    print(out_dir / "plots")

    if len(summary_df):
        cols = [
            "dataset", "n_items", "n_workers", "n_votes", "density",
            "votes_per_item_mean", "items_per_worker_mean",
            "majority_vote_accuracy",
            "pair_overlap_mean", "share_pairs_overlap_ge_5",
        ]
        print("\nSummary:")
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
