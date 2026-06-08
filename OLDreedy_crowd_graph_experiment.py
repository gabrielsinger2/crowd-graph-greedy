#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy dependence-aware crowd selection on standard crowdsourcing datasets.

This file is meant as a first proof-of-concept for the following pipeline:

    dataset -> worker error rates -> naive error-correlation graph
            -> graph-dependent surrogate
            -> greedy subset selection
            -> evaluation + plots

It does NOT rerun the baseline methods from the paper. If you already have
paper results, you can pass them as a CSV and the script will merge/plot them.

================================================================================
Expected input format
================================================================================

The script tries to be permissive. For each dataset, it searches under:

    data_root/<dataset_name>/

and tries to find either:

1) A single annotation file containing item, worker, label, and possibly truth.
   Accepted column aliases include:
        item:   item, item_id, task, task_id, object, object_id, example
        worker: worker, worker_id, annotator, annotator_id, user, user_id
        label:  label, answer, annotation, vote, response
        truth:  truth, true_label, ground_truth, gold, y

2) Separate annotation and truth files.
   It tries to merge them using the inferred item column.

If no truth column is found, the script uses majority vote as pseudo-truth.
That is useful for debugging, but it is not a proper evaluation protocol.

================================================================================
Dataset registry
================================================================================

The registry below includes the 17 real-world datasets described in the paper:
- Venanzi et al. (2015): CF, CF*, MS, SP, SP*, ZCall, ZCin, ZCus
- Zheng et al. (2017): prod, senti, face, adult, dog, plus overlapping datasets
- Zhang et al. (2014): bird, rte, trec, web, dog, with overlap across collections

Depending on the repository you downloaded from, names may differ:
    ZCall / ZC_all, ZCin / ZC_in, ZCus / ZC_us,
    CF* / CF_amt, SP* / SP_amt, senti / tweet.

The script includes aliases for these names.

================================================================================
Important modeling note
================================================================================

The theoretical majority-vote risk in the binary correctness formulation is

    R_phi(S) = P( sum_{i in S} E_i >= ceil(|S|/2) ),

where E_i = 1{worker i is wrong}.

For multi-class datasets, the actual prediction is a plurality vote among the
selected workers. The binary tail surrogate remains a conservative proxy:
if strictly more than half of selected workers are correct, plurality is correct,
but plurality can be correct even with fewer than half correct.

================================================================================
Example usage
================================================================================

    python greedy_crowd_graph_experiment.py \
        --data_root ./data \
        --out_dir ./results_greedy \
        --budget 5 \
        --corr_threshold 0.10 \
        --min_pair_overlap 5 \
        --fixed_budget

You can also specify a budget as a percentage of the available workers in each
dataset. For example, to select 50% of workers per dataset:

    python greedy_crowd_graph_experiment.py \
        --data_root ./data \
        --out_dir ./results_greedy \
        --budget_percent 50 \
        --fixed_budget

To run only a subset of datasets:

    python greedy_crowd_graph_experiment.py --datasets CF SP adult bird

To merge paper baselines already copied into a CSV:

    python greedy_crowd_graph_experiment.py \
        --paper_baselines_csv paper_results.csv

Expected paper_results.csv columns:
    dataset, method, accuracy

================================================================================
Outputs
================================================================================

out_dir/
    per_dataset_results.csv
    selected_workers.json
    dataset_registry.csv
    plot_accuracy_by_dataset.png
    plot_accuracy_vs_budget.png
    plot_surrogate_vs_budget.png
    plot_graph_density.png

Dependencies:
    numpy, pandas, matplotlib

Optional:
    networkx is not required.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Dataset registry
# =============================================================================

@dataclass(frozen=True)
class DatasetInfo:
    canonical_name: str
    aliases: Tuple[str, ...]
    collections: Tuple[str, ...]
    task: str
    original_reference: str


DATASETS: List[DatasetInfo] = [
    DatasetInfo("CF", ("CF", "weather", "crowdflower_weather"), ("Venanzi2015",),
                "Sentiment analysis for tweets about weather.",
                "Josephy et al. 2014 / Venanzi et al. 2015"),
    DatasetInfo("CF*", ("CF*", "CF_amt", "CF-star", "CF_reannotated"), ("Venanzi2015",),
                "Re-annotated version of CF.",
                "Venanzi et al. 2015"),
    DatasetInfo("MS", ("MS", "music", "music_genre"), ("Venanzi2015",),
                "Music genre classification from 30-second music samples.",
                "Rodrigues et al. 2013 / Venanzi et al. 2015"),
    DatasetInfo("SP", ("SP", "sentiment_polarity", "movie_reviews"), ("Venanzi2015",),
                "Sentiment analysis for movie reviews.",
                "Rodrigues et al. 2013 / Venanzi et al. 2015"),
    DatasetInfo("SP*", ("SP*", "SP_amt", "SP-star", "SP_reannotated"), ("Venanzi2015",),
                "Re-annotated version of SP.",
                "Venanzi et al. 2015"),
    DatasetInfo("ZCall", ("ZCall", "ZC_all", "ZCall_all", "zc_all"), ("Venanzi2015",),
                "URI relevance to named entity from news.",
                "Demartini et al. 2012 / Venanzi et al. 2015"),
    DatasetInfo("ZCin", ("ZCin", "ZC_in", "zc_in"), ("Venanzi2015",),
                "URI relevance to named entity from news.",
                "Demartini et al. 2012 / Venanzi et al. 2015"),
    DatasetInfo("ZCus", ("ZCus", "ZC_us", "zc_us"), ("Venanzi2015",),
                "URI relevance to named entity from news.",
                "Demartini et al. 2012 / Venanzi et al. 2015"),
    DatasetInfo("prod", ("prod", "product", "products"), ("Zheng2017",),
                "Entity resolution: whether two product descriptions refer to the same product.",
                "Wang et al. 2012 / Zheng et al. 2017"),
    DatasetInfo("senti", ("senti", "tweet", "tweets", "sentiment"), ("Zheng2017",),
                "Sentiment analysis for companies mentioned in tweets.",
                "Zheng et al. 2017"),
    DatasetInfo("face", ("face", "facial_expression"), ("Zheng2017",),
                "Facial expression classification.",
                "Mozafari et al. 2014 / Zheng et al. 2017"),
    DatasetInfo("adult", ("adult", "website_adult", "age_appropriateness"), ("Zheng2017",),
                "Age-appropriateness rating of websites.",
                "Mason & Suri 2012 / Zheng et al. 2017"),
    DatasetInfo("bird", ("bird", "duck", "duck_bird"), ("Zhang2014",),
                "Whether an image contains at least one duck bird.",
                "Welinder et al. 2010 / Zhang et al. 2014"),
    DatasetInfo("dog", ("dog", "dogs", "dog_breed"), ("Zheng2017", "Zhang2014"),
                "Dog breed labeling.",
                "Zhang et al. 2014 / Zheng et al. 2017"),
    DatasetInfo("rte", ("rte", "textual_entailment"), ("Zhang2014",),
                "Recognising textual entailment.",
                "Snow et al. 2008 / Zhang et al. 2014"),
    DatasetInfo("trec", ("trec", "TREC"), ("Zhang2014",),
                "Quality/relevance of retrieved documents.",
                "TREC 2011 crowdsourcing track / Zhang et al. 2014"),
    DatasetInfo("web", ("web", "web_search"), ("Zhang2014",),
                "Relevance of web search results.",
                "Zhou et al. 2012 / Zhang et al. 2014"),
]


# =============================================================================
# Column inference and loading
# =============================================================================

ITEM_CANDIDATES = [
    "item", "item_id", "task", "task_id", "object", "object_id", "example",
    "example_id", "question", "question_id", "url", "doc", "doc_id", "record"
]
WORKER_CANDIDATES = [
    "worker", "worker_id", "annotator", "annotator_id", "user", "user_id",
    "labeler", "labeler_id", "rater", "rater_id", "source", "source_id"
]
LABEL_CANDIDATES = [
    "label", "answer", "annotation", "vote", "response", "worker_label",
    "observed_label", "class", "category"
]
TRUTH_CANDIDATES = [
    "truth", "true_label", "ground_truth", "gold", "gold_label", "y",
    "target", "gt", "correct_label"
]


def normalize_colname(c: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def infer_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    norm_to_orig = {normalize_colname(c): c for c in df.columns}
    candidate_norm = [normalize_colname(c) for c in candidates]
    for c in candidate_norm:
        if c in norm_to_orig:
            return norm_to_orig[c]
    for c_norm, orig in norm_to_orig.items():
        for cand in candidate_norm:
            if cand in c_norm or c_norm in cand:
                return orig
    return None


def read_table_any(path: Path) -> Optional[pd.DataFrame]:
    if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".dat"}:
        return None

    # Try common separators with headers.
    for sep in [None, ",", "\t", ";", " "]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2 and len(df) > 0:
                return df
        except Exception:
            pass

    # Try no-header fallback.
    for sep in [None, ",", "\t", ";", " "]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", header=None)
            if df.shape[1] >= 2 and len(df) > 0:
                return df
        except Exception:
            pass

    return None


def standardize_single_file(df: pd.DataFrame, source_file: str) -> Optional[pd.DataFrame]:
    """
    Return dataframe with columns: item, worker, label, truth? if possible.
    """
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    item_col = infer_column(df, ITEM_CANDIDATES)
    worker_col = infer_column(df, WORKER_CANDIDATES)
    label_col = infer_column(df, LABEL_CANDIDATES)
    truth_col = infer_column(df, TRUTH_CANDIDATES)

    if item_col and worker_col and label_col:
        cols = {"item": item_col, "worker": worker_col, "label": label_col}
        out = pd.DataFrame({
            "item": df[cols["item"]],
            "worker": df[cols["worker"]],
            "label": df[cols["label"]],
        })
        if truth_col is not None and truth_col != label_col:
            out["truth"] = df[truth_col]
        out["source_file"] = source_file
        return out.dropna(subset=["item", "worker", "label"])

    # No-header fallback: common annotation format is item, worker, label.
    # If there are exactly 3 columns, assume item-worker-label.
    # If there are 4+ columns, assume item-worker-label-truth.
    if all(isinstance(c, int) for c in df.columns) or all(str(c).isdigit() for c in df.columns):
        if df.shape[1] >= 3:
            out = pd.DataFrame({
                "item": df.iloc[:, 0],
                "worker": df.iloc[:, 1],
                "label": df.iloc[:, 2],
            })
            if df.shape[1] >= 4:
                out["truth"] = df.iloc[:, 3]
            out["source_file"] = source_file
            return out.dropna(subset=["item", "worker", "label"])

    return None


def looks_like_truth_file(path: Path, df: pd.DataFrame) -> bool:
    name = path.name.lower()
    if any(k in name for k in ["truth", "gold", "gt", "ground"]):
        return True
    truth_col = infer_column(df, TRUTH_CANDIDATES)
    item_col = infer_column(df, ITEM_CANDIDATES)
    worker_col = infer_column(df, WORKER_CANDIDATES)
    return item_col is not None and truth_col is not None and worker_col is None


def standardize_truth_file(df: pd.DataFrame, source_file: str) -> Optional[pd.DataFrame]:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    item_col = infer_column(df, ITEM_CANDIDATES)
    truth_col = infer_column(df, TRUTH_CANDIDATES)

    if item_col and truth_col:
        return pd.DataFrame({"item": df[item_col], "truth": df[truth_col], "truth_file": source_file}).dropna()

    # no-header fallback: item, truth
    if df.shape[1] >= 2:
        return pd.DataFrame({"item": df.iloc[:, 0], "truth": df.iloc[:, 1], "truth_file": source_file}).dropna()

    return None


def find_dataset_dir(data_root: Path, info: DatasetInfo) -> Optional[Path]:
    if not data_root.exists():
        return None
    candidates = [info.canonical_name] + list(info.aliases)
    for alias in candidates:
        direct = data_root / alias
        if direct.exists() and direct.is_dir():
            return direct
    # Recursive loose search.
    lower_aliases = {a.lower().replace("*", "").replace("_", "") for a in candidates}
    for p in data_root.rglob("*"):
        if p.is_dir():
            key = p.name.lower().replace("*", "").replace("_", "")
            if key in lower_aliases:
                return p
    return None


def load_dataset(data_root: Path, info: DatasetInfo) -> Optional[pd.DataFrame]:
    ddir = find_dataset_dir(data_root, info)
    if ddir is None:
        return None

    tables: List[pd.DataFrame] = []
    truth_tables: List[pd.DataFrame] = []

    for path in sorted(ddir.rglob("*")):
        df = read_table_any(path)
        if df is None:
            continue

        if looks_like_truth_file(path, df):
            tdf = standardize_truth_file(df, str(path.relative_to(data_root)))
            if tdf is not None:
                truth_tables.append(tdf)

        sdf = standardize_single_file(df, str(path.relative_to(data_root)))
        if sdf is not None:
            # Exclude obvious truth-only files mistakenly read as single-file annotations.
            if not ("worker" in sdf and sdf["worker"].nunique(dropna=True) <= 1 and "truth" in sdf):
                tables.append(sdf)

    if not tables:
        return None

    # Choose the largest candidate annotation table.
    ann = max(tables, key=len).copy()
    ann["dataset"] = info.canonical_name

    if "truth" not in ann.columns and truth_tables:
        truth = max(truth_tables, key=len).drop_duplicates("item")
        ann = ann.merge(truth[["item", "truth"]], on="item", how="left")

    # Clean types as strings to avoid accidental numeric/categorical mismatch.
    for c in ["item", "worker", "label"]:
        ann[c] = ann[c].astype(str)
    if "truth" in ann.columns:
        ann["truth"] = ann["truth"].astype(str)

    return ann


# =============================================================================
# Basic aggregation utilities
# =============================================================================

def majority_vote(labels: Sequence[object], tie_break_order: Optional[List[object]] = None) -> Optional[object]:
    labels = [x for x in labels if pd.notna(x)]
    if len(labels) == 0:
        return None
    counts = pd.Series(labels).value_counts()
    top_count = counts.iloc[0]
    ties = list(counts[counts == top_count].index)
    if len(ties) == 1:
        return ties[0]
    if tie_break_order:
        for y in tie_break_order:
            if y in ties:
                return y
    return sorted(map(str, ties))[0]


def add_pseudo_truth_if_needed(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df.copy()
    if "truth" in df.columns and df["truth"].notna().any():
        return df, False

    warnings.warn(
        "No ground truth found. Using majority vote as pseudo-truth. "
        "This is only for debugging, not proper evaluation."
    )
    mv = df.groupby("item")["label"].apply(lambda x: majority_vote(list(x))).reset_index()
    mv = mv.rename(columns={"label": "truth"})
    df = df.merge(mv, on="item", how="left")
    return df, True


def train_test_split_items(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    items = np.array(sorted(df["item"].unique()))
    rng.shuffle(items)
    n_test = max(1, int(round(len(items) * test_frac)))
    test_items = set(items[:n_test])
    train = df[~df["item"].isin(test_items)].copy()
    test = df[df["item"].isin(test_items)].copy()
    return train, test


def estimate_worker_error_rates(train: pd.DataFrame, alpha: float = 1.0) -> Dict[str, float]:
    train = train.dropna(subset=["truth"])
    q = {}
    for worker, g in train.groupby("worker"):
        n = len(g)
        err = (g["label"].astype(str) != g["truth"].astype(str)).sum()
        q[worker] = float((err + alpha) / (n + 2.0 * alpha))
    return q


def estimate_error_matrix(train: pd.DataFrame) -> pd.DataFrame:
    train = train.dropna(subset=["truth"]).copy()
    train["error"] = (train["label"].astype(str) != train["truth"].astype(str)).astype(float)
    err = train.pivot_table(index="item", columns="worker", values="error", aggfunc="mean")
    return err


def estimate_error_correlation_graph(
    train: pd.DataFrame,
    corr_threshold: float = 0.10,
    min_pair_overlap: int = 5,
    positive_only: bool = True,
) -> Tuple[Set[Tuple[str, str]], pd.DataFrame]:
    """
    Naive graph:
        edge (i,j) if empirical error correlation > threshold.

    Returns:
        edges: set of sorted worker pairs
        corr_df: pairwise correlation table
    """
    err = estimate_error_matrix(train)
    workers = list(err.columns)
    edges: Set[Tuple[str, str]] = set()
    rows = []

    for a, b in itertools.combinations(workers, 2):
        pair = err[[a, b]].dropna()
        overlap = len(pair)
        if overlap < min_pair_overlap:
            continue

        xa, xb = pair[a].values, pair[b].values
        if np.std(xa) == 0 or np.std(xb) == 0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(xa, xb)[0, 1])
            if np.isnan(rho):
                rho = 0.0

        rows.append({"worker_i": a, "worker_j": b, "rho_error": rho, "overlap": overlap})

        score = rho if positive_only else abs(rho)
        if score > corr_threshold:
            edges.add(tuple(sorted((a, b))))

    corr_df = pd.DataFrame(rows)
    return edges, corr_df


def graph_density(nodes: Sequence[str], edges: Set[Tuple[str, str]]) -> float:
    nodes = list(nodes)
    if len(nodes) <= 1:
        return 0.0
    node_set = set(nodes)
    m = sum(1 for a, b in edges if a in node_set and b in node_set)
    return 2.0 * m / (len(nodes) * (len(nodes) - 1))


def resolve_worker_budget(
    n_workers: int,
    budget: int,
    budget_percent: Optional[float] = None,
    min_budget: int = 1,
) -> int:
    """
    Resolve the selected-worker budget for a dataset.

    If budget_percent is provided, the budget is computed as

        ceil(n_workers * budget_percent / 100).

    Examples:
        n_workers=100, budget_percent=50  -> 50
        n_workers=13,  budget_percent=50  -> 7
        n_workers=13,  budget_percent=10  -> 2

    The result is clipped to [min_budget, n_workers].
    """
    if n_workers <= 0:
        return 0

    if budget_percent is None:
        raw_budget = int(budget)
    else:
        if budget_percent <= 0:
            raise ValueError("--budget_percent must be strictly positive.")
        if budget_percent > 100:
            raise ValueError("--budget_percent must be <= 100.")
        raw_budget = int(math.ceil(n_workers * float(budget_percent) / 100.0))

    return int(max(min_budget, min(raw_budget, n_workers)))


# =============================================================================
# Graph-dependent surrogate
# =============================================================================

def is_independent_subset(subset: Sequence[str], edges: Set[Tuple[str, str]]) -> bool:
    for a, b in itertools.combinations(subset, 2):
        if tuple(sorted((a, b))) in edges:
            return False
    return True


def max_independent_set_exact(nodes: Sequence[str], edges: Set[Tuple[str, str]], max_bruteforce_nodes: int = 22) -> List[str]:
    """
    Exact maximum independent set by brute force for small induced graphs.
    For larger sets, falls back to greedy.
    """
    nodes = list(nodes)
    if len(nodes) == 0:
        return []
    if len(nodes) > max_bruteforce_nodes:
        return max_independent_set_greedy(nodes, edges)

    # Try largest subsets first.
    for k in range(len(nodes), 0, -1):
        for subset in itertools.combinations(nodes, k):
            if is_independent_subset(subset, edges):
                return list(subset)
    return []


def max_independent_set_greedy(nodes: Sequence[str], edges: Set[Tuple[str, str]]) -> List[str]:
    """
    Simple greedy independent set: sort by degree ascending, add if possible.
    """
    nodes = list(nodes)
    node_set = set(nodes)
    deg = {v: 0 for v in nodes}
    for a, b in edges:
        if a in node_set and b in node_set:
            deg[a] += 1
            deg[b] += 1

    chosen: List[str] = []
    for v in sorted(nodes, key=lambda u: (deg[u], str(u))):
        if all(tuple(sorted((v, u))) not in edges for u in chosen):
            chosen.append(v)
    return chosen


def elementary_symmetric_sums(values: Sequence[float]) -> List[float]:
    """
    e[k] = sum over subsets of size k of product values.
    """
    e = [1.0]
    for v in values:
        e.append(0.0)
        for k in range(len(e) - 1, 0, -1):
            e[k] += e[k - 1] * float(v)
    return e


def graph_surrogate(
    S: Sequence[str],
    q: Dict[str, float],
    edges: Set[Tuple[str, str]],
    max_bruteforce_nodes: int = 22,
) -> float:
    """
    Proposed graph-dependent surrogate:

        B_G(S) = sum_{j=t_S}^{|S|} sum_{A in partial_j S}
                 prod_{i in A cap V_alpha(S)} q_i.

    This is evaluated combinatorially without enumerating all A.

    Let I = V_alpha(S), a = |I|, C = S \ I, c = |C|.
    For fixed j:
        sum_{A: |A|=j} prod_{i in A cap I} q_i
      = sum_l C(c, j-l) e_l(q_I),
    where e_l is the elementary symmetric polynomial of degree l.
    """
    S = list(dict.fromkeys(S))
    B = len(S)
    if B == 0:
        return float("inf")

    tS = math.ceil(B / 2)
    I = max_independent_set_exact(S, edges, max_bruteforce_nodes=max_bruteforce_nodes)
    I_set = set(I)
    c = B - len(I)

    qI = [float(q.get(i, 0.5)) for i in I]
    e = elementary_symmetric_sums(qI)

    total = 0.0
    for j in range(tS, B + 1):
        for ell in range(0, min(len(I), j) + 1):
            if j - ell <= c:
                total += math.comb(c, j - ell) * e[ell]
    return float(total)


def simple_accuracy_diversity_surrogate(
    S: Sequence[str],
    q: Dict[str, float],
    corr_df: Optional[pd.DataFrame],
    lambda_corr: float = 1.0,
) -> float:
    """
    Easier baseline surrogate:
        mean error + lambda * sum positive error correlations inside S.
    """
    S = list(dict.fromkeys(S))
    if len(S) == 0:
        return float("inf")
    score = float(np.mean([q.get(i, 0.5) for i in S]))

    if corr_df is not None and len(corr_df) > 0:
        Sset = set(S)
        penalty = 0.0
        for _, row in corr_df.iterrows():
            if row["worker_i"] in Sset and row["worker_j"] in Sset:
                penalty += max(float(row["rho_error"]), 0.0)
        score += lambda_corr * penalty / max(1, len(S))
    return score


def greedy_select(
    workers: Sequence[str],
    q: Dict[str, float],
    edges: Set[Tuple[str, str]],
    budget: int,
    fixed_budget: bool = True,
    surrogate: str = "graph_bound",
    corr_df: Optional[pd.DataFrame] = None,
    lambda_corr: float = 1.0,
    max_bruteforce_nodes: int = 22,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Greedy forward selection.

    If fixed_budget=True:
        always select exactly budget workers if available.

    If fixed_budget=False:
        keep the best set found along the path of sizes 1..budget.
    """
    workers = sorted(list(dict.fromkeys(workers)))
    selected: List[str] = []
    history = []

    def score(S: Sequence[str]) -> float:
        if surrogate == "graph_bound":
            return graph_surrogate(S, q, edges, max_bruteforce_nodes=max_bruteforce_nodes)
        if surrogate == "accuracy_diversity":
            return simple_accuracy_diversity_surrogate(S, q, corr_df, lambda_corr=lambda_corr)
        raise ValueError(f"Unknown surrogate: {surrogate}")

    best_set: List[str] = []
    best_score = float("inf")

    max_steps = min(budget, len(workers))
    for step in range(1, max_steps + 1):
        candidates = [w for w in workers if w not in selected]
        cand_rows = []
        for w in candidates:
            S_new = selected + [w]
            val = score(S_new)
            cand_rows.append((val, w, S_new))

        cand_rows.sort(key=lambda x: (x[0], q.get(x[1], 0.5), x[1]))
        val, w_star, S_star = cand_rows[0]
        selected = S_star

        current_score = score(selected)
        current_density = graph_density(selected, edges)
        current_alpha = len(max_independent_set_exact(selected, edges, max_bruteforce_nodes=max_bruteforce_nodes))

        history.append({
            "step": step,
            "selected_worker": w_star,
            "subset_size": len(selected),
            "surrogate_value": current_score,
            "mean_error_rate": float(np.mean([q.get(w, 0.5) for w in selected])),
            "graph_density": current_density,
            "alpha_independent_set": current_alpha,
            "selected_set": "|".join(selected),
        })

        if current_score < best_score:
            best_score = current_score
            best_set = selected.copy()

    if fixed_budget:
        final_set = selected
    else:
        final_set = best_set

    return final_set, pd.DataFrame(history)


# =============================================================================
# Evaluation
# =============================================================================

def label_prior_order(train: pd.DataFrame) -> List[str]:
    if "truth" in train.columns and train["truth"].notna().any():
        counts = train.drop_duplicates("item")["truth"].astype(str).value_counts()
    else:
        counts = train["label"].astype(str).value_counts()
    return list(counts.index)


def predict_with_selected_workers(
    df: pd.DataFrame,
    selected: Sequence[str],
    tie_order: Optional[List[str]] = None,
    fallback_to_all: bool = False,
) -> pd.DataFrame:
    selected = set(selected)
    rows = []
    for item, g in df.groupby("item"):
        gs = g[g["worker"].isin(selected)]
        used_fallback = False
        if len(gs) == 0 and fallback_to_all:
            gs = g
            used_fallback = True
        pred = majority_vote(list(gs["label"]), tie_order) if len(gs) > 0 else None
        truth = None
        if "truth" in g.columns and g["truth"].notna().any():
            truth = g["truth"].dropna().astype(str).iloc[0]
        rows.append({
            "item": item,
            "prediction": pred,
            "truth": truth,
            "covered": pred is not None,
            "used_fallback": used_fallback,
            "n_selected_votes": len(gs),
        })
    return pd.DataFrame(rows)


def evaluate_predictions(pred: pd.DataFrame) -> Dict[str, float]:
    covered = pred[pred["covered"]].copy()
    if len(covered) == 0:
        return {"accuracy": np.nan, "coverage": 0.0, "n_eval_items": 0, "fallback_rate": np.nan}
    valid = covered.dropna(subset=["truth"])
    if len(valid) == 0:
        return {"accuracy": np.nan, "coverage": len(covered) / len(pred), "n_eval_items": len(covered), "fallback_rate": float(covered["used_fallback"].mean())}
    acc = (valid["prediction"].astype(str) == valid["truth"].astype(str)).mean()
    return {
        "accuracy": float(acc),
        "coverage": float(len(covered) / len(pred)),
        "n_eval_items": int(len(valid)),
        "fallback_rate": float(covered["used_fallback"].mean()),
    }


# =============================================================================
# Plotting
# =============================================================================

def save_plots(results: pd.DataFrame, out_dir: Path, paper_baselines: Optional[pd.DataFrame] = None) -> None:
    if results.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: accuracy by dataset.
    final = results[results["is_final"]].copy()
    if not final.empty:
        plt.figure(figsize=(12, 5))
        plot_df = final.sort_values("accuracy")
        plt.bar(plot_df["dataset"], plot_df["accuracy"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Greedy dependence-aware selection: accuracy by dataset")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_accuracy_by_dataset.png", dpi=200)
        plt.close()

    # Plot 2: accuracy vs budget.
    plt.figure(figsize=(12, 6))
    for dataset, g in results.groupby("dataset"):
        g = g.sort_values("budget")
        if g["accuracy"].notna().any():
            plt.plot(g["budget"], g["accuracy"], marker="o", label=dataset)
    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs selected budget")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_accuracy_vs_budget.png", dpi=200)
    plt.close()

    # Plot 3: surrogate vs budget.
    plt.figure(figsize=(12, 6))
    for dataset, g in results.groupby("dataset"):
        g = g.sort_values("budget")
        if g["surrogate_value"].notna().any():
            plt.plot(g["budget"], g["surrogate_value"], marker="o", label=dataset)
    plt.xlabel("Budget")
    plt.ylabel("Surrogate value")
    plt.title("Graph-dependent surrogate vs selected budget")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_surrogate_vs_budget.png", dpi=200)
    plt.close()

    # Plot 4: selected graph density.
    plt.figure(figsize=(12, 5))
    plot_df = final.sort_values("selected_graph_density") if not final.empty else results.sort_values("selected_graph_density")
    if not plot_df.empty:
        plt.bar(plot_df["dataset"], plot_df["selected_graph_density"])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Selected graph density")
        plt.title("Dependency density inside selected crowd")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_graph_density.png", dpi=200)
        plt.close()

    # Optional merged baseline plot.
    if paper_baselines is not None and not paper_baselines.empty and not final.empty:
        merged_rows = []
        for _, row in final.iterrows():
            merged_rows.append({"dataset": row["dataset"], "method": "GreedyGraph", "accuracy": row["accuracy"]})
        base = paper_baselines[["dataset", "method", "accuracy"]].copy()
        comp = pd.concat([base, pd.DataFrame(merged_rows)], ignore_index=True)
        comp.to_csv(out_dir / "merged_with_paper_baselines.csv", index=False)

        # Plot only methods present; can be crowded but useful.
        plt.figure(figsize=(14, 6))
        methods = list(comp["method"].dropna().unique())
        datasets = list(final["dataset"].unique())
        x = np.arange(len(datasets))
        width = 0.8 / max(1, len(methods))
        for k, method in enumerate(methods):
            vals = []
            for d in datasets:
                sub = comp[(comp["dataset"] == d) & (comp["method"] == method)]
                vals.append(float(sub["accuracy"].iloc[0]) if len(sub) else np.nan)
            plt.bar(x + k * width, vals, width=width, label=method)
        plt.xticks(x + width * (len(methods) - 1) / 2, datasets, rotation=60, ha="right")
        plt.ylabel("Accuracy")
        plt.title("GreedyGraph vs paper baselines")
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_greedy_vs_paper_baselines.png", dpi=200)
        plt.close()


# =============================================================================
# Main experiment
# =============================================================================

def run_one_dataset(
    data_root: Path,
    info: DatasetInfo,
    out_dir: Path,
    budget: int,
    budget_percent: Optional[float],
    fixed_budget: bool,
    test_frac: float,
    seed: int,
    corr_threshold: float,
    min_pair_overlap: int,
    smoothing_alpha: float,
    surrogate: str,
    lambda_corr: float,
    fallback_to_all: bool,
    max_bruteforce_nodes: int,
) -> Tuple[List[Dict], Dict]:
    df = load_dataset(data_root, info)
    if df is None:
        warnings.warn(f"Dataset {info.canonical_name} not found under {data_root}. Skipping.")
        return [], {}

    df, used_pseudo_truth = add_pseudo_truth_if_needed(df)
    train, test = train_test_split_items(df, test_frac=test_frac, seed=seed)

    q = estimate_worker_error_rates(train, alpha=smoothing_alpha)
    edges, corr_df = estimate_error_correlation_graph(
        train,
        corr_threshold=corr_threshold,
        min_pair_overlap=min_pair_overlap,
        positive_only=True,
    )

    workers = sorted(q.keys())
    if len(workers) == 0:
        warnings.warn(f"No workers found for {info.canonical_name}. Skipping.")
        return [], {}

    tie_order = label_prior_order(train)

    rows = []
    selected_by_budget = {}

    # Dataset-specific budget. If --budget_percent is provided, each dataset uses
    # ceil(percent * number_of_workers / 100), clipped to [1, n_workers].
    max_budget = resolve_worker_budget(
        n_workers=len(workers),
        budget=budget,
        budget_percent=budget_percent,
        min_budget=1,
    )

    for B in range(1, max_budget + 1):
        selected, hist = greedy_select(
            workers,
            q=q,
            edges=edges,
            budget=B,
            fixed_budget=True,
            surrogate=surrogate,
            corr_df=corr_df,
            lambda_corr=lambda_corr,
            max_bruteforce_nodes=max_bruteforce_nodes,
        )

        pred = predict_with_selected_workers(
            test,
            selected,
            tie_order=tie_order,
            fallback_to_all=fallback_to_all,
        )
        metrics = evaluate_predictions(pred)

        val = graph_surrogate(selected, q, edges, max_bruteforce_nodes=max_bruteforce_nodes) if surrogate == "graph_bound" else simple_accuracy_diversity_surrogate(selected, q, corr_df, lambda_corr=lambda_corr)
        alpha = len(max_independent_set_exact(selected, edges, max_bruteforce_nodes=max_bruteforce_nodes))
        density = graph_density(selected, edges)

        row = {
            "dataset": info.canonical_name,
            "budget": B,
            "budget_mode": "percent" if budget_percent is not None else "absolute",
            "budget_percent": budget_percent,
            "selected_size": len(selected),
            "selected_workers": "|".join(selected),
            "surrogate": surrogate,
            "surrogate_value": val,
            "accuracy": metrics["accuracy"],
            "coverage": metrics["coverage"],
            "n_eval_items": metrics["n_eval_items"],
            "fallback_rate": metrics["fallback_rate"],
            "selected_graph_density": density,
            "selected_alpha": alpha,
            "n_workers_total": len(workers),
            "n_items_total": df["item"].nunique(),
            "n_votes_total": len(df),
            "n_edges_graph": len(edges),
            "global_graph_density": graph_density(workers, edges),
            "used_pseudo_truth": used_pseudo_truth,
            "is_final": False,
        }
        rows.append(row)
        selected_by_budget[B] = selected

    if fixed_budget:
        final_budget = max_budget
    else:
        # Choose budget with lowest surrogate along the sweep.
        valid = [r for r in rows if not np.isnan(r["surrogate_value"])]
        final_budget = min(valid, key=lambda r: r["surrogate_value"])["budget"] if valid else max_budget

    for r in rows:
        r["is_final"] = (r["budget"] == final_budget)

    selected_info = {
        "dataset": info.canonical_name,
        "final_budget": final_budget,
        "budget_mode": "percent" if budget_percent is not None else "absolute",
        "budget_percent": budget_percent,
        "selected_by_budget": selected_by_budget,
        "q_error_rates": q,
        "edges": list([list(e) for e in sorted(edges)]),
        "used_pseudo_truth": used_pseudo_truth,
    }

    # Save per-dataset intermediate details.
    ds_out = out_dir / "per_dataset" / info.canonical_name
    ds_out.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(ds_out / "pairwise_error_correlations.csv", index=False)
    pd.DataFrame(rows).to_csv(ds_out / "budget_sweep.csv", index=False)
    with open(ds_out / "selected_workers.json", "w", encoding="utf-8") as f:
        json.dump(selected_info, f, indent=2, ensure_ascii=False)

    return rows, selected_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy graph-dependent crowd selection.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root folder containing dataset folders.")
    parser.add_argument("--out_dir", type=str, default="./results_greedy_graph", help="Output folder.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional list of dataset names to run.")
    parser.add_argument("--budget", type=int, default=5, help="Maximum or fixed number of selected annotators. Ignored when --budget_percent is provided.")
    parser.add_argument("--budget_percent", type=float, default=None, help="Dataset-specific budget as a percentage of the available workers, e.g. 50 means select ceil(0.5 * n_workers).")
    parser.add_argument("--fixed_budget", action="store_true", help="If set, final selection uses exactly the resolved budget.")
    parser.add_argument("--test_frac", type=float, default=0.30, help="Item-level test fraction.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corr_threshold", type=float, default=0.10, help="Positive error-correlation threshold for edges.")
    parser.add_argument("--min_pair_overlap", type=int, default=5, help="Minimum co-labeled items for pairwise correlation.")
    parser.add_argument("--smoothing_alpha", type=float, default=1.0, help="Beta smoothing for worker error rates.")
    parser.add_argument("--surrogate", choices=["graph_bound", "accuracy_diversity"], default="graph_bound")
    parser.add_argument("--lambda_corr", type=float, default=1.0, help="Penalty weight for accuracy_diversity surrogate.")
    parser.add_argument("--fallback_to_all", action="store_true", help="If selected workers do not cover a test item, use all available votes.")
    parser.add_argument("--paper_baselines_csv", type=str, default=None, help="Optional CSV with columns dataset,method,accuracy.")
    parser.add_argument("--max_bruteforce_nodes", type=int, default=22, help="Exact maximum independent set up to this subset size.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        wanted = {d.lower() for d in args.datasets}
        dataset_infos = [
            info for info in DATASETS
            if info.canonical_name.lower() in wanted or any(a.lower() in wanted for a in info.aliases)
        ]
    else:
        dataset_infos = DATASETS

    # Save registry.
    pd.DataFrame([asdict(d) for d in DATASETS]).to_csv(out_dir / "dataset_registry.csv", index=False)

    all_rows: List[Dict] = []
    selected_all: Dict[str, Dict] = {}

    for info in dataset_infos:
        print(f"\n=== Running {info.canonical_name} ===")
        rows, selected_info = run_one_dataset(
            data_root=data_root,
            info=info,
            out_dir=out_dir,
            budget=args.budget,
            budget_percent=args.budget_percent,
            fixed_budget=args.fixed_budget,
            test_frac=args.test_frac,
            seed=args.seed,
            corr_threshold=args.corr_threshold,
            min_pair_overlap=args.min_pair_overlap,
            smoothing_alpha=args.smoothing_alpha,
            surrogate=args.surrogate,
            lambda_corr=args.lambda_corr,
            fallback_to_all=args.fallback_to_all,
            max_bruteforce_nodes=args.max_bruteforce_nodes,
        )
        all_rows.extend(rows)
        if selected_info:
            selected_all[info.canonical_name] = selected_info

    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "per_dataset_results.csv", index=False)

    with open(out_dir / "selected_workers.json", "w", encoding="utf-8") as f:
        json.dump(selected_all, f, indent=2, ensure_ascii=False)

    paper_baselines = None
    if args.paper_baselines_csv is not None and Path(args.paper_baselines_csv).exists():
        paper_baselines = pd.read_csv(args.paper_baselines_csv)

    save_plots(results, out_dir, paper_baselines=paper_baselines)

    print("\nDone.")
    print(f"Results saved to: {out_dir.resolve()}")
    if results.empty:
        print("No datasets were loaded. Check --data_root and dataset folder names.")
    else:
        final = results[results["is_final"]].copy()
        print(final[["dataset", "budget", "budget_mode", "budget_percent", "selected_size", "accuracy", "coverage", "surrogate_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
