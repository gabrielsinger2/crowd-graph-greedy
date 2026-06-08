# Crowd Graph Greedy

Greedy dependence-aware crowd selection for crowdsourcing datasets.

This repository contains a proof-of-concept implementation for selecting annotators under dependency-aware risk surrogates. The core idea is:

\[
\text{labels} \rightarrow \text{worker error rates} \rightarrow \text{naive error-correlation graph}
\rightarrow \text{graph-dependent surrogate} \rightarrow \text{greedy subset selection}.
\]

The script is designed for the 17 real-world crowdsourcing datasets often used in label aggregation papers:

- `CF`, `CF*`, `MS`, `SP`, `SP*`, `ZCall`, `ZCin`, `ZCus`
- `prod`, `senti`, `face`, `adult`
- `bird`, `dog`, `rte`, `trec`, `web`

It does **not** rerun paper baselines. If you already have baseline numbers, place them in a CSV and pass them with `--paper_baselines_csv`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/crowd-graph-greedy.git
cd crowd-graph-greedy

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## Data layout

Place each dataset in a folder under `data/`:

```text
data/
  CF/
  SP/
  adult/
  bird/
  ...
```

The loader tries to infer columns automatically. It expects annotation files containing columns equivalent to:

```text
item, worker, label, truth
```

Accepted aliases include:

- item: `item`, `item_id`, `task`, `task_id`, `object_id`
- worker: `worker`, `worker_id`, `annotator`, `annotator_id`, `user`
- label: `label`, `answer`, `annotation`, `vote`, `response`
- truth: `truth`, `true_label`, `ground_truth`, `gold`, `y`

If no truth is found, the script uses majority vote as pseudo-truth. This is useful only for debugging, not for proper evaluation.

---

## Run

Run all detected datasets:

```bash
python greedy_crowd_graph_experiment.py \
  --data_root ./data \
  --out_dir ./results \
  --budget 5 \
  --corr_threshold 0.10 \
  --min_pair_overlap 5 \
  --fixed_budget
```

Run only selected datasets:

```bash
python greedy_crowd_graph_experiment.py \
  --data_root ./data \
  --out_dir ./results \
  --datasets CF SP adult bird \
  --budget 5 \
  --fixed_budget
```

Use the simpler accuracy-diversity surrogate:

```bash
python greedy_crowd_graph_experiment.py \
  --data_root ./data \
  --out_dir ./results \
  --budget 5 \
  --surrogate accuracy_diversity \
  --lambda_corr 1.0 \
  --fixed_budget
```

Merge with baseline results from the paper:

```bash
python greedy_crowd_graph_experiment.py \
  --data_root ./data \
  --out_dir ./results \
  --budget 5 \
  --fixed_budget \
  --paper_baselines_csv paper_results.csv
```

The baseline CSV must have columns:

```text
dataset,method,accuracy
```

---

## Outputs

The script writes:

```text
results/
  per_dataset_results.csv
  selected_workers.json
  dataset_registry.csv
  plot_accuracy_by_dataset.png
  plot_accuracy_vs_budget.png
  plot_surrogate_vs_budget.png
  plot_graph_density.png
  per_dataset/
    <dataset>/
      pairwise_error_correlations.csv
      budget_sweep.csv
      selected_workers.json
```

---

## Mathematical idea

For a selected subset \(S\), let

\[
E_i = 1 - Z_i
\]

be the error indicator of source \(i\). If ties are counted as errors, the majority vote fails when

\[
\sum_{i\in S} E_i \geq t_S,
\qquad
t_S = \left\lceil \frac{|S|}{2} \right\rceil.
\]

The exact dependent risk

\[
R_\phi(S,x)
=
\mathbb P\left(
\sum_{i\in S} E_i \geq t_S
\mid X=x
\right)
\]

depends on the full joint law of the selected errors. This repository uses a simple estimated dependency graph and greedily minimizes a graph-dependent surrogate.

---

## GitHub push

To create the GitHub repository from this folder:

```bash
git init
git add .
git commit -m "Initial commit: greedy crowd graph experiment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crowd-graph-greedy.git
git push -u origin main
```

Then, from another computer:

```bash
git clone https://github.com/YOUR_USERNAME/crowd-graph-greedy.git
cd crowd-graph-greedy
pip install -r requirements.txt
```
