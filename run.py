"""Run the experiment for every embedding model and every debiasing method.

    python run.py              # all models in MODELS
    python run.py glove100     # just one (by slug)

Writes results_<slug>_<fullrank|k3|hard|inlp>_<scores|pairwise|spillover>.csv. Every table has `model`, `method` and `k`
(k = number of directions removed).
"""
import logging; logging.disable(logging.CRITICAL)
import gc
import sys
import gensim.downloader as api
import numpy as np
import pandas as pd
from forge_bias import *

# (slug, gensim-data name, display name)
MODELS = [
    ("glove100", "glove-wiki-gigaword-100", "GloVe wiki-gigaword 100d"),
    ("fasttext300", "fasttext-wiki-news-subwords-300", "fastText wiki-news 300d"),
]
# (tag, label shown on the dashboard and in the CSVs, run_experiment arguments)
METHODS = [
    ("fullrank", "Subtract group subspace (Bolukbasi neutralize)", dict(method="neutralize", k=None)),
    ("k3", "Subtract top 3 directions", dict(method="neutralize", k=3)),
    ("hard", "Hard debias (Bolukbasi neutralize + equalize)", dict(method="hard", k=None)),
    ("inlp", "INLP (iterative nullspace projection)", dict(method="inlp", k=None)),
]
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
want = set(sys.argv[1:])

for slug, name, display in MODELS:
    if want and slug not in want:
        continue
    print(f"\n===== {display} ({name}) =====", flush=True)
    kv = api.load(name)
    res = {}
    for tag, label, kw in METHODS:
        scores, pairs, summary, dropped = run_experiment(kv, **kw)
        res[tag] = (scores, pairs, summary)
        for df in (scores, pairs, summary):
            df.insert(0, "model", display)
            df.insert(1, "method", label)
        scores.round(5).to_csv(f'results_{slug}_{tag}_scores.csv', index=False)
        pairs.round(5).to_csv(f'results_{slug}_{tag}_pairwise.csv', index=False)
        summary.round(5).to_csv(f'results_{slug}_{tag}_spillover.csv', index=False)
        print(f"[{tag}] wrote {len(scores)} score rows, {len(pairs)} pair rows, {len(summary)} summary rows", flush=True)
    miss = {g: w for g, w in dropped.items() if w}
    print(f"OOV: {sum(len(w) for w in miss.values())} words dropped across {len(miss)} lists")

    vecs, _ = build_vectors(kv)
    b = category_subspace(vecs, 'Age')
    after = subtract_subspace(vecs, b)
    print("residual along Age axis (max abs):", float(np.abs(after[('group', 'old_young')] @ b.T).max()))

    full, k3 = (category_table(res[t][2]) for t in ("fullrank", "k3"))
    tab = full.copy()
    tab["k3_own_after"] = k3["own_after"].values
    tab["k3_others_mean"] = k3["others_mean_chg"].values
    tab["k3_others_max"] = k3["others_max_chg"].values
    print("\nValence gaps; k3_* = capped at k=3. * = flagged. Spillover = change in the OTHER categories' gaps.\n")
    print(tab.round(4).to_string(index=False), flush=True)
    del kv, vecs, after, res
    gc.collect()
