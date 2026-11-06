"""
forge_bias: measure how debiasing one group's axis shifts bias toward ALL groups.

Pipeline:
  1. Load a pre-trained embedding (any gensim KeyedVectors).
  2. Bias score per group  = mean cos(group, GOOD) - mean cos(group, BAD)
     (also computed for COMPETENT vs INCOMPETENT).
  3. Debias a target category by subtracting its group subspace
     (Bolukbasi et al. 2016 "neutralize" step, simplest version).
  4. Recompute bias scores and pairwise group-group similarity.
  5. Write before/after CSVs.

GROUPS is the 72-group list from Charlesworth et al. (2023, JPSP):
https://osf.io/gzuy4/files/cdx5z (loaded from data/groups_72.csv).
"""
from __future__ import annotations
import itertools
import os
import numpy as np
import pandas as pd

# ---- Concept word lists (from the project notes; user-editable) ----------
CONCEPTS = {
    "good": ["nice", "pleasant", "wonderful", "good", "fantastic"],
    "bad": ["bad", "horrible", "horrific", "disgusting", "unpleasant"],
    "competent": ["competent", "talented", "smart", "capable", "able"],
    "incompetent": ["incompetent", "untalented", "dumb", "incapable", "unable"],
}

# ---- The 72 groups (Charlesworth et al. 2023, JPSP; OSF gzuy4, file cdx5z) --------
# data/groups_72.csv: category, group, words. Group = one row of the OSF file
# (e.g. "old_young" = the "old" word list, defined relative to "young").
# Categories come from stigmaclusters_relativegroups.csv; see data/groups_72.csv.
def load_groups(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "groups_72.csv")
    out = {}
    for r in pd.read_csv(path).itertuples():
        out.setdefault(r.category, {})[r.group] = str(r.words).split()
    return out


GROUPS = load_groups()


def group_table(groups=GROUPS):
    return [(cat, g, w) for cat, d in groups.items() for g, w in d.items()]


def _filter_words(kv, words):
    return [w for w in words if w in kv.key_to_index]


def build_vectors(kv, groups=GROUPS, concepts=CONCEPTS):
    """Return {name: (n_words, dim) array} for every group and concept, OOV dropped."""
    vecs, dropped = {}, {}
    for cat, g, words in group_table(groups):
        ok = _filter_words(kv, words)
        dropped[g] = [w for w in words if w not in ok]
        vecs[("group", g)] = np.stack([kv[w] for w in ok])
    for c, words in concepts.items():
        ok = _filter_words(kv, words)
        dropped[c] = [w for w in words if w not in ok]
        vecs[("concept", c)] = np.stack([kv[w] for w in ok])
    return vecs, dropped


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def mean_cos(a, b):
    """Average cosine similarity between every row of a and every row of b."""
    return float((_unit(a) @ _unit(b).T).mean())


def bias_scores(vecs, groups=GROUPS):
    rows = []
    for cat, g, _ in group_table(groups):
        gv = vecs[("group", g)]
        val = mean_cos(gv, vecs[("concept", "good")]) - mean_cos(gv, vecs[("concept", "bad")])
        comp = mean_cos(gv, vecs[("concept", "competent")]) - mean_cos(gv, vecs[("concept", "incompetent")])
        rows.append({"category": cat, "group": g, "valence": val, "competence": comp})
    return pd.DataFrame(rows)


def pairwise(vecs, groups=GROUPS):
    names = [g for _, g, _ in group_table(groups)]
    rows = []
    for a, b in itertools.combinations(names, 2):
        rows.append({"group_a": a, "group_b": b,
                     "similarity": mean_cos(vecs[("group", a)], vecs[("group", b)])})
    return pd.DataFrame(rows)


# ---- Debiasing ---------------------------------------------------------------
def distinct_centroids(vecs, category, groups=GROUPS):
    """Unit-mean centroid per group, with groups that have identical centroids
    (e.g. the six identical "white_*" lists) collapsed to one.
    Returns (names_of_distinct_groups, centroids[n_distinct, dim])."""
    names = list(groups[category].keys())
    cents = np.stack([_unit(vecs[("group", g)]).mean(axis=0) for g in names])
    _, idx = np.unique(np.round(cents, 6), axis=0, return_index=True)
    idx = np.sort(idx)
    return [names[i] for i in idx], cents[idx]


def category_subspace(vecs, category, groups=GROUPS, k=None):
    """Subspace spanned by the category's DISTINCT group centroids (Bolukbasi-style).

    k=None  -> full rank (rank of the centred distinct centroids; 2 groups -> 1 axis).
    k=int   -> the top-k directions, capped at that rank.
    Duplicated groups are collapsed first so they do not get extra weight."""
    names, cents = distinct_centroids(vecs, category, groups)
    if len(names) < 2:
        raise ValueError(f"'{category}' has fewer than 2 distinct groups; nothing to subtract.")
    cents = cents - cents.mean(axis=0)
    _, s, vt = np.linalg.svd(cents, full_matrices=False)
    rank = int((s > s.max() * 1e-6).sum())
    k_used = rank if k is None else min(int(k), rank)
    return vt[:k_used]  # (k_used, dim), orthonormal rows


def subtract_subspace(vecs, basis):
    """Project every vector off the subspace: v - B^T B v."""
    out = {}
    for key, v in vecs.items():
        out[key] = v - (v @ basis.T) @ basis
    return out


def category_gap(scores, category, col):
    """Within-category gap = max - min group score (the double difference, generalised)."""
    v = scores.loc[scores["category"] == category, col]
    return float(v.max() - v.min())


def run_experiment(kv, groups=GROUPS, concepts=CONCEPTS, k=None):
    """Before/after tables for every debiasable category.
    k=None uses the full rank of each category's distinct-group subspace; k=int caps it.
    Every output table carries the k actually used."""
    vecs, dropped = build_vectors(kv, groups, concepts)
    base_scores = bias_scores(vecs, groups)
    base_pairs = pairwise(vecs, groups)
    score_rows, pair_rows, summary_rows = [], [], []
    for cat in groups:
        if len(distinct_centroids(vecs, cat, groups)[0]) < 2:
            continue
        basis = category_subspace(vecs, cat, groups, k=k)
        k_used = int(basis.shape[0])
        n_distinct = len(distinct_centroids(vecs, cat, groups)[0])
        after_vecs = subtract_subspace(vecs, basis)
        after_scores = bias_scores(after_vecs, groups)
        after_pairs = pairwise(after_vecs, groups)

        s = base_scores.merge(after_scores, on=["category", "group"], suffixes=("_before", "_after"))
        s.insert(0, "target_category", cat)
        s["valence_change"] = s["valence_after"] - s["valence_before"]
        s["competence_change"] = s["competence_after"] - s["competence_before"]
        s["k"] = k_used
        score_rows.append(s)

        p = base_pairs.merge(after_pairs, on=["group_a", "group_b"], suffixes=("_before", "_after"))
        p.insert(0, "target_category", cat)
        p["k"] = k_used
        pair_rows.append(p)

        for c2, d in groups.items():
            if len(d) < 2:
                continue
            for metric in ("valence", "competence"):
                gb = category_gap(s, c2, f"{metric}_before")
                ga = category_gap(s, c2, f"{metric}_after")
                summary_rows.append({
                    "target_category": cat, "category": c2, "is_target": c2 == cat,
                    "metric": metric, "gap_before": gb, "gap_after": ga, "gap_change": ga - gb,
                    "k": k_used, "target_n_groups": len(groups[cat]), "target_n_distinct": n_distinct,
                })
    return (pd.concat(score_rows, ignore_index=True),
            pd.concat(pair_rows, ignore_index=True),
            pd.DataFrame(summary_rows),
            dropped)


def category_table(summary, metric="valence", flag=("Race", "Mental health", "Physical disability")):
    """One row per debiased category: size, k, own gap before/after, spillover to the others."""
    rows = []
    for cat, d in summary[summary["metric"] == metric].groupby("target_category", sort=False):
        own, oth = d[d["is_target"]].iloc[0], d[~d["is_target"]]
        rows.append({
            "category": cat + (" *" if cat in flag else ""),
            "groups": int(own["target_n_groups"]), "distinct": int(own["target_n_distinct"]),
            "k": int(own["k"]),
            "own_before": own["gap_before"], "own_after": own["gap_after"],
            "others_mean_chg": oth["gap_change"].mean(), "others_max_chg": oth["gap_change"].max(),
            "n_others_up": int((oth["gap_change"] > 0).sum()), "n_others": len(oth),
        })
    return pd.DataFrame(rows)


def double_difference(scores, group_a, group_b, col="valence_before"):
    """How much stronger is A=good/bad than B=good/bad (the notes' headline index)."""
    one = scores.drop_duplicates("group").set_index("group")[col]
    return float(one[group_a] - one[group_b])
