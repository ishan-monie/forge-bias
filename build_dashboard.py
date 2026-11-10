"""Build the single-file dashboards from the full-rank results CSVs written by run.py.

v1-v3 are single-model pages (they use the first model). v4 embeds every model that has results
and lets the viewer switch between them and compare.
"""
import json, os
import numpy as np, pandas as pd

# (slug, display name, gensim-data name) - keep in sync with run.py
MODELS = [
    ("glove100", "GloVe wiki-gigaword 100d", "glove-wiki-gigaword-100"),
    ("fasttext300", "fastText wiki-news 300d", "fasttext-wiki-news-subwords-300"),
]
# result-file tags, in dropdown order (tag -> id used by the page); keep in sync with run.py
TAGS = ["fullrank", "k3", "hard", "inlp"]
WORDS = pd.read_csv("data/groups_72.csv").set_index("group")["words"].map(lambda s: tuple(sorted(s.split())))


def build_model(slug, display, gname):
    tags = [t for t in TAGS if os.path.exists(f"results_{slug}_{t}_scores.csv")]
    rd = lambda t, kind: pd.read_csv(f"results_{slug}_{t}_{kind}.csv")
    sc, pr, sm = rd(tags[0], "scores"), rd(tags[0], "pairwise"), rd(tags[0], "spillover")
    targets = list(dict.fromkeys(sc["target_category"]))
    base = sc[sc.target_category == targets[0]][["category", "group"]].reset_index(drop=True)
    groups = base["group"].tolist()
    gi = {g: i for i, g in enumerate(groups)}
    cats = list(dict.fromkeys(base["category"]))

    # nodes: collapse groups with identical word lists
    parent = list(range(len(groups)))
    first = {}
    for g in groups:
        first.setdefault(WORDS[g], g)
        parent[gi[g]] = gi[first[WORDS[g]]]
    roots = list(dict.fromkeys(parent))
    node_of = {r: n for n, r in enumerate(roots)}
    node_idx = [node_of[parent[i]] for i in range(len(groups))]
    members = [[groups[i] for i in range(len(groups)) if node_idx[i] == n] for n in range(len(roots))]
    rep = [groups[r] for r in roots]
    n = len(roots)
    label = lambda m: m[0] if len(m) == 1 else f"{m[0]} +{len(m)-1}"

    def simmat(df, col):
        M = np.eye(n)
        look = {}
        for a, b, s in zip(df.group_a, df.group_b, df[col]):
            look[(a, b)] = s; look[(b, a)] = s
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = look[(rep[i], rep[j])]
        return M
    tri = lambda M: [round(float(M[i, j]), 3) for i in range(n) for j in range(i + 1, n)]

    t0 = pr[pr.target_category == targets[0]]
    Sb = simmat(t0, "similarity_before")
    D = 1 - Sb; np.fill_diagonal(D, 0)          # classical MDS layout from the "before" similarities
    J = np.eye(n) - 1 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    X = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    cat_of = sc.drop_duplicates("group").set_index("group")["category"]

    data = {
        "model": f"{display} ({gname})", "name": display, "categories": cats, "targets": targets,
        "groups": [{"g": g, "c": c, "n": node_idx[i]} for i, (g, c) in enumerate(zip(base.group, base.category))],
        "before": {m: [round(x, 4) for x in sc[sc.target_category == targets[0]][f"{m}_before"]] for m in ("valence", "competence")},
        "nodes": [{"label": label(members[i]), "members": members[i], "cat": cat_of[rep[i]],
                   "x": round(float(X[i, 0]), 4), "y": round(float(X[i, 1]), 4)} for i in range(n)],
        "simBefore": tri(Sb),
    }
    # pairwise table (all 72 groups) for the CSV download: pair order + "before"; "after" is stored per method
    pr0 = pr[pr.target_category == targets[0]].reset_index(drop=True)
    key = list(zip(pr0.group_a, pr0.group_b))
    data["pairs"] = {"a": [gi[x] for x, _ in key], "b": [gi[y] for _, y in key],
                     "before": [round(float(v), 5) for v in pr0.similarity_before]}
    data["methods"] = {}
    for tag in tags:
        sc_t, pr_t, sm_t = (sc, pr, sm) if tag == tags[0] else (rd(tag, "scores"), rd(tag, "pairwise"), rd(tag, "spillover"))
        assert list(sc_t.drop_duplicates("group").group) == groups and np.allclose(
            sc_t[sc_t.target_category == targets[0]].valence_before.values, sc[sc.target_category == targets[0]].valence_before.values, atol=1e-4)
        mb = {
            "method": sm_t["method"].iloc[0],
            "after": {t: {m: [round(x, 4) for x in sc_t[sc_t.target_category == t][f"{m}_after"]] for m in ("valence", "competence")} for t in targets},
            "k": {t: int(sm_t[sm_t.target_category == t].k.iloc[0]) for t in targets},
            "nGroups": {t: int(sm_t[sm_t.target_category == t].target_n_groups.iloc[0]) for t in targets},
            "nDistinct": {t: int(sm_t[sm_t.target_category == t].target_n_distinct.iloc[0]) for t in targets},
            "simAfter": {t: tri(simmat(pr_t[pr_t.target_category == t], "similarity_after")) for t in targets},
            "pairsAfter": {},
        }
        for t in targets:
            pt = pr_t[pr_t.target_category == t].set_index(["group_a", "group_b"]).loc[key]
            assert len(pt) == len(key) == 2556 and np.allclose(pt.similarity_before.values, pr0.similarity_before.values, atol=1e-4)
            mb["pairsAfter"][t] = [round(float(v), 5) for v in pt.similarity_after]
        # the page recomputes gaps from group scores; check against the summary CSV
        for t in targets:
            for m in ("valence", "competence"):
                for c in cats:
                    idx = [i for i, g in enumerate(data["groups"]) if g["c"] == c]
                    ga = max(mb["after"][t][m][i] for i in idx) - min(mb["after"][t][m][i] for i in idx)
                    row = sm_t[(sm_t.target_category == t) & (sm_t.category == c) & (sm_t.metric == m)]
                    assert abs(ga - row.gap_after.iloc[0]) < 2e-4, (slug, tag, t, c, m, ga, row.gap_after.iloc[0])
        data["methods"][tag] = mb
    return data


built = {}
for slug, display, gname in MODELS:
    if os.path.exists(f"results_{slug}_fullrank_scores.csv"):
        built[slug] = build_model(slug, display, gname)
        print(f"built data for {display}")
first_slug = next(iter(built))
b0 = built[first_slug]   # older single-model pages: first model, first method, no pairwise table
single = json.dumps({**{k: v for k, v in b0.items() if k not in ("pairs", "methods")}, **{k: v for k, v in b0["methods"]["fullrank"].items() if k != "pairsAfter"}}, separators=(",", ":"))
multi = json.dumps({"order": list(built), "methodOrder": list(b0["methods"]), "models": built}, separators=(",", ":"))

PAGES = [("dashboard_template_v1.html", "forge_dashboard_v1.html", single),
         ("dashboard_template_v2.html", "forge_dashboard_v2.html", single),
         ("dashboard_template_v3.html", "forge_dashboard_v3.html", single),
         ("dashboard_template_v4.html", "forge_dashboard_v4.html", multi)]
for tpl, out, blob in PAGES:
    if not os.path.exists(tpl):
        continue
    html = open(tpl, encoding="utf-8").read().replace("/*__DATA__*/null", blob)
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}: {len(html)/1024:.0f} KB")
    if out == "forge_dashboard_v4.html":      # landing page for GitHub Pages
        open("index.html", "w", encoding="utf-8").write(html)
        print("wrote index.html (copy of v4)")
