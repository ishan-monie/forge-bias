# LLM bias prototype

When you remove one social bias from a word embedding (for example racial bias), does bias toward the **other** groups get better, worse, or stay the same?

**Live dashboard:** https://ishan-monie.github.io/forge-bias/

This is a prototype, not a research finding: small pre-trained models, short word lists, and no significance testing.

## What it does

1. Takes a pre-trained embedding (GloVe 100d and fastText 300d so far).
2. Scores each of 72 social groups: mean cosine similarity to positive words minus mean cosine similarity to negative words (also for competent vs. incompetent).
3. Defines a category's **gap** as its highest group score minus its lowest.
4. "Debiases" one category by removing the subspace spanned by its distinct group centroids from every vector (the neutralize step of Bolukbasi et al., 2016). Full rank by default; `k` can cap the number of directions.
5. Re-scores everything and reports how every other category's gap changed (the spillover).

The dashboard (`index.html`) shows the spillover for any chosen category, per-group scores, a before/after similarity network, a comparison of any two groups, a side-by-side comparison of the two models, and CSV downloads (group scores and pairwise similarity).

## Run it

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py            # downloads the models via gensim (GloVe ~130 MB, fastText ~1 GB), writes results_*.csv
python build_dashboard.py  # builds index.html from the results
```

`python run.py glove100` runs a single model. Every results file carries the `model`, `method` and `k` used.

## Files

- `forge_bias.py` - scoring, subspace removal, before/after tables
- `run.py` - runs every model at full rank and at k=3
- `build_dashboard.py`, `dashboard_template_v4.html` - build the single-file dashboard (`index.html`)
- `dashboard_template_v2.html` - the more technical single-model page (`forge_dashboard_v2.html`)
- `data/groups_72.csv` - the 72 groups: category, group, words
- `results_<model>_<fullrank|k3>_<scores|pairwise|spillover>.csv` - outputs

## Data and credits

- The 72 group word lists and their categories come from Charlesworth et al. (2023), *Journal of Personality and Social Psychology*, via the authors' OSF project: https://osf.io/gzuy4/ (`stigmalists_relativegroups.csv`, `stigmaclusters_relativegroups.csv`). The lists contain offensive terms because they come from the original study of stigma in language. Categories for four rows that were blank in the clusters file (Fertility, Attractiveness) and the merge of Blue/White collar into Occupation were assigned here.
- Embeddings are loaded through [gensim-data](https://github.com/piskvorky/gensim-data): `glove-wiki-gigaword-100` (Pennington et al., 2014) and `fasttext-wiki-news-subwords-300` (Mikolov et al., 2018).
- Method: Bolukbasi et al. (2016), "Man is to computer programmer as woman is to homemaker?"

## Known limitations

- No gender category exists in the 72 groups.
- Several groups share identical word lists (for example six `white_*` groups). They stay in the scoring tables but are collapsed when computing the subspace.
- Words with several meanings ("white", "black", "native", "dumb") are blended into one point.
- A change smaller than 0.005 is treated as "no change"; this is a judgment call, not a statistical test.
