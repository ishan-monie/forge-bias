# Source material for forge_bias

Everything the project is built on, with where each item is used. Downloaded 2026-11-06.

## 1. The 72 groups: Charlesworth et al. (2023)

Charlesworth, T. E. S., Sanjeev, N., Hatzenbuehler, M. L., & Banaji, M. R. (2023). *Identifying and predicting stereotype change in large language corpora: 72 groups, 115 years (1900-2015), and 4 text sources.* Journal of Personality and Social Psychology. PubMed: https://pubmed.ncbi.nlm.nih.gov/37616081/

Authors' OSF project (code and data): https://osf.io/gzuy4/

| File (in `charlesworth_2023_stigma/`) | What it is | Where we use it |
|---|---|---|
| `Charlesworth_Stigma_Main_text_063023.pdf` | The paper (manuscript version hosted on the authors' OSF page; may differ from the journal's final typeset version) | Defines the 72 groups, the "relative" word-list design (e.g. old vs. young) and the four broad clusters. Cite this. |
| `Charlesworth_Stigma_SM_063023.pdf` | Supplementary materials | Background on group lists and clusters |
| `OSF_README.pdf` | The authors' instructions for reproducing their analyses | Explains the file layout of the OSF project |
| `stigmalists_relativegroups.csv` | **The 72 group word lists**: one row per group, `name,words` (OSF file id cdx5z, `wordstim/` folder). Has a byte-order mark. | Loaded into `data/groups_72.csv` (the `GROUPS` used by `forge_bias.py`) |
| `stigmaclusters_relativegroups.csv` | Cluster and category per group, `name,cluster,category` (OSF file id qyvfx) | Gives each group's category |

Things we changed or decided (these are ours, not the paper's):
- Four rows have a blank category in the clusters file (infertile/fertile, unattractive/attractive). We labelled them **Fertility** and **Attractiveness**.
- The file splits **Blue collar** and **White collar**. We merged them into **Occupation**.
- Our "categories" are the file's third column (15 of them). The paper's four "broad clusters" (Social group, Occupation, Mental, Body) are its second column, which we do not use.
- The paper's embeddings are Google Books, COHA, NYT and Common Crawl. We use different, widely available embeddings (below), so our numbers are not a reproduction of the paper.
- There is no Gender category among the 72 groups.

## 2. Method papers (in `methods/`)

| File | Reference | Where we use it |
|---|---|---|
| `Bolukbasi_2016_...pdf` | Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). *Man is to computer programmer as woman is to homemaker? Debiasing word embeddings.* arXiv:1607.06520 | The "neutralize" step: removing a category's group subspace from every vector (`subtract_subspace`, `category_subspace`) |
| `Pennington_2014_GloVe.pdf` | Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global vectors for word representation.* EMNLP. https://aclanthology.org/D14-1162/ | Model 1: `glove-wiki-gigaword-100` (Wikipedia 2014 + Gigaword 5, 100 dimensions) |
| `Mikolov_2018_...fastText.pdf` | Mikolov, T., Grave, E., Bojanowski, P., Puhrsch, C., & Joulin, A. (2018). *Advances in pre-training distributed word representations.* LREC. arXiv:1712.09405 | Model 2: `fasttext-wiki-news-subwords-300` (Wikipedia 2017, UMBC, statmt.org news, with sub-word information, 300 dimensions) |

The pre-trained vectors themselves are loaded through gensim-data (https://github.com/piskvorky/gensim-data) and are not stored here.

## 3. Measures we use (no separate file)

- Score per group = mean cosine similarity to positive words minus mean cosine similarity to negative words. The relative-closeness idea follows the word-embedding association approach cited in the paper (Caliskan et al., 2016).
- The positive/negative and competent/incompetent word lists are from the project brief and are listed in `forge_bias.py` (`CONCEPTS`).
- Gap = highest minus lowest group score within a category. Changes under 0.005 are called "no change" (our own cutoff, not a statistical test).

## Not included

- The project's internal notes (e.g. the "Additional notes on AI alignment" document in Downloads) were not used as a source for the analysis and are not copied here.

> Note for this repository: the paper and method PDFs listed above are kept out of the repo (publisher/arXiv copies); fetch them from the links given. The two source CSVs are included.
