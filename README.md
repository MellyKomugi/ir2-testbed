# IR 2 shared testbed

The shared testbed of the [IR 2 seminar](https://gerdes.fr/inforet/#6-the-shared-testbed). Every method is scored on the same data by the same script, so the class builds one comparable table: **[RESULTS.md](RESULTS.md)**.

## Start

Python 3.10 or later.

```
git clone https://github.com/kimgerdes/ir2-testbed
cd ir2-testbed
pip install -r requirements.txt
python testbed.py baselines --systems bm25 minilm
```

The first run downloads the data and the model. Each line should end with `✓ same as baselines/scores.json`. To run all five baselines, use `python testbed.py baselines`.

If your paper involves training, run the starter notebook once: [open it in Colab](https://colab.research.google.com/github/kimgerdes/ir2-testbed/blob/main/starter-contrastive-training.ipynb). It fine-tunes BGE-small with in-batch and hard negatives and measures alignment and uniformity.

## What is in it

| Dataset | Documents | Queries | What it stresses |
|---|:-:|:-:|---|
| SciFact | 5,183 | 300 | scientific claims with high word overlap |
| NFCorpus | 3,633 | 323 | lay medical queries against abstracts |
| ArguAna | 8,674 | 1,406 | the query is an argument, the target its counter-argument |

Baselines:

- `bm25`: BM25, k1 = 1.2, b = 0.75, Porter stems without stopwords
- `minilm`: `all-MiniLM-L6-v2`
- `bge`: `BAAI/bge-small-en-v1.5`, with its query prefix
- `rrf`: reciprocal rank fusion of BM25 and MiniLM, k = 60
- `ce`: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking the BM25 top 100, on the whole document. The spring notebook cut documents to 512 characters, which costs 0.086 on SciFact: check how your baselines are set up too

Documents are title + abstract, first 200 words, as in the spring course. Metrics are nDCG@10 and Recall@100 from `pytrec_eval`, and tied scores keep the order of your run. On ArguAna a query's own document is removed before scoring, as in BEIR.

## Add your method

1. Copy `pairs/_template` to `pairs/<paper number>-<short name>`, for example `pairs/15-hyde`.
2. Edit `method.py` so that it writes one run per dataset to `runs/`. A run is `{query id: {document id: score}}`, with at least the top 100 per query.
   - A new encoder: replace the model.
   - A change to the query: change `queries` before searching.
   - A reranker: start from `testbed.baseline_run(name, "bm25")` and give the same documents your own scores.
3. Score it:

   ```
   python testbed.py score pairs/15-hyde/runs --method "HyDE + BGE-small" --pair 15-hyde --against bge
   ```

   This writes `results.json` next to `runs/`. `--against` names the closest baseline, which is the one your method changes. Without it, each dataset is compared with its best baseline.
4. Open a pull request by Wednesday 18:00 before your session. Run files stay on your machine. The table rebuilds itself once the pull request is merged.

**Training:** you may learn from corpus text, but never from the test queries or `qrels`: they are the test labels.

**A paper that does not produce a ranking** (evaluation, security, RAG generation) has no row and needs no `results.json`. Put your numbers in your README, and say in your report which testbed number your method changes and by how much.

See `pairs/00-beyond-abstracts` for a complete example.

## Your folder

| File | What it is |
|---|---|
| `README.md` | how to run, in at most 10 lines |
| `reproduce.ipynb` | the reproduction of one key result |
| `method.py` | your method on the testbed |
| `results.json` | written by `testbed.py score` |
| `demo.md` | the three demo queries and the link to the fallback recording |
| `slides.pdf`, `report.pdf` | the talk and the report |

## `results.json`

```json
{
 "pair": "15-hyde",
 "method": "HyDE + BGE-small",
 "datasets": {
  "scifact": {
   "n_queries": 300,
   "ndcg10": 0.0,
   "recall100": 0.0,
   "baseline": {"name": "bge", "ndcg10": 0.0, "recall100": 0.0},
   "ndcg10_vs_baseline": {"delta": 0.0, "p": 0.0, "ci95": [0.0, 0.0]},
   "recall100_vs_baseline": {"delta": 0.0, "p": 0.0, "ci95": [0.0, 0.0]}
  }
 }
}
```

`delta` is your score minus the baseline's, `p` comes from a paired t-test over queries, and `ci95` is a bootstrap interval. A +0.01 with p = 0.4 is noise, not a win.
