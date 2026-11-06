"""IR 2 shared testbed: the spring baselines on SciFact, NFCorpus and ArguAna, scored one way for everybody.
nDCG@10 and Recall@100 with pytrec_eval (graded qrels; ArguAna without the query's own document, as in BEIR),
paired t-test and bootstrap 95 % CI of the per-query difference against a baseline.

Usage: python testbed.py baselines [--systems bm25 minilm] [--datasets scifact]   → run the baselines, compare with baselines/scores.json
       python testbed.py score <run dir> --method "My method" --pair 15-hyde [--against bge] [--out results.json]
A run is <run dir>/<dataset>.json: {query id: {document id: score}}, at least the top 100 per query."""
import argparse, json, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SCORES = ROOT / "baselines" / "scores.json"
BASE_RUNS = ROOT / "runs" / "baselines"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DATASETS = ["scifact", "nfcorpus", "arguana"]
SYSTEMS = ["bm25", "minilm", "bge", "rrf", "ce"]
LABELS = {"bm25": "BM25", "minilm": "all-MiniLM-L6-v2", "bge": "bge-small-en-v1.5",
          "rrf": "RRF (BM25 + MiniLM)", "ce": "BM25 top-100 + cross-encoder"}
K, MAX_WORDS, RRF_K = 100, 200, 60
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def device():
    """A GPU if there is one, else the CPU: Apple's MPS backend runs out of memory at these batch sizes."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    """Downloads the dataset once. Returns corpus {id: {"title", "text"}}, queries {id: text} and qrels {id: {doc: grade}}."""
    folder = DATA / name
    if not (folder / "corpus.jsonl").exists():
        DATA.mkdir(exist_ok=True)
        archive = DATA / f"{name}.zip"
        print(f"downloading {name}", file=sys.stderr)
        urllib.request.urlretrieve(URL.format(name), archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA)
        archive.unlink()
    corpus, queries, qrels = {}, {}, {}
    with open(folder / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    with open(folder / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    with open(folder / "qrels" / "test.tsv") as f:
        next(f)
        for line in f:
            q, d, grade = line.split("\t")
            qrels.setdefault(q, {})[d] = int(grade)
    return corpus, {q: queries[q] for q in qrels}, qrels


def doc_texts(corpus):
    """The spring text of each document: title + abstract, first 200 words."""
    return [" ".join((d["title"] + " " + d["text"]).split()[:MAX_WORDS]) for d in corpus.values()]


def bm25_search(name, corpus, queries):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("stopwords", quiet=True)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    def preprocess(t): return [porter.stem(x) for x in analyze(t) if x not in sw]
    ids = list(corpus)
    bm25 = BM25Okapi([preprocess(t) for t in doc_texts(corpus)], k1=1.2, b=0.75)
    run = {}
    for q, text in queries.items():
        s = bm25.get_scores(preprocess(text))
        run[q] = {ids[j]: float(s[j]) for j in np.argsort(-s)[:K + 1]}
    return run


def dense_search(model, corpus, queries, texts=None, query_prefix="", batch_size=64):
    """Any sentence-transformers model: encodes `texts` (default: the spring text), ranks by cosine."""
    ids, qids = list(corpus), list(queries)
    D = model.encode(texts or doc_texts(corpus), batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    Q = model.encode([query_prefix + queries[q] for q in qids], batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    S = Q @ D.T
    top = np.argsort(-S, axis=1)[:, :K + 1]
    return {q: {ids[j]: float(S[i, j]) for j in top[i]} for i, q in enumerate(qids)}


def rrf(runs):
    fused = {}
    for q in runs[0]:
        s = {}
        for run in runs:
            for rank, d in enumerate(sorted(run[q], key=run[q].get, reverse=True)[:K], 1):
                s[d] = s.get(d, 0.0) + 1.0 / (RRF_K + rank)
        fused[q] = dict(sorted(s.items(), key=lambda x: -x[1])[:K + 1])
    return fused


def rerank(model, corpus, queries, candidates):
    """The spring cross-encoder on the whole document. The spring cut to 512 characters costs 0.086 on SciFact."""
    texts = dict(zip(corpus, doc_texts(corpus)))
    run = {}
    for q, text in queries.items():
        cand = sorted(candidates[q], key=candidates[q].get, reverse=True)[:K]
        s = model.predict([[text, texts[d]] for d in cand], batch_size=64, show_progress_bar=False)
        run[q] = {d: float(x) for d, x in zip(cand, s)}
    return run


def save_run(run, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(run, open(path, "w"))


def baseline_run(name, system):
    """A baseline's run, computed once and kept in runs/baselines: reranking papers start from baseline_run(name, "bm25")."""
    path = BASE_RUNS / name / f"{system}.json"
    if path.exists():
        return json.load(open(path))
    from sentence_transformers import CrossEncoder, SentenceTransformer
    corpus, queries, _ = load(name)
    if system == "bm25":
        run = bm25_search(name, corpus, queries)
    elif system == "minilm":
        run = dense_search(SentenceTransformer("all-MiniLM-L6-v2", device=device()), corpus, queries)
    elif system == "bge":
        run = dense_search(SentenceTransformer("BAAI/bge-small-en-v1.5", device=device()), corpus, queries, query_prefix=BGE_QUERY)
    elif system == "rrf":
        run = rrf([baseline_run(name, "bm25"), baseline_run(name, "minilm")])
    elif system == "ce":
        run = rerank(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device()), corpus, queries, baseline_run(name, "bm25"))
    else:
        sys.exit(f"unknown baseline {system}: choose from {SYSTEMS}")
    save_run(run, path)
    return run


def score_run(run, name):
    """Per-query scores of a run: (query ids, nDCG@10 array, Recall@100 array)."""
    import pytrec_eval
    corpus, queries, qrels = load(name)
    qids = list(queries)
    missing = [q for q in qids if q not in run]
    if missing:
        sys.exit(f"{name}: {len(missing)} of {len(qids)} queries missing from the run, e.g. {missing[:3]}")
    unknown = {d for q in qids for d in run[q] if d not in corpus}
    if unknown:
        sys.exit(f"{name}: {len(unknown)} document ids are not in the corpus, e.g. {sorted(unknown)[:3]}")
    trimmed = {}
    for q in qids:
        ranked = sorted(run[q], key=run[q].get, reverse=True)
        if name == "arguana":                     # the query is itself a document: it must not retrieve itself
            ranked = [d for d in ranked if d != q]
        trimmed[q] = {d: float(K - i) for i, d in enumerate(ranked[:K])}   # ranks: pytrec_eval would re-sort ties by id
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in qids}, {"ndcg_cut.10", "recall.100"})
    s = evaluator.evaluate(trimmed)
    return qids, np.array([s[q]["ndcg_cut_10"] for q in qids]), np.array([s[q]["recall_100"] for q in qids])


def compare(a, b, seed=0):
    """Difference a − b over the same queries: mean, paired t-test, bootstrap 95 % CI."""
    from scipy import stats
    d = a - b
    boots = np.random.default_rng(seed).choice(d, size=(10000, len(d))).mean(axis=1)
    return {"delta": round(float(d.mean()), 4), "p": float(stats.ttest_rel(a, b).pvalue),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}


def stored():
    return json.load(open(SCORES)) if SCORES.exists() else {}


def cmd_baselines(args):
    table, kept = {}, stored()
    for name in args.datasets:
        for system in args.systems:
            t0 = time.time()
            qids, nd, rc = score_run(baseline_run(name, system), name)
            old = kept.get(name, {}).get(system)
            check = ""
            if old:
                check = "  ✓ same as baselines/scores.json" if abs(np.mean(old["ndcg10"]) - nd.mean()) < 0.002 \
                    else f"  ✗ baselines/scores.json has {np.mean(old['ndcg10']):.3f}"
            print(f"{name:9} {LABELS[system]:30} nDCG@10 {nd.mean():.3f}  Recall@100 {rc.mean():.3f}  {time.time() - t0:5.0f}s{check}", flush=True)
            entry = kept.setdefault(name, {"qids": qids})
            entry[system] = {"ndcg10": nd.round(6).tolist(), "recall100": rc.round(6).tolist()}
    if args.update:
        SCORES.parent.mkdir(exist_ok=True)
        json.dump(kept, open(SCORES, "w"))


def cmd_score(args):
    kept, rows = stored(), {}
    for name in DATASETS:
        path = Path(args.run_dir) / f"{name}.json"
        if not path.exists():
            print(f"{name}: no run file, skipped")
            continue
        qids, nd, rc = score_run(json.load(open(path)), name)
        base = kept[name]
        systems = [s for s in SYSTEMS if s in base]
        against = args.against or max(systems, key=lambda s: np.mean(base[s]["ndcg10"]))
        if base["qids"] != qids:
            sys.exit(f"{name}: query order differs from baselines/scores.json")
        bn, br = np.array(base[against]["ndcg10"]), np.array(base[against]["recall100"])
        cn, cr = compare(nd, bn), compare(rc, br)
        rows[name] = {"n_queries": len(qids), "ndcg10": round(float(nd.mean()), 4), "recall100": round(float(rc.mean()), 4),
                      "baseline": {"name": against, "ndcg10": round(float(bn.mean()), 4), "recall100": round(float(br.mean()), 4)},
                      "ndcg10_vs_baseline": cn, "recall100_vs_baseline": cr}
        print(f"{name:9} nDCG@10 {nd.mean():.3f} vs {against} {bn.mean():.3f}: {cn['delta']:+.3f} {cn['ci95']} p={cn['p']:.3g}"
              f" · Recall@100 {rc.mean():.3f} vs {br.mean():.3f}")
    out = Path(args.out or Path(args.run_dir).parent / "results.json")
    json.dump({"pair": args.pair, "method": args.method, "datasets": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("baselines")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    p.add_argument("--update", action="store_true", help="rewrite baselines/scores.json (instructor)")
    p.set_defaults(run=cmd_baselines)
    p = sub.add_parser("score")
    p.add_argument("run_dir")
    p.add_argument("--method", required=True)
    p.add_argument("--pair", required=True, help="the folder name in pairs/, e.g. 15-hyde")
    p.add_argument("--against", choices=SYSTEMS, help="the closest baseline; default: the best one on each dataset")
    p.add_argument("--out", help="default: results.json next to the run folder")
    p.set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)
