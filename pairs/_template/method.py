"""Our method on the shared testbed. Writes one run per dataset to runs/, then:
python testbed.py score pairs/<our folder>/runs --method "<method>" --pair <our folder> --against <closest baseline>"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed

# A new encoder: replace the model. A new query: change `queries`. A reranker: start from
# testbed.baseline_run(name, "bm25") and return the same documents with your scores.
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=testbed.device())
for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name)      # never tune on qrels: they are the test labels
    run = testbed.dense_search(model, corpus, queries, query_prefix=testbed.BGE_QUERY)
    testbed.save_run(run, HERE / "runs" / f"{name}.json")
