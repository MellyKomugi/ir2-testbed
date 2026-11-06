"""Worked example: the instructor's model presentation as a testbed row.
Usage: python pairs/00-beyond-abstracts/method.py <model dir>    (the checkpoint is not public: read this one, don't run it)
Then:  python testbed.py score pairs/00-beyond-abstracts/runs --method "Beyond Abstracts (ModernBERT, both_random)" \
         --pair 00-beyond-abstracts --against bge"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed

model = SentenceTransformer(sys.argv[1], device=testbed.device())
model.max_seq_length = 512                               # abstracts and arguments fit
for name in testbed.DATASETS:
    corpus, queries, qrels = testbed.load(name)
    texts = [(d["title"] + ". " + d["text"]).strip(". ") for d in corpus.values()]
    run = testbed.dense_search(model, corpus, queries, texts=texts, batch_size=16)
    testbed.save_run(run, HERE / "runs" / f"{name}.json")
