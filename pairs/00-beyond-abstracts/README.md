# Beyond Abstracts: Learning Scientific Paper Embeddings from Full-Text Windows

Worked example of a pair folder: the row from the model presentation (Djemmal, Olutola & Gerdes 2026). `method.py` uses a checkpoint that is not public, so read this one rather than run it.

The row is a transfer test. The trained model encodes each document as one vector, and nothing is retrained: the testbed has neither citations nor full texts, so it cannot test the paper's claim about body-passage queries. It is compared with `bge`, the closest dense baseline. The slide compares it with BM25: `--against bm25` gives +0.040 on SciFact, a gain that disappears against `bge` and `rrf`.

```
python pairs/00-beyond-abstracts/method.py <model dir>
python testbed.py score pairs/00-beyond-abstracts/runs --method "Beyond Abstracts (ModernBERT, both_random)" --pair 00-beyond-abstracts --against bge
```
