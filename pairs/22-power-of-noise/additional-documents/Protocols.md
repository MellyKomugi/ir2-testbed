# Experimental protocols — *The Power of Noise* (IR 2 seminar)

Paper: Cuconasu et al., *The Power of Noise: Redefining Retrieval for RAG Systems*, SIGIR 2024.
Original code and data: <https://github.com/florin-git/The-Power-of-Noise>

The project has two parts:

1. **Reproduction** of the paper's key results on NQ-open, at small scale, in our own notebook (Protocol 1).
2. **Application** of the paper's oracle design to the shared testbed, on SciFact claim verification (Protocol 2).

An optional MS MARCO experiment (Protocol 3) is run only if time remains.

Items marked **[TO VERIFY]** must be checked in the original code or data before running.

---

## 0. Shared conventions

### Generator (LLM)

A single model is used throughout, so that the reproduction and the testbed results can be compared.

| Setting | Value |
| --- | --- |
| Model | `meta-llama/Llama-2-7b-chat-hf` (gated: requires a Hugging Face account, acceptance of the Meta license and an access token for download) |
| Quantization | 4-bit with `bitsandbytes`, as in the paper |
| Decoding | Greedy, max 15 new tokens (reproduction); max 50 new tokens on the testbed to allow citations |
| Prompting | Zero-shot: task instruction, then documents, then the question last |

### Hardware

| Resource | Requirement |
| --- | --- |
| GPU | NVIDIA RTX 5060, 8 GB VRAM: enough for a 7B model in 4 bits (~4 GB) plus prompts of up to 4,096 tokens, with small batches |
| Software | Recent PyTorch and `bitsandbytes` builds supporting the Blackwell architecture **[TO VERIFY at installation]** |
| Disk | ~20 GB free: full model weights (~13 GB, downloaded before quantization), data subsets and outputs |
| RAM | 8 GB: tight when loading the model; other applications closed |
| Fallback | Google Colab (T4) if local generation fails or is too slow |

If disk space is insufficient, a pre-quantized 4-bit checkpoint of the same model may be used; this is a deviation from the paper and is reported as such.

### Document types

| Type | Reproduction (NQ-open) | Testbed (SciFact) |
| --- | --- | --- |
| Gold | Annotated passage containing the answer | Relevant abstract of the claim (with its SUPPORT or CONTRADICT label) |
| Distracting / hard negative | Highly ranked by the retriever but without the answer | Highest-ranked documents of the BM25 (or dense) run, minus the relevant ones |
| Random | Sampled from the corpus | Sampled uniformly, excluding the relevant abstracts and the query's top-100 |
| Retrieved | Retriever's top-k (may be relevant or distracting) | Retriever's top-k from our BM25 run |

### Gold position

Following Section 5.2 of the paper, the gold document is placed **near** the question (last in the context) or **far** from it (first). The question is always last. In the paper's realistic setting (Table 3), random documents come first and retrieved documents next to the question:

```
[Instruction] [random documents] [retrieved documents] [Question]
```

**[TO VERIFY in the original code: the order inside the retrieved block when `put_retrieved_first` is `False`.]**

### Reproducibility

- Fixed random seed for query sampling and random-document sampling, reported with the results.
- All per-query outputs (query id, prompt, raw generation, parsed answer, scores) are saved to disk, incrementally, so that runs can resume after an interruption.
- Each run is logged with model id, quantization, configuration and seed.
- Output lengths and truncation rates (15-token cap) are logged to diagnose gaps with the paper.

### Statistics

- Scores are computed **per query**, so that conditions can be compared with paired tests.
- Paired t-test and bootstrap 95 % confidence interval of the per-query difference, with the tests of the shared script.
- Additionally, Wilcoxon signed-rank test, as used in the paper (p < 0.01).
- Comparisons between the reproduction and the testbed are made on **trends**, not on absolute values (different tasks and metrics).
- A difference that is not significant is reported as noise, not as a gain.

---

## Protocol 1 — Reproduction on NQ-open

### 1.1 Goal

Reproduce, at small scale, the paper's main results: **adding random documents to the retrieved documents improves LLM accuracy** (Table 3, Section 5.4), and the two oracle effects that the testbed mirrors, **distracting documents hurt** (Table 1) and **random documents help** (Table 2). Success criterion: the **direction** of each effect is reproduced and significant; absolute values may differ because of the smaller sample.

To keep in mind:
- The distractors chosen by a higher-performing retriever are more harmful, and the article explicitly states this. The impact does indeed depend on the retriever. What the article concludes, however, is that switching to a different dense retriever does not solve the problem: the accuracy remains well below 0.5642, the value of the gold set alone.

### 1.2 Implementation

The reproduction is implemented in our own notebook (`reproduce.ipynb`) in the seminar repository, not by running the original scripts. We reuse the authors' **data** and reimplement in the notebook:

- prompt construction (instruction and template of Figure 1 of the paper, document layout);
- generation with the LLM;
- scoring with the paper's accuracy.

For fidelity, the prompt template, the document layout and the accuracy function are ported from the original code (`src/generate_answers_llm_mixed.py`, `src/generate_answers_llm.py`, `src/read_generation_results.py`), and any difference is documented.

### 1.3 Data

| Resource | Source | Needed for |
| --- | --- | --- |
| NQ-open questions, each with its gold passage (test split: 2,889) | Hugging Face `florin-hf/nq_open_gold` | All experiments |
| 10K sample of the train split | `data` folder of the original repository | Tables 1 and 2 (oracle) |
| Search results (Contriever, BM25), as **global indices and scores only** | `data` folder / Google Drive of the original repository | Tables 1 and 3 |
| Corpus subsets (text of the passages that can appear in prompts) | Google Drive, `data/processed` | Tables 1 and 3 |

The full Wikipedia corpus (21,035,236 passages, ~25 GB in RAM) is never downloaded, and no indexing or retrieval is run: we reuse the authors' search results.

- **Table 2** (gold + random) needs only `nq_open_gold`: random documents can be drawn from the gold passages of other questions.
- **Tables 1 and 3** need the **text** of the distracting and retrieved passages, which the search results do not contain. They rely on the Drive subsets, which keep the passages' original global indices.

**[TO VERIFY before running:]**
- which subset file corresponds to each configuration (test split, retrieved + random; train sample, gold + distracting);
- that the subsets contain the random documents, and how they were sampled;
- the mapping between global indices and subset entries.

**Fallback if the subsets do not fit our configurations:** keep Table 2 (which needs no subset) and build a mini-corpus from the gold passages of `nq_open_gold`, on which BM25 is run locally to obtain retrieved and distracting passages (distractors are those without the answer). This is a deviation from the paper (smaller corpus) and is reported as such.

### 1.4 Sample

**500 questions** per experiment, sampled with a fixed seed (the paper uses 2,889 test questions for Table 3 and 10,000 train questions for Tables 1 and 2).

### 1.5 Configurations

**Main experiment — retrieved + random (Table 3)**
Random documents first, retrieved documents next to the question.

| Parameter | Values |
| --- | --- |
| Retriever (authors' search results) | Contriever, BM25 |
| Retrieved documents | 1, 4 |
| Random documents | 0, 3, 10 |

12 configurations × 500 questions = 6,000 generations.

**Baseline — closed-book**
Question only, no documents: 500 generations.

**Oracle — gold + distracting (Table 1, "Near" setting)**

| Parameter | Values |
| --- | --- |
| Distracting documents (top-ranked by Contriever, without the answer) | 0, 1, 2, 4 |
| Gold position | Near |

4 configurations × 500 questions = 2,000 generations.

**Oracle — gold + random (Table 2, "Near" setting)**
Mirrors the testbed (gold + random vs gold + hard negatives), so that trends can be compared one to one.

| Parameter | Values |
| --- | --- |
| Random documents | 2, 4 |
| Gold position | Near |

2 configurations × 500 questions = 1,000 generations (the 0-document case is shared with Table 1).

**Optional, if time allows:** gold position (near / mid / far).

### 1.6 Metric

Accuracy, as in the paper: an output is correct if it contains at least one of the reference answers. Saved per query (0/1).

### 1.7 Reference values from the paper (Llama2-7B)

Table 3, retrieved + random (test set):

| Retriever | Retrieved | 0 random | 3 random | 10 random |
| --- | --- | --- | --- | --- |
| Contriever | 1 | 0.1620 | 0.1301 | 0.1796 |
| Contriever | 4 | 0.1866 | 0.2316 | 0.2502 |
| BM25 | 1 | 0.2008 | 0.1568 | 0.2108 |
| BM25 | 4 | 0.2028 | 0.2520 | 0.2835 |

Table 1, gold + distracting, "Near" (train sample): 0 → 0.5642, 1 → 0.4283, 2 → 0.3974, 4 → 0.3795.
Closed-book: 0.1123.

---

## Protocol 2 — Application to the testbed (SciFact)

### 2.1 Goal

Transpose the paper's oracle design to scientific claims: do hard negatives hurt, does random noise help, and does the position of the gold abstract matter, when the documents are scientific abstracts rather than Wikipedia passages?

The method does not change any retrieval number (**nDCG@10 and Recall@100 are unchanged by construction**), so `results.json` is not modified. The reader results (balanced accuracy per condition, with the tests) are reported in this README.

### 2.2 Data

SciFact only (300 test claims, 5,183 abstracts).

- The testbed's SciFact `queries.jsonl` keeps the original labels in its metadata field. **188 of the 300 claims** have SUPPORT or CONTRADICT on their relevant abstract.
- The other 112 were "not enough info" in the original release and are left out (a third class is possible as an extension).
- About two thirds of the labels are SUPPORT, so the metric is **balanced accuracy**, not raw accuracy.
- NFCorpus and ArguAna are not used: they have no answers or labels that fit this design.

**[TO VERIFY in the repo:** name of the metadata field; number of relevant abstracts per claim and how claims with several are handled.**]**

### 2.3 Retrieval

One BM25 ranking (top-100) per query, generated with the testbed baseline script. Sanity check: nDCG@10 = 0.676 and the Recall@100 of the testbed baselines. The dense run (MiniLM) is available for an ablation on the source of hard negatives.

### 2.4 Prompts and conditions

For each of the 188 claims, the context is built around its **gold abstract**:

| Condition | Context |
| --- | --- |
| Closed-book | Claim only (floor) |
| Gold only | Gold abstract |
| Gold + random | Gold abstract + k random abstracts |
| Gold + hard negatives | Gold abstract + k hard negatives |

| Parameter | Values |
| --- | --- |
| k | 2, 4, 8 (bounded by the 4,096-token context: ~250 tokens per abstract) |
| Gold position | Near (last, next to the claim) and far (first) |

2 + 2 × 3 × 2 × 2 = 14 conditions × 188 claims = **2,632 generations**.

**Optional extension (Table 3 transposed):** real retrieval, i.e. BM25 top-3 + random noise placed first, to test the heuristic with a retriever that can fail.

**Instruction.** Answer exactly `SUPPORT` or `CONTRADICT`, then list the documents used, most important first, in a fixed format (`Documents used: [4, 5]`). An output with no valid label scores 0; the number of invalid outputs is reported per condition.

### 2.5 Scores

**Main: balanced accuracy**, the mean of the recall on SUPPORT and on CONTRADICT (chance = 0.5). To obtain per-query scores for paired tests, each query's correctness (0/1) is weighted by 1 / (2 × number of claims of its class), so that the mean equals the balanced accuracy.

**Attribution:** does the reader cite the gold abstract, and does a distractor pull the citation away from it?

- gold cited (0/1) and **MRR** of the cited list (1 / rank of the gold abstract, 0 if absent); nDCG@k is also computed (with a single gold abstract it behaves like MRR);
- rate of queries where a distractor is cited before the gold.

### 2.6 Comparisons

- **Effect of noise:** gold + random vs gold only, for each k and position.
- **Effect of hard negatives:** gold + hard negatives vs gold + random, same k and position.
- **Position:** near vs far.
- **Optional:** hard negatives from the dense run instead of BM25.
- **Comparison with Protocol 1:** Table 1 ↔ hard negatives, Table 2 ↔ random, on trends.

The bootstrap recomputes the balanced accuracy on each resample; the Wilcoxon test is applied to per-query correctness.

### 2.7 Limits

- 188 claims give low statistical power: small effects will not be significant, and confidence intervals are reported.
- A hard negative can be a relevant abstract that the qrels do not label.
- **Pilot first:** if the balanced accuracy of the gold-only condition is close to 0.5 on 50 claims, the model cannot do the task and the experiment cannot discriminate between conditions. In that case, the choice of model is discussed with the instructor.

---

## Protocol 3 (optional) — MS MARCO

Only if time remains. The MS MARCO subset is not in the testbed repository, and reference answers would have to be retrieved from the question-answering release **[TO VERIFY]**. If run: Protocol 1 design on 300 to 500 queries, reported in this README.

---

## Compute plan

| Task | Where | Notes |
| --- | --- | --- |
| Data download and preparation | Laptop | `nq_open_gold` and the corpus subsets only |
| BM25 on SciFact, prompt construction, scoring, statistics | Laptop | CPU |
| LLM generation | Laptop GPU (RTX 5060), Colab T4 as fallback | See below |

| Part | Generations | Estimated time |
| --- | --- | --- |
| Reproduction | ~9,500 (6,000 Table 3 + 2,000 Table 1 + 1,000 Table 2 + 500 closed-book) | 3 to 7 h |
| Testbed | ~2,600 | 2 to 4 h |

Before launching a full grid, one configuration is timed on 50 questions, and the number of configurations is fixed to fit the compute budget; any reduction is reported.

---

## Points to verify before running

1. Corpus subsets on the authors' Drive: which file matches each configuration, whether it contains the random documents, and the mapping between global indices and subset entries.
2. Order of the documents inside the retrieved block in the original code.
3. SciFact metadata: field name and handling of claims with several relevant abstracts.
4. Installation of PyTorch and `bitsandbytes` builds supporting the RTX 5060.