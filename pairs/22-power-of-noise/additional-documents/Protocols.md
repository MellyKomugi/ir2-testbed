# Experimental protocols — *The Power of Noise* (IR 2 seminar)

Paper: Cuconasu et al., *The Power of Noise: Redefining Retrieval for RAG Systems*, SIGIR 2024.
Original code and data: <https://github.com/florin-git/The-Power-of-Noise>

The project has two parts:

1. **Reproduction** of the paper's key result on NQ-open, at small scale, in our own notebook (Protocol 1).
2. **Application** of the paper's heuristic to the seminar's shared testbed, Track A (Protocol 2).

An optional complementary experiment on MS MARCO is described in Protocol 3.

Items marked **[TO VERIFY]** must be checked in the original code or data, or confirmed by the instructor, before running.

---

## 0. Shared conventions

### Generator (LLM)

A single model is used throughout, so that the reproduction and the testbed results can be compared.

| Setting | Value |
| --- | --- |
| Model | `meta-llama/Llama-2-7b-chat-hf` (gated: requires a Hugging Face account, acceptance of the Meta license and an access token for download) |
| Quantization | 4-bit with `bitsandbytes`, as in the paper **[TO VERIFY: exact settings in the original code]** |
| Decoding | Greedy, max 15 new tokens (reproduction); max 50 new tokens on the testbed to allow citations |
| Prompting | Zero-shot: task instruction, then documents, then the question last |

### Hardware

| Resource | Requirement |
| --- | --- |
| GPU | NVIDIA RTX 5060, 8 GB VRAM: enough for a 7B model in 4 bits (~4 GB) plus prompts of up to 4,096 tokens, with small batches |
| Software | Recent PyTorch and `bitsandbytes` builds supporting the Blackwell architecture **[TO VERIFY at installation]** |
| Disk | ~20 GB free: full model weights (~13 GB, downloaded before quantization), corpus subset, testbed data and outputs |
| RAM | 8 GB: tight when loading the model; other applications closed |
| Fallback | Google Colab (T4) if local generation fails or is too slow |

If disk space is insufficient, a pre-quantized 4-bit checkpoint of the same model may be used; this is a deviation from the paper and is reported as such.

### Document types

| Type | Definition (paper) | Definition (testbed) |
| --- | --- | --- |
| Gold / relevant | Passage containing the answer | Document with a positive label in the qrels |
| Distracting | Highly ranked by the retriever but does not contain the answer | Document in the retriever's top-10 that is not in the qrels |
| Random | Sampled from the corpus | Sampled uniformly from the corpus, excluding the qrels and the retriever's top-100 |
| Retrieved | Retriever's top-k (may be relevant or distracting) | Retriever's top-k from our ranking |

### Prompt layout

The paper's realistic setting places random documents first and retrieved documents next to the question:

```
[Instruction] [random documents] [retrieved documents] [Question]
```

- **Reproduction:** we follow the order used in the original code. **[TO VERIFY: order inside the retrieved block when `put_retrieved_first` is `False`.]**
- **Testbed:** the top-ranked document is placed **closest to the question** (reversed order), since the paper shows that relevant documents are best exploited near the question. The direct order (top-1 first) is run as an ablation.

### Reproducibility

- Fixed random seed for query sampling and random-document sampling, reported with the results.
- All per-query outputs (query id, prompt, raw generation, parsed answer, scores) are saved to disk.
- Each run is logged with model id, quantization, configuration and seed.

### Statistics

- Scores are computed **per query**, so that configurations can be compared with paired tests.
- Testbed: paired t-test and bootstrap 95 % confidence interval of the per-query difference, from the shared script (`testbed.py score`).
- Additionally, Wilcoxon signed-rank test, as used in the paper (p < 0.01).
- A difference that is not significant is reported as noise, not as a gain.

---

## Protocol 1 — Reproduction on NQ-open

### 1.1 Goal

Reproduce, at small scale, the paper's main result: **adding random documents to the retrieved documents improves LLM accuracy** (Table 3, Section 5.4). Secondary goal: the drop caused by distracting documents (Table 1). Success criterion: the **direction** of each effect is reproduced and significant; absolute values may differ because of the smaller sample.

### 1.2 Implementation

The reproduction is implemented in our own notebook (`reproduce.ipynb`) in the seminar repository, not by running the original scripts. We reuse the authors' **data** (questions, search results, corpus subset) and reimplement in the notebook:

- prompt construction (instruction and template of Figure 1 of the paper, document layout);
- generation with the LLM;
- scoring with the paper's accuracy.

For fidelity, the prompt template, the document layout and the accuracy function are ported from the original code (`src/generate_answers_llm_mixed.py`, `src/generate_answers_llm.py`, `src/read_generation_results.py`), and any difference is documented.

### 1.3 Data

| Resource | Source | Use |
| --- | --- | --- |
| NQ-open test split (2,889 questions) | Hugging Face `florin-hf/nq_open_gold`, split `test` | Main experiment |
| 10K sample of the train split | `data` folder of the original repository | Secondary experiment (oracle) |
| Search results (Contriever, BM25) | `data` folder / Google Drive of the original repository, e.g. `contriever_search_results_at150.pkl` | Retrieved and distracting documents |
| Corpus subsets | Google Drive of the original repository, `data/processed` | Text of the documents that can appear in prompts |

**Only the corpus subsets are downloaded, never the full corpus** (21,035,236 passages, ~25 GB in RAM). No indexing or retrieval is run for the reproduction.

Search results store **global indices** in the full corpus. Each subset keeps only the passages that can appear in the prompts of a given configuration, indexed by their original position, so that the notebook can look them up without the full corpus.

**[TO VERIFY before running:]**
- which subset file corresponds to each configuration (test split, retrieved + random; train sample, gold + distracting);
- that the subset contains the random documents (the README mentions random entries drawn from NQ), and how they were sampled;
- the mapping between global indices and subset entries.

### 1.4 Sample

- Main experiment: **500 questions** sampled from the test split with a fixed seed (the paper uses all 2,889).
- Secondary experiment: **500 questions** sampled from the 10K train sample (the paper uses all 10,000).

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

**Secondary experiment — gold + distracting (Table 1, "Near" setting)**

| Parameter | Values |
| --- | --- |
| Distracting documents (top-ranked by Contriever, without the answer) | 0, 1, 2, 4 |
| Gold position | Last (next to the question) |

4 configurations × 500 questions = 2,000 generations.

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

## Protocol 2 — Application to the testbed (Track A)

### 2.1 Goal

Test whether the paper's findings hold on another domain: do distracting documents hurt, and does random noise help, when the documents are scientific or medical abstracts rather than Wikipedia passages?

The method does not change the retriever's ranking: **nDCG@10 and Recall@100 are unchanged by construction** (delta = 0, reported in `results.json` with this explanation). The numbers that change are measured on the **LLM's output**, with the three scores of Section 2.6.

### 2.2 Data

| Dataset | Docs | Queries | Use |
| --- | --- | --- | --- |
| SciFact | 5,183 | 300 | Main dataset: all three scores |
| NFCorpus | 3,633 | 323 | Scores b and c |
| ArguAna | 8,674 | 1,406 (subsample of 300) | Scores b and c |

For score a on SciFact, the SUPPORTS / REFUTES / NOT ENOUGH INFO labels come from the original SciFact release (AllenAI), since they are not in the BEIR qrels. **[TO VERIFY with the instructor.]**

### 2.3 Retrieval: one ranking per dataset

The authors' search results cover NQ only, so a new ranking is generated for each Track A dataset:

| Retriever | How | Role |
| --- | --- | --- |
| **BM25** | Baseline script of the testbed repository | Default retriever for all prompts |
| Contriever (optional) | `facebook/contriever`, mean pooling of the last hidden states as in the paper, exact inner-product search (FAISS `IndexFlatIP`) | Same retriever as the paper |
| Dense (MiniLM), RRF (optional) | Baseline scripts of the testbed repository | Retriever ablation |

Each ranking keeps the **top-100** documents per query. Sanity check: the nDCG@10 and Recall@100 of our rankings must match the testbed baselines (e.g. BM25 on SciFact: nDCG@10 = 0.676).

### 2.4 Prompts

For each query, two prompts are built with the **same heuristic** (a few documents, then random noise):

| Prompt | Content | Role |
| --- | --- | --- |
| **Real** | Random documents + top-k of our BM25 ranking | What a deployed system would do; the qrels are not used to build it |
| **Ideal** | Random documents + relevant documents from the qrels (up to k) | Upper bound: what the LLM does with perfect retrieval |

| Parameter | Values |
| --- | --- |
| k (retrieved or relevant documents) | 3 (paper's recommended range: 3 to 5) |
| Random documents | 0, 3, 6 (bounded by the 4,096-token context of Llama2) |
| Order of the retrieved block | Reversed (top-1 next to the question); direct order as ablation |

Instruction (SciFact): answer SUPPORTS, REFUTES or NOT ENOUGH INFO, then list the documents used, most important first, in a fixed format (e.g. `Documents used: [4, 5]`).
Instruction (NFCorpus, ArguAna): answer the query from the documents, then list the documents used in the same format. **[TO VERIFY with the instructor: expected task on ArguAna.]**

SciFact: 2 prompts × 3 noise levels × 300 queries = 1,800 generations, plus 900 for the order ablation.

### 2.5 Input descriptors (not outcomes)

For each prompt, we record the number of relevant and distracting documents it contains and the nDCG@k of its retrieved block against the qrels. These describe the quality of the context; they do not change with noise and are used to explain the outcomes.

### 2.6 Outcome scores

**a. Answer (SciFact only).** The predicted label is compared with the ground-truth label (accuracy, 0/1). Without labels, the real prompt's answer is compared with the ideal prompt's answer (agreement, 0/1), keeping in mind that the ideal answer can also be wrong.

**b. Attribution.** The list of cited documents is scored against the qrels with **MRR** (is the first cited document relevant?) and **nDCG@k** (graded relevance where available, e.g. NFCorpus). This measures whether the LLM relies on the relevant documents it was given, not the retriever. Reported both over all queries and over queries whose prompt contains at least one relevant document, to separate retrieval failures from generation failures.

**c. Relative similarity.** The answer is embedded (encoder: `all-MiniLM-L6-v2`, available in the testbed). Success (1) if its maximum similarity to the **relevant** documents of the prompt is higher than its maximum similarity to the **distracting** documents of the prompt, computed over all documents of the prompt, not only the cited ones. Undefined when the prompt contains no relevant or no distracting document; these cases are counted and reported. Absolute similarity is not used, because an answer built from a distractor is also similar to the relevant document.

**Parsing.** An output with no valid label or no valid citation list scores 0 on the corresponding score; the number of invalid outputs is reported per configuration.

### 2.7 Comparisons

- Effect of noise: 0 vs 3 vs 6 random documents, for each prompt type.
- Cost of imperfect retrieval: real vs ideal prompt, at each noise level.
- Ablation: reversed vs direct order of the retrieved block.
- Optional: retriever = BM25 vs Contriever vs dense (MiniLM) vs RRF(BM25 + dense).
- Comparison with Protocol 1: **trends**, not absolute values (different tasks and metrics).

### 2.8 Reporting

- `results.json`: nDCG@10 and Recall@100 with delta = 0 and its explanation; scores a, b and c with paired t-test p-values and bootstrap CIs against the no-noise baseline.
- README: Wilcoxon p-values, input descriptors, invalid-output counts, ablations.

---

## Protocol 3 (optional) — MS MARCO as a QA set

Conditional on obtaining reference answers for the spring MS MARCO subset (e.g. from the MS MARCO question-answering release, if query ids match). **[TO VERIFY with the instructor.]**

If available: same design as Protocol 1 (one BM25 ranking generated on the subset, retrieved + random, reversed order), with substring accuracy as in the paper, on 300 to 500 queries. This experiment bridges the reproduction (NQ-open) and the testbed, and is reported in the README, not in the shared table.

---

## Compute plan

| Task | Where | Notes |
| --- | --- | --- |
| Data download and preparation | Laptop | Corpus subsets only; the full corpus is never downloaded |
| Rankings on Track A (BM25, optional Contriever) | Laptop | Corpora of 3,600 to 8,700 documents |
| Prompt construction, scoring, statistics | Laptop | CPU |
| LLM generation | Laptop GPU (RTX 5060), Colab T4 as fallback | ~11,200 generations in total (reproduction 8,500; SciFact 2,700), plus NFCorpus and ArguAna |

Before launching a full grid, one configuration is timed on 50 questions; the number of configurations is then fixed to fit the compute budget, and any reduction is reported.

---

## Pending questions for the instructor

1. Can the original SciFact labels be added to the testbed?
2. Is the three-score evaluation (answer, attribution, relative similarity) acceptable?
3. Which task and metric are expected for "a RAG loop over ArguAna"?
4. Is the MS MARCO experiment worth running?