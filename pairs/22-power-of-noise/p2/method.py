"""Protocol 2: passage corpus, BM25 and BGE runs, per-claim draws, gold windows,
prompts, token dry-run, and the D14 pilot generation.
python method.py  downloads the original SciFact release into data/ on first run,
rebuilds the run files, draws.json and dry_run.txt, and asserts the frozen control
numbers. The JSON outputs are never overwritten: a recomputation must give back the
same content, or the script stops (the BGE run, not bit-reproducible across machines,
is computed once then read back); dry_run.txt, a report, is rewritten each run.
python method.py --pilot  runs the 50-claim pilot on the GPU machine (both models);
python method.py --pilot-report  parses the generations back home.
The full grid generation comes in a later pass."""
import bisect
import hashlib
import json
import os
import random
import re
import sys
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

# ---- CONFIG

HERE = os.path.dirname(os.path.abspath(__file__))
TESTBED = os.path.join(HERE, "..", "..", "..")
TITLES_FILE = os.path.join(TESTBED, "data/scifact/corpus.jsonl")
QUERIES_FILE = os.path.join(TESTBED, "data/scifact/queries.jsonl")
QRELS_FILE = os.path.join(TESTBED, "data/scifact/qrels/test.tsv")
DATA_DIR = os.path.join(HERE, "data")
DATA_FILES = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"]
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

sys.path.insert(0, TESTBED)
import testbed

RUN_FILE = os.path.join(HERE, "bm25_run.json")
BEST_FILE = os.path.join(HERE, "bm25_best_passage.json")
BGE_RUN_FILE = os.path.join(HERE, "bge_run.json")
BGE_BEST_FILE = os.path.join(HERE, "bge_best_passage.json")
DRAWS_FILE = os.path.join(HERE, "draws.json")
DRY_FILE = os.path.join(HERE, "dry_run.txt")
PILOT_FILE = os.path.join(HERE, "draws_pilot.json")
GEN_DIR = os.path.join(HERE, "generations")

MAX_WORDS = 100        # word budget of one prompt document
MIN_WORDS = 5          # drops the corpus's one single-word passage ("...")
K1 = 1.2               # BM25 settings of the testbed
B = 0.75
RUN_DEPTH = 101        # the testbed keeps 100 + 1 documents per query
N_DOCS = 18            # frozen at 18 (the grid stops at 14): redrawing with less would change the file
SEED = 10
GEN_TOKENS = 30        # answer budget: bare greedy, no stop string (D16)
INSTRUCTION_ID = "d5"  # the frozen core instruction; E3 variants will get their own id
N_PILOT = 50

# ---- DATA

def download_data():
    # the original SciFact release: sentence-split abstracts and annotated claims
    # (the testbed corpus only keeps a joined text field)
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILES[-1])):
        return
    import tarfile
    import urllib.request
    print("downloading the SciFact release", file=sys.stderr)
    archive = os.path.join(HERE, "data.tar.gz")
    urllib.request.urlretrieve(SCIFACT_URL, archive)
    os.makedirs(DATA_DIR, exist_ok=True)
    with tarfile.open(archive) as tar:
        for name in DATA_FILES:        # only the files we read, written where we decide
            with open(os.path.join(DATA_DIR, name), "wb") as f:
                f.write(tar.extractfile("data/" + name).read())
    os.remove(archive)


def load_corpus():
    doc_ids = []
    titles = {}
    with open(TITLES_FILE) as f:
        for line in f:
            d = json.loads(line)
            doc_ids.append(d["_id"])
            titles[d["_id"]] = d.get("title", "")
    sentences = {}
    with open(os.path.join(DATA_DIR, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            sentences[str(d["doc_id"])] = d["abstract"]
    assert len(doc_ids) == 5183
    assert all(doc in sentences for doc in doc_ids)
    return doc_ids, titles, sentences


def load_queries():
    qrels = {}
    with open(QRELS_FILE) as f:
        next(f)
        for line in f:
            qid, doc, grade = line.split("\t")
            qrels.setdefault(qid, {})[doc] = int(grade)
    queries = {}
    with open(QUERIES_FILE) as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    assert len(queries) == 300
    return queries, qrels


def load_claims(split):
    # golds = the abstracts with annotated evidence, NOT all the cited abstracts:
    # some claims also cite abstracts the annotators judged evidence-free
    claims = {}
    with open(os.path.join(DATA_DIR, f"claims_{split}.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            if not c["evidence"]:
                continue
            labels = set()
            for entries in c["evidence"].values():
                for e in entries:
                    labels.add(e["label"])
            assert len(labels) == 1
            cited = set()
            for doc in c["cited_doc_ids"]:
                cited.add(str(doc))
            claims[str(c["id"])] = {"text": c["claim"], "label": labels.pop(),
                                    "golds": list(c["evidence"]), "cited": cited,
                                    "evidence": c["evidence"]}
    return claims


def sister_evidence(claims):
    # a sister cites at least one same abstract (original train/dev); its annotated
    # evidence can prove our claim too (paraphrases, negations), so the draws skip it
    evidence = {}
    cited_by = {}
    for split in ("train", "dev"):
        with open(os.path.join(DATA_DIR, f"claims_{split}.jsonl")) as f:
            for line in f:
                c = json.loads(line)
                cid = str(c["id"])
                for doc in c["cited_doc_ids"]:
                    cited_by.setdefault(str(doc), set()).add(cid)
                evidence[cid] = set(c["evidence"])
    sisters = {}
    for qid in claims:
        family = set()
        for doc in claims[qid]["cited"]:
            family |= cited_by[doc]
        family.discard(qid)
        proofs = set()
        for sid in family:
            proofs |= evidence[sid]
        sisters[qid] = proofs - claims[qid]["cited"]
    return sisters

# ---- RETRIEVAL

try:
    sw = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    sw = set(stopwords.words("english"))
porter = PorterStemmer()
analyze = CountVectorizer().build_analyzer()


def preprocess(t):
    # the testbed's own preprocessing line
    return [porter.stem(x) for x in analyze(t) if x not in sw]


def spans(words):
    out = []
    last = -1
    for i in range(len(words)):
        end = i
        total = words[i]
        while end + 1 < len(words) and total + words[end + 1] <= MAX_WORDS:
            end += 1
            total += words[end]
        if end > last:                 # otherwise contained in the previous window
            out.append((i, end + 1, total))
            last = end
    return out


def build_passages(doc_ids, sentences):
    passages = []
    windows = []
    for j, doc in enumerate(doc_ids):
        words = [len(s.split()) for s in sentences[doc]]
        kept = []
        for a, b, total in spans(words):
            if MIN_WORDS <= total <= MAX_WORDS:
                kept.append((a, b))
        windows.append(kept)
        for a, b in kept:
            passages.append((j, a, b))
    assert all(windows)                # every abstract keeps at least one passage
    return passages, windows


def passage_text(titles, sentences, doc, a, b):
    # corpus sentences often end with spaces and a line break: keep single spaces
    text = titles[doc] + " " + " ".join(sentences[doc][a:b])
    return " ".join(text.split())


def split_id(pid):
    # passage ids are "doc:a-b", sentences [a, b) with b excluded
    doc, span = pid.split(":")
    a, b = span.split("-")
    return doc, int(a), int(b)


def build_index(passages, doc_ids, titles, sentences):
    texts = []
    for j, a, b in passages:
        texts.append(preprocess(passage_text(titles, sentences, doc_ids[j], a, b)))
    return BM25Okapi(texts, k1=K1, b=B)


def rank_abstracts(scores, passages):
    # one abstract = its best passage, whatever engine scored them;
    # strict > keeps the leftmost window on ties
    best_score = {}
    best_window = {}
    for p, (j, a, b) in enumerate(passages):
        if j not in best_score or scores[p] > best_score[j]:
            best_score[j] = scores[p]
            best_window[j] = (a, b)
    # tied abstracts: corpus order, so the files do not depend on the machine
    order = sorted(best_score, key=lambda j: (-best_score[j], j))
    return order, best_score, best_window


def run_entry(order, best_score, best_window, doc_ids):
    # the two dictionaries written per query: abstract scores and best passage ids
    scores_out = {}
    ids_out = {}
    for j in order[:RUN_DEPTH]:
        a, b = best_window[j]
        scores_out[doc_ids[j]] = float(best_score[j])
        ids_out[doc_ids[j]] = f"{doc_ids[j]}:{a}-{b}"
    return scores_out, ids_out


def dense_run(passages, doc_ids, titles, sentences, queries):
    # the testbed's bge settings (cosine, normalized, query prefix), but on the
    # passages and on the full matrix: testbed.dense_search encodes title+200 words
    # and keeps 101 rows, which would truncate before MaxP
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    texts = []
    for j, a, b in passages:
        texts.append(passage_text(titles, sentences, doc_ids[j], a, b))
    D = model.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
    qids = list(queries)
    Q = model.encode([testbed.BGE_QUERY + queries[q] for q in qids], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    run = {}
    best_pass = {}
    for i, qid in enumerate(qids):
        scores = Q[i] @ D.T
        order, best_score, best_window = rank_abstracts(scores, passages)
        run[qid], best_pass[qid] = run_entry(order, best_score, best_window, doc_ids)
    return run, best_pass


def evaluate(run):
    # scored by the shared tool of the course
    qids, ndcg, recall = testbed.score_run(run, "scifact")
    return ndcg.mean(), recall.mean()

# ---- SAMPLING

def count_blocks(words):
    n = 0
    i = 0
    while i < len(words):
        total = words[i]
        while i + 1 < len(words) and total + words[i + 1] <= MAX_WORDS:
            i += 1
            total += words[i]
        n += 1
        i += 1
    return n


def block_scale(doc_ids, sentences):
    cum = []
    total = 0
    for doc in doc_ids:
        words = [len(s.split()) for s in sentences[doc]]
        total += count_blocks(words)
        cum.append(total)
    return cum


def draw_randoms(cum, windows, doc_ids, banned, rng):
    # regime B: an abstract weighs its disjoint 100-word blocks, like a passage draw in
    # the paper's corpus; its window is drawn right after it (part of the frozen
    # sequence); no content filter beyond `banned`, faithful to the paper
    picks = []
    seen = set()
    while len(picks) < N_DOCS:
        j = bisect.bisect_right(cum, rng.random() * cum[-1])
        if j in seen or doc_ids[j] in banned:
            continue
        seen.add(j)
        a, b = windows[j][rng.randrange(len(windows[j]))]
        picks.append(f"{doc_ids[j]}:{a}-{b}")
    return picks


def pick_distractors(ranked, best_pass, golds, banned):
    # D19: remove the golds from the ranking FIRST, then take 18; sister evidence
    # is skipped like a gold
    picks = []
    for doc in ranked:
        if doc in golds or doc in banned:
            continue
        picks.append(best_pass[doc])
        if len(picks) == N_DOCS:
            break
    assert len(picks) == N_DOCS
    return picks


def write_once(path, data):
    # runs and draws are made once: a recomputation must give back the same bytes
    text = json.dumps(data)
    if os.path.exists(path):
        with open(path) as f:
            assert f.read() == text, path + " would change"
    else:
        with open(path, "w") as f:
            f.write(text)


def write_draws(path, draws):
    # same freeze, key by key: a key already in the file must not move (random, bm25);
    # a key the file does not have yet (bge) is added next to them
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(json.dumps(draws))
        return
    with open(path) as f:
        old = json.load(f)
    assert list(old) == list(draws), path + " claims would change"
    added = False
    for qid in draws:
        for key in draws[qid]:
            if key in old[qid]:
                assert old[qid][key] == draws[qid][key], path + " " + qid + " " + key + " would change"
            else:
                old[qid][key] = draws[qid][key]
                added = True
    if added:
        with open(path, "w") as f:
            f.write(json.dumps(old))

# ---- PROMPTS

INSTRUCTION = ("You are given scientific documents and a claim. Answer SUPPORT if the "
               "documents support the claim or CONTRADICT if they contradict it, then cite "
               "the document that justifies your answer, in the format: "
               "SUPPORT|CONTRADICT — Document [n].")
CLOSED_BOOK = ("You are given a claim. Answer SUPPORT if your knowledge supports the "
               "claim or CONTRADICT if it contradicts it.")


def gold_span(words, sets):
    # the window of whole sentences <= 100 words holding the MOST complete evidence
    # sets; ties: most compact set, then shortest window; then grown before, then after
    marks = []
    for s in sets:
        marks.append((min(s), max(s)))
    best_key = None
    best_start = None
    best_end = None
    for i in range(len(words)):
        total = 0
        for j in range(i, len(words)):
            total += words[j]
            if total > MAX_WORDS:
                break
            inside = []
            for a, b in marks:
                if a >= i and b <= j:
                    inside.append(sum(words[a:b + 1]))
            if inside:
                key = (-len(inside), min(inside), total)
                if best_key is None or key < best_key:
                    best_key = key
                    best_start = i
                    best_end = j
    if best_key is None:
        return None
    n_sets = -best_key[0]
    compact = best_key[1]
    a = best_start
    b = best_end
    total = sum(words[a:b + 1])
    while a > 0 and total + words[a - 1] <= MAX_WORDS:
        a -= 1
        total += words[a]
    while b < len(words) - 1 and total + words[b + 1] <= MAX_WORDS:
        b += 1
        total += words[b]
    return n_sets, compact, a, b + 1


def gold_window(claim, sentences):
    # D3: the gold whose best window holds the most complete sets; ties: most compact
    # set, then shortest abstract. If no set fits 100 words anywhere, widen to the most
    # compact set (claims 1110 and 1241 of the test, three more on the train)
    best_key = None
    best = None
    for doc, entries in claim["evidence"].items():
        words = [len(s.split()) for s in sentences[doc]]
        sets = []
        for e in entries:
            sets.append(sorted(e["sentences"]))
        found = gold_span(words, sets)
        if found is None:
            continue
        n_sets, compact, a, b = found
        key = (-n_sets, compact, sum(words), doc)
        if best_key is None or key < best_key:
            best_key = key
            best = (doc, a, b)
    if best:
        doc, a, b = best
        return doc, a, b, False
    widest_key = None
    widest = None
    for doc, entries in claim["evidence"].items():
        words = [len(s.split()) for s in sentences[doc]]
        for e in entries:
            a = min(e["sentences"])
            b = max(e["sentences"])
            key = (sum(words[a:b + 1]), doc, a, b)
            if widest_key is None or key < widest_key:
                widest_key = key
                widest = (doc, a, b + 1)
    doc, a, b = widest
    return doc, a, b, True


def gold_pids(claims, sentences):
    # one frozen passage id per claim for its gold window; widened = beyond 100 words
    pids = {}
    widened = []
    for qid in sorted(claims, key=int):
        doc, a, b, wide = gold_window(claims[qid], sentences)
        pids[qid] = f"{doc}:{a}-{b}"
        if wide:
            widened.append(qid)
    return pids, widened


def top3_lists(claims, run, best_pass, bge_run, bge_best):
    # the 3 best abstracts of each engine, shown by their best passage, in run order
    top3 = {}
    for qid in claims:
        top3[qid] = {}
        for engine, ranked, best in (("bm25", run, best_pass), ("bge", bge_run, bge_best)):
            picks = []
            for doc in list(ranked[qid])[:3]:
                picks.append(best[qid][doc])
            top3[qid][engine] = picks
    return top3


def grid_cells(model):
    # Phi-2 stops at k=8 like the paper; k=16 and 18 were dropped by the pre-declared
    # length rule (dense arm overflows at 16); no mid position at k=1
    ks = [1, 2, 4, 6, 8]
    if model == "llama2":
        ks += [10, 12, 14]
    cells = ["closed_book", "gold_only"]
    for arm in ("random", "bm25", "bge"):
        for k in ks:
            for pos in ("far", "mid", "near"):
                if k == 1 and pos == "mid":
                    continue
                cells.append(f"{arm}_k{k}_{pos}")
    cells.append("top3_bm25")
    cells.append("top3_bge")
    return cells


def cell_documents(cell, qid, draws, gold_pid, top3):
    # the ordered passage ids shown in one cell; the gold slot comes from the position
    # rule of the paper: far = first, mid = (n-1)//2, near = last of n = k+1 documents
    if cell == "closed_book":
        return []
    if cell == "gold_only":
        return [gold_pid[qid]]
    if cell.startswith("top3"):
        # the paper's code reverses the retrieved documents: best ranked LAST, by the claim
        return list(reversed(top3[qid][cell.split("_")[1]]))
    if cell.startswith("mixed"):
        # the paper's table 3 line: random documents BEFORE the reversed retrieved ones;
        # a random falling on a top3 abstract is replaced by the next spare of the list
        picked = top3[qid][cell.split("_")[1]]
        used = set()
        for pid in picked:
            used.add(split_id(pid)[0])
        rand = []
        for pid in draws[qid]["random"]:
            if split_id(pid)[0] in used:
                continue
            rand.append(pid)
            if len(rand) == 6:
                break
        return rand + list(reversed(picked))
    arm, k, pos = cell.split("_")
    k = int(k[1:])
    docs = draws[qid][arm][:k]
    if pos == "far":
        slot = 0
    elif pos == "mid":
        slot = k // 2                  # = (n-1)//2 with n = k+1 documents
    else:
        slot = k
    return docs[:slot] + [gold_pid[qid]] + docs[slot:]


def build_prompt(text, docs, titles, sentences):
    # docs = ordered passage ids; returns (prompt, mapping n -> passage id);
    # raw prompt, no chat template, like the paper
    if not docs:
        return CLOSED_BOOK + f"\nClaim: {text}\nAnswer:", {}
    lines = []
    mapping = {}
    for n, pid in enumerate(docs, 1):
        doc, a, b = split_id(pid)
        title = " ".join(titles[doc].split())
        body = " ".join(" ".join(sentences[doc][a:b]).split())
        lines.append(f"Document [{n}](Title: {title}) {body}")
        mapping[n] = pid
    prompt = INSTRUCTION + "\nDocuments:\n" + "\n".join(lines) + f"\nClaim: {text}\nAnswer:"
    return prompt, mapping

# ---- GENERATION

def load_tokenizers():
    # the public llama tokenizer copy has no config.json: snapshot_download hands back
    # its cache folder (offline too); model_max_length silences a wrong 2048 warning
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    folder = snapshot_download("hf-internal-testing/llama-tokenizer")
    return {"phi2": AutoTokenizer.from_pretrained("microsoft/phi-2"),
            "llama2": AutoTokenizer.from_pretrained(folder, model_max_length=4096)}


def dry_run(claims, draws, gold_pid, top3, titles, sentences):
    # builds every prompt of the grid and counts with the two real tokenizers;
    # one overflow = loud stop. Writes the tightest margin of each arm to dry_run.txt
    toks = load_tokenizers()
    # llama window hard-coded for now: the real model's config is asserted at generation
    limits = {"phi2": 2048, "llama2": 4096}

    # counter check against the review's frozen number: claim 636, 16 BM25
    # distractors, gold last -> 4059 tokens, start token included
    docs = draws["636"]["bm25"][:16] + [gold_pid["636"]]
    prompt, mapping = build_prompt(claims["636"]["text"], docs, titles, sentences)
    assert len(toks["llama2"](prompt)["input_ids"]) == 4059

    # the gold lands where the cell says: with k=4 it is document 1, 3 or 5 of 5
    for pos, slot in (("far", 1), ("mid", 3), ("near", 5)):
        docs = cell_documents(f"random_k4_{pos}", "1140", draws, gold_pid, top3)
        prompt, mapping = build_prompt(claims["1140"]["text"], docs, titles, sentences)
        assert mapping[slot] == gold_pid["1140"]

    lines = []
    for model in ("phi2", "llama2"):
        cells = grid_cells(model) + ["mixed_bm25", "mixed_bge"]
        tight = {}
        for cell in cells:
            arm = cell.split("_")[0]
            for qid in claims:
                docs = cell_documents(cell, qid, draws, gold_pid, top3)
                shown = []
                for pid in docs:
                    shown.append(split_id(pid)[0])
                assert len(set(shown)) == len(shown), f"{model} {cell} claim {qid}: same abstract twice"
                prompt, mapping = build_prompt(claims[qid]["text"], docs, titles, sentences)
                n = len(toks[model](prompt)["input_ids"])
                margin = limits[model] - GEN_TOKENS - n
                assert margin >= 0, f"{model} {cell} claim {qid}: {n} tokens over"
                if arm not in tight or margin < tight[arm][0]:
                    tight[arm] = (margin, cell, qid, n)
        lines.append(f"{model}  (limit {limits[model] - GEN_TOKENS} = {limits[model]} - {GEN_TOKENS} ; "
                     f"{len(cells)} cells x {len(claims)} claims, no overflow; tightest margin per arm)")
        for arm in tight:
            margin, cell, qid, n = tight[arm]
            lines.append(f"  {arm:7} margin {margin:5d}  {cell:15}  claim {qid:5}  ({n} tokens)")
    with open(DRY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def load_model(name):
    # fp16 on the gpu; the real window size is asserted here, not the tokenizer's
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = {"phi2": "microsoft/phi-2", "llama2": "meta-llama/Llama-2-7b-chat-hf"}[name]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                                 device_map="cuda")
    model.eval()
    window = {"phi2": 2048, "llama2": 4096}[name]
    assert model.config.max_position_embeddings == window
    return model, tok, window


def resume_keys(path):
    # a run killed mid-write can leave half a last line: drop it, keep the rest
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        lines = f.readlines()
    if lines:
        try:
            json.loads(lines[-1])
        except ValueError:
            lines = lines[:-1]
            with open(path, "w") as f:
                f.writelines(lines)
    done = set()
    for line in lines:
        done.add(json.loads(line)["claim_id"])
    return done


def generate_cell(model, tok, window, name, cell, claims, draws, gold_pid, top3,
                  titles, sentences):
    # bare greedy, 30 new tokens (D16); one JSONL line per claim, appended;
    # (path, claim_id) is the resume key, so a relaunch only fills the holes (D20)
    import torch
    path = os.path.join(GEN_DIR, name, INSTRUCTION_ID, cell + ".jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = resume_keys(path)
    start = time.time()
    new = 0
    with open(path, "a") as out:
        for qid in sorted(claims, key=int):
            if qid in done:
                continue
            docs = cell_documents(cell, qid, draws, gold_pid, top3)
            prompt, mapping = build_prompt(claims[qid]["text"], docs, titles, sentences)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            assert ids["input_ids"].shape[1] + GEN_TOKENS <= window
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            row = {"claim_id": qid, "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                   "output": text, "mapping": mapping}
            out.write(json.dumps(row) + "\n")
            out.flush()
            new += 1
    minutes = (time.time() - start) / 60
    if new:
        print(f"{name} {cell}: {new} generations, {new / minutes:.1f} per minute", flush=True)
    else:
        print(f"{name} {cell}: nothing new (resume)", flush=True)


def pilot():
    # D14: 50 train claims, closed_book / gold_only / random_k8_near on both models,
    # plus the long random_k14_near on llama; never touches the frozen test draws
    download_data()
    doc_ids, titles, sentences = load_corpus()
    claims = load_claims("train")
    eligible = []
    for qid in claims:
        if all(doc in sentences for doc in claims[qid]["evidence"]):
            eligible.append(qid)
    rng = random.Random(SEED)          # a fresh generator, not the one behind draws.json
    chosen = {}
    for qid in sorted(rng.sample(eligible, N_PILOT), key=int):
        chosen[qid] = claims[qid]
    sisters = sister_evidence(chosen)
    passages, windows = build_passages(doc_ids, sentences)
    cum = block_scale(doc_ids, sentences)
    draws = {}
    for qid in chosen:
        banned = chosen[qid]["cited"] | sisters[qid]
        draws[qid] = {"random": draw_randoms(cum, windows, doc_ids, banned, rng)}
    write_once(PILOT_FILE, draws)
    gold_pid, widened = gold_pids(chosen, sentences)
    print(len(chosen), "pilot claims,", len(widened), "widened gold windows", flush=True)
    for name in ("phi2", "llama2"):
        cells = ["closed_book", "gold_only", "random_k8_near"]
        if name == "llama2":
            cells.append("random_k14_near")
        try:
            model, tok, window = load_model(name)
        except OSError as e:           # gated or absent repo: say it, go on (asserts still stop)
            print(name, "not available:", e, flush=True)
            continue
        for cell in cells:
            generate_cell(model, tok, window, name, cell, chosen, draws, gold_pid, {},
                          titles, sentences)
        del model

# ---- PARSE

def parse_output(text):
    # first label pattern wins ("not supported" matches too: counted by the report,
    # the pilot decides the final patterns); citations = every [n] or "document n"
    label = None
    m = re.search(r"support|contradict|refute", text, re.I)
    if m:
        if m.group(0).lower() == "support":
            label = "SUPPORT"
        else:
            label = "CONTRADICT"
    cites = []
    for c in re.finditer(r"\[(\d+)\]|document\s+(\d+)", text, re.I):
        cites.append(int(c.group(1) or c.group(2)))
    return label, cites


def pilot_report():
    # D14 thresholds: labels >= 90 %, usable citation >= 70 %. Two label readings:
    # raw = first pattern anywhere; strict = drops "SUPPORT | CONTRADICT" echoes that
    # 30 tokens cannot resolve, and labels not on the first line (hallucinated
    # document headers). The gold sits LAST in these cells: slot = len(mapping)
    lines = []
    for name in ("phi2", "llama2"):
        for cell in ("closed_book", "gold_only", "random_k8_near", "random_k14_near"):
            path = os.path.join(GEN_DIR, name, INSTRUCTION_ID, cell + ".jsonl")
            if not os.path.exists(path):
                continue
            rows = []
            with open(path) as f:
                for line in f:
                    rows.append(json.loads(line))
            raw = 0
            strict = 0
            echo = 0
            unresolved = 0
            first_gold = 0
            any_gold = 0
            negated = 0
            bad = []
            for row in rows:
                text = row["output"]
                label, cites = parse_output(text)
                raw += label is not None
                negated += bool(re.search(r"not?\s+support", text, re.I))
                gold_slot = len(row["mapping"])
                if gold_slot:
                    first_gold += bool(cites) and cites[0] == gold_slot
                    any_gold += gold_slot in cites
                m = re.match(r"\s*SUPPORT\s*\|\s*CONTRADICT", text)
                if m:
                    echo += 1
                    after, more = parse_output(text[m.end():])
                    if after is None:
                        unresolved += 1
                    else:
                        strict += 1
                else:
                    head, more = parse_output(text.split("\n")[0])
                    strict += head is not None
                if label is None and len(bad) < 3:
                    bad.append(text.replace("\n", " ")[:120])
            n = len(rows)
            lines.append(f"{name} {cell}: {n} outputs; label raw {100 * raw / n:.0f} % / "
                         f"strict {100 * strict / n:.0f} % (echoes {echo}, unresolved {unresolved}); "
                         f"gold cited first {first_gold}/{n}, anywhere {any_gold}/{n}; negated {negated}")
            for text in bad:
                lines.append(f"  unparsed: {text}")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "pilot_report.txt"), "w") as f:
        f.write(report)
    print(report, flush=True)


def main():
    download_data()
    doc_ids, titles, sentences = load_corpus()
    queries, qrels = load_queries()
    claims = load_claims("dev")        # the BEIR test split is the original dev
    support = 0
    multi = 0
    for c in claims.values():
        support += c["label"] == "SUPPORT"
        multi += len(c["golds"]) > 1
    assert len(claims) == 188
    assert support == 124
    assert len(claims) - support == 64
    assert multi == 10
    for qid in claims:
        assert claims[qid]["cited"] == set(qrels[qid])      # AI2 citations match the BEIR qrels
    sisters = sister_evidence(claims)

    passages, windows = build_passages(doc_ids, sentences)
    assert len(passages) == 24917
    print(len(passages), "passages")

    bm25 = build_index(passages, doc_ids, titles, sentences)
    run = {}
    best_pass = {}
    for qid in queries:
        scores = bm25.get_scores(preprocess(queries[qid]))
        order, best_score, best_window = rank_abstracts(scores, passages)
        run[qid], best_pass[qid] = run_entry(order, best_score, best_window, doc_ids)
    write_once(RUN_FILE, run)
    write_once(BEST_FILE, best_pass)
    with open(RUN_FILE) as f:
        ndcg, recall = evaluate(json.load(f))
    print(f"bm25 on passages: nDCG@10 {ndcg:.3f}  Recall@100 {recall:.3f}")
    assert round(ndcg, 3) == 0.691
    assert round(recall, 3) == 0.921

    # dense scores are not bit-reproducible across machines: computed once, then read
    if os.path.exists(BGE_RUN_FILE):
        with open(BGE_RUN_FILE) as f:
            bge_run = json.load(f)
        with open(BGE_BEST_FILE) as f:
            bge_best = json.load(f)
        print("bge run read from file")
    else:
        bge_run, bge_best = dense_run(passages, doc_ids, titles, sentences, queries)
        write_once(BGE_RUN_FILE, bge_run)
        write_once(BGE_BEST_FILE, bge_best)
    ndcg, recall = evaluate(bge_run)
    print(f"bge on passages:  nDCG@10 {ndcg:.3f}  Recall@100 {recall:.3f}")
    assert round(ndcg, 2) == 0.72
    assert round(recall, 2) == 0.96

    cum = block_scale(doc_ids, sentences)
    rng = random.Random(SEED)          # one shared generator, claims in a fixed order
    draws = {}
    for qid in sorted(claims, key=int):
        banned = claims[qid]["cited"] | sisters[qid]
        draws[qid] = {"random": draw_randoms(cum, windows, doc_ids, banned, rng),
                      "bm25": pick_distractors(run[qid], best_pass[qid],
                                               set(claims[qid]["golds"]), sisters[qid]),
                      "bge": pick_distractors(bge_run[qid], bge_best[qid],
                                              set(claims[qid]["golds"]), sisters[qid])}
    for qid in claims:
        for key in ("random", "bm25", "bge"):
            docs = []
            for pid in draws[qid][key]:
                docs.append(split_id(pid)[0])
            assert len(set(docs)) == N_DOCS
            assert not set(docs) & (set(claims[qid]["golds"]) | sisters[qid])
        rand_docs = [split_id(pid)[0] for pid in draws[qid]["random"]]
        assert not set(rand_docs) & claims[qid]["cited"]
    write_draws(DRAWS_FILE, draws)

    # frozen control: distractor n°1 of claim 1140 shows sentences 2 to 5 of abstract 41710132
    assert draws["1140"]["bm25"][0] == "41710132:2-6"

    gold_pid, widened = gold_pids(claims, sentences)
    assert widened == ["1110", "1241"]
    assert gold_pid["1110"].split(":")[1] == "13-14"
    assert gold_pid["1241"].split(":")[1] == "0-5"
    assert gold_pid["261"].split(":")[0] == "10697096"      # multi-gold: its other gold wins
    inside = 0
    for qid in claims:
        doc, a, b = split_id(gold_pid[qid])
        words = [len(s.split()) for s in sentences[doc]]
        wins = []
        for i, end, total in spans(words):
            wins.append((i, end))
        inside += (a, b) in wins
    assert inside == 187               # all but the widened window of claim 1241
    print("gold windows: 187/188 are maximal windows (1241 widened to its set);")
    print("1110 and 1241 exceed 100 words, documented exceptions")

    top3 = top3_lists(claims, run, best_pass, bge_run, bge_best)
    assert len(grid_cells("phi2")) == 46
    assert len(grid_cells("llama2")) == 73
    for line in dry_run(claims, draws, gold_pid, top3, titles, sentences):
        print(line)
    print("wrote", DRY_FILE)


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        pilot()
    elif "--pilot-report" in sys.argv:
        pilot_report()
    else:
        main()
