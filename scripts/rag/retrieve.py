"""Retrievers (task 3.3): a lexical floor that runs anywhere, and an embedding retriever for VM03.

The floor is not a stub. Table retrieval is largely keyword matching (the question says EMEA and Q3, so does the
table), so an embedding model that does not beat it has not earned its latency, and a recall quoted without it
says nothing about whether the embeddings did anything. It is pure Python: no network, no compiled extensions,
which is what makes tasks 3.1-3.4 testable on the authoring laptop.

  tfidf   cosine over log-scaled TF-IDF
  bm25    Okapi BM25 (k1 1.5, b 0.75)
  embed:<name>               sentence-transformers cosine, one of EMBED_MODELS (task 3.10)
  bm25>rerank:<name>         BM25 candidates reordered by a cross-encoder, one of RERANKERS (task 3.11)

The model-backed retrievers need sentence-transformers (~/venv-embed on VM03) and are not imported unless asked
for. Each records its model id, the snapshot revision it loaded and how many inputs it truncated.

Chunks are ranked, then collapsed to their parent table (best chunk score) BEFORE k is applied; the other order
makes k mean different things for whole-table and row-group chunking.
"""
import math
import re
from collections import Counter
from pathlib import Path

LEXICAL = ("tfidf", "bm25")

# Each embedding model with the query/passage convention its model card documents. MiniLM is a symmetric
# sentence-similarity model; the other three are retrieval-trained (asymmetric: short query, long passage).
EMBED_MODELS = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2", "query": "", "passage": ""},
    "bge-small": {"id": "BAAI/bge-small-en-v1.5",
                  "query": "Represent this sentence for searching relevant passages: ", "passage": ""},
    "e5-small": {"id": "intfloat/e5-small-v2", "query": "query: ", "passage": "passage: "},
    "bge-base": {"id": "BAAI/bge-base-en-v1.5",
                 "query": "Represent this sentence for searching relevant passages: ", "passage": ""},
}
RERANKERS = {
    "ms-marco-minilm": {"id": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    "bge-reranker-base": {"id": "BAAI/bge-reranker-base"},
}
# A cross-encoder can only reorder the candidates it is handed. chart_metadata already has its gold in BM25's top
# 5 for ~97% of rows; question_only for ~34%, so it gets a deeper pool, or a first-stage miss reads as a reranker
# failure. BM25's recall at this depth is reported beside every reranker row as its ceiling.
RERANK_DEPTH = {"chart_metadata": 10, "question_only": 50}
RERANK_MAX_TOKENS = 512
RETRIEVERS = LEXICAL + tuple(f"embed:{n}" for n in EMBED_MODELS) + tuple(f"bm25>rerank:{n}" for n in RERANKERS)
STOP = set("the a an of in on for to and or is was were what which how many much by from at as with be are this "
           "that between than did does do its it there their per value values".split())


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


class Lexical:
    def __init__(self, chunks, kind="bm25", k1=1.5, b=0.75):
        self.chunks, self.kind, self.k1, self.b = chunks, kind, k1, b
        self.docs = [Counter(tokens(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()} if kind == "bm25" \
            else {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        if kind == "tfidf":
            self.norms = [math.sqrt(sum(((1 + math.log(c)) * self.idf[t]) ** 2 for t, c in d.items())) or 1.0
                          for d in self.docs]
        self.index = {}
        for i, d in enumerate(self.docs):
            for t in d:
                self.index.setdefault(t, []).append(i)

    def scores(self, query):
        q = Counter(tokens(query))
        s = Counter()
        if self.kind == "bm25":
            for t in q:
                for i in self.index.get(t, ()):
                    f = self.docs[i][t]
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            return s
        qw = {t: (1 + math.log(c)) * self.idf.get(t, 0) for t, c in q.items()}
        qn = math.sqrt(sum(v * v for v in qw.values())) or 1.0
        for t, w in qw.items():
            for i in self.index.get(t, ()):
                s[i] += w * (1 + math.log(self.docs[i][t])) * self.idf[t]
        return Counter({i: v / (qn * self.norms[i]) for i, v in s.items()})


def snapshot(model_id, revision=None):
    """The revision (commit sha) of the cached snapshot a model was loaded from; None if it cannot be resolved."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, "config.json", revision=revision)
        parts = Path(str(path)).parts
        return parts[parts.index("snapshots") + 1]
    except Exception:  # noqa: BLE001 - provenance is best-effort; a missing sha is recorded as None, not guessed
        return None


class Embedding:
    def __init__(self, chunks, name="minilm", revision=None):
        from sentence_transformers import SentenceTransformer
        spec = EMBED_MODELS[name]
        # revision pins a snapshot (task 3.17 reloads the exact ones 3.10 measured); None = whatever is cached
        model = SentenceTransformer(spec["id"], revision=revision) if revision else SentenceTransformer(spec["id"])
        self.chunks, self.spec, self.model = chunks, spec, model
        texts = [spec["passage"] + c["text"] for c in chunks]
        lengths = [len(x) for x in self.model.tokenizer(texts)["input_ids"]]
        self.info = {"id": spec["id"], "revision": snapshot(spec["id"], revision), "query_prefix": spec["query"],
                     "passage_prefix": spec["passage"], "max_seq_length": self.model.max_seq_length,
                     "passages_truncated": sum(n > self.model.max_seq_length for n in lengths),
                     "passages": len(texts)}
        self.vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def scores(self, query):
        q = self.model.encode([self.spec["query"] + query], normalize_embeddings=True)[0]
        return Counter({i: float(v) for i, v in enumerate(self.vecs @ q)})


class Reranked:
    """BM25 over whole tables, then the top `depth` tables reordered by a cross-encoder; the rest keep BM25 order."""

    def __init__(self, chunks, name, depth):
        from sentence_transformers import CrossEncoder
        if len({c["chart_id"] for c in chunks}) != len(chunks):
            raise ValueError("the reranker reads whole tables: build it over whole_table chunks")
        spec = RERANKERS[name]
        self.chunks, self.depth = chunks, depth
        self.first = Lexical(chunks, "bm25")
        self.text = {c["chart_id"]: c["text"] for c in chunks}
        self.model = CrossEncoder(spec["id"], max_length=RERANK_MAX_TOKENS)
        self.info = {"id": spec["id"], "revision": snapshot(spec["id"]), "first_stage": "bm25", "depth": depth,
                     "max_length": RERANK_MAX_TOKENS, "pairs": 0, "pairs_truncated": 0}

    def rank(self, query):
        ranked = rank_tables(self.first, query)
        head = ranked[:self.depth]
        if not head:
            return ranked
        pairs = [(query, self.text[c]) for c in head]
        enc = self.model.tokenizer([q for q, _ in pairs], [t for _, t in pairs])["input_ids"]
        self.info["pairs"] += len(pairs)
        self.info["pairs_truncated"] += sum(len(x) > RERANK_MAX_TOKENS for x in enc)
        s = [float(x) for x in self.model.predict(pairs, batch_size=32, show_progress_bar=False)]
        order = sorted(range(len(head)), key=lambda i: (-s[i], i))   # ties keep BM25 order
        return [head[i] for i in order] + ranked[self.depth:]


def make(chunks, kind, plan=None):
    if kind in LEXICAL:
        return Lexical(chunks, kind)
    if kind.startswith("embed:") and kind[6:] in EMBED_MODELS:
        return Embedding(chunks, kind[6:])
    if kind.startswith("bm25>rerank:") and kind[12:] in RERANKERS:
        if plan not in RERANK_DEPTH:
            raise ValueError(f"a reranker needs the query plan to set its depth: one of {tuple(RERANK_DEPTH)}")
        return Reranked(chunks, kind[12:], RERANK_DEPTH[plan])
    raise ValueError(f"retriever must be one of {RETRIEVERS}")


def is_reranker(kind):
    return kind.startswith("bm25>rerank:")


def rank_tables(retriever, query, k=None):
    """Chart ids ranked by their best chunk's score (collapse first, then cut at k). Ties broken by chart id."""
    if hasattr(retriever, "rank"):
        ranked = retriever.rank(query)
        return ranked[:k] if k else ranked
    best = {}
    for i, v in retriever.scores(query).items():
        cid = retriever.chunks[i]["chart_id"]
        if v > best.get(cid, float("-inf")):
            best[cid] = v
    ranked = sorted(best, key=lambda c: (-best[c], c))
    return ranked[:k] if k else ranked
