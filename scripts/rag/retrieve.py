"""Retrievers (task 3.3): a lexical floor that runs anywhere, and an embedding retriever for VM03.

The floor is not a stub. Table retrieval is largely keyword matching (the question says EMEA and Q3, so does the
table), so an embedding model that does not beat it has not earned its latency, and a recall quoted without it
says nothing about whether the embeddings did anything. It is pure Python: no network, no compiled extensions,
which is what makes tasks 3.1-3.4 testable on the authoring laptop.

  tfidf   cosine over log-scaled TF-IDF
  bm25    Okapi BM25 (k1 1.5, b 0.75)
  embed   sentence-transformers cosine (VM03 only; any model id; not imported unless asked for)

Chunks are ranked, then collapsed to their parent table (best chunk score) BEFORE k is applied; the other order
makes k mean different things for whole-table and row-group chunking.
"""
import math
import re
from collections import Counter

RETRIEVERS = ("tfidf", "bm25", "embed")
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


class Embedding:
    def __init__(self, chunks, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.chunks, self.model = chunks, SentenceTransformer(model)
        self.vecs = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return Counter({i: float(v) for i, v in enumerate(self.vecs @ q)})


def make(chunks, kind, **kw):
    if kind in ("tfidf", "bm25"):
        return Lexical(chunks, kind)
    if kind == "embed":
        return Embedding(chunks, **kw)
    raise ValueError(f"retriever must be one of {RETRIEVERS}")


def rank_tables(retriever, query, k=None):
    """Chart ids ranked by their best chunk's score (collapse first, then cut at k). Ties broken by chart id."""
    best = {}
    for i, v in retriever.scores(query).items():
        cid = retriever.chunks[i]["chart_id"]
        if v > best.get(cid, float("-inf")):
            best[cid] = v
    ranked = sorted(best, key=lambda c: (-best[c], c))
    return ranked[:k] if k else ranked
