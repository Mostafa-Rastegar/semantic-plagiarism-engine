"""
SimHash with TF-IDF weights.

algorithm:
    1. tokenize -> n-gram tokens
    2. compute tf-idf weight per token over the corpus
    3. for each doc:
        v = zeros(64)
        for token w:
            h = hash64(w)            # 64-bit
            for each bit position i:
                if bit i of h == 1:  v[i] += weight(w)
                else:                v[i] -= weight(w)
        final_bit_i = 1 if v[i] > 0 else 0
    4. sim(d1, d2) = 1 - hamming(d1, d2) / 64

we do tf-idf manually (math.log on counts) so the only sklearn dep is optional.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


_BITS = 64


def hash64(token: str) -> int:
    """64-bit hash of a token (md5 truncated)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def ngrams(tokens: Sequence[str], n: int = 1) -> List[str]:
    """build n-grams. n=1 -> unigram tokens (fastest)."""
    if n <= 1:
        return list(tokens)
    out = []
    for i in range(len(tokens) - n + 1):
        out.append(" ".join(tokens[i : i + n]))
    return out


def compute_idf(corpus_tokens: Iterable[Sequence[str]]) -> Dict[str, float]:
    """
    classic IDF with smoothing: idf(t) = log((1+N) / (1+df(t))) + 1
    """
    df: Dict[str, int] = defaultdict(int)
    n_docs = 0
    for toks in corpus_tokens:
        n_docs += 1
        for t in set(toks):
            df[t] += 1
    idf = {}
    for t, c in df.items():
        idf[t] = math.log((1.0 + n_docs) / (1.0 + c)) + 1.0
    return idf


def simhash_signature(tokens: Sequence[str], idf: Dict[str, float] | None = None) -> int:
    """
    compute the 64-bit simhash fingerprint as a python int.
    if idf is None, weight=1 (plain simhash).
    """
    if not tokens:
        return 0
    tf = Counter(tokens)
    v = np.zeros(_BITS, dtype=np.float64)
    for tok, count in tf.items():
        w = count * (idf.get(tok, 1.0) if idf else 1.0)
        h = hash64(tok)
        # walk bits low->high
        for i in range(_BITS):
            if (h >> i) & 1:
                v[i] += w
            else:
                v[i] -= w
    # signs -> bits
    out = 0
    for i in range(_BITS):
        if v[i] > 0:
            out |= (1 << i)
    return out


def simhash_signature_fast(tokens: Sequence[str], idf: Dict[str, float] | None = None) -> int:
    """
    vectorised version of simhash_signature. faster for large vocabularies.
    keeps the same semantics: bit-position i over weighted sum of sign(bit_i(hash(tok))).
    """
    if not tokens:
        return 0
    tf = Counter(tokens)
    toks = list(tf.keys())
    counts = np.array([tf[t] for t in toks], dtype=np.float64)
    if idf:
        weights = counts * np.array([idf.get(t, 1.0) for t in toks], dtype=np.float64)
    else:
        weights = counts
    hashes = np.array([hash64(t) for t in toks], dtype=np.uint64)

    # build bit matrix (n_tokens, 64): bit i is (hash >> i) & 1
    # broadcast: shift each hash by 0..63
    bit_positions = np.arange(_BITS, dtype=np.uint64)
    # shape (n_tokens, 64): use Python loop only on bits (small fixed 64)
    bits = ((hashes[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.int8)
    # signs: 1 -> +w, 0 -> -w
    signed = (2 * bits - 1).astype(np.float64) * weights[:, None]
    v = signed.sum(axis=0)
    out = 0
    for i in range(_BITS):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming_distance(a: int, b: int) -> int:
    """popcount of XOR (Brian Kernighan / int.bit_count)."""
    x = a ^ b
    # python 3.10+ has int.bit_count
    if hasattr(x, "bit_count"):
        return x.bit_count()
    # fallback
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def simhash_similarity(a: int, b: int) -> float:
    """1 - hamming/64. in [0, 1]."""
    return 1.0 - (hamming_distance(a, b) / _BITS)


class SimHasher:
    """wrapper that fits idf on a corpus and then signs docs."""

    def __init__(self, ngram: int = 1, use_idf: bool = True):
        self.ngram = ngram
        self.use_idf = use_idf
        self.idf: Dict[str, float] | None = None

    def fit(self, corpus_tokens: Sequence[Sequence[str]]) -> "SimHasher":
        # build ngrams per doc and idf
        all_ng = [ngrams(t, self.ngram) for t in corpus_tokens]
        if self.use_idf:
            self.idf = compute_idf(all_ng)
        else:
            self.idf = None
        # cache for reuse if user calls transform with same list
        self._cached = all_ng
        return self

    def sign(self, tokens: Sequence[str]) -> int:
        ng = ngrams(tokens, self.ngram)
        return simhash_signature_fast(ng, self.idf)

    def sign_corpus(self, corpus_tokens: Sequence[Sequence[str]]) -> List[int]:
        # use cached ngrams when possible
        if getattr(self, "_cached", None) and len(self._cached) == len(corpus_tokens):
            return [simhash_signature_fast(ng, self.idf) for ng in self._cached]
        return [self.sign(t) for t in corpus_tokens]
