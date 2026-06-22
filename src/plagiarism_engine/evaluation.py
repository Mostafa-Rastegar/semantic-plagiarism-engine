"""
evaluation utilities: precision/recall/F1 over labeled pairs, plus runtime measurement
for the two pipelines (MinHash+LSH and TF-IDF SimHash).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .lsh import LSHIndex
from .minhash import MinHasher, estimated_jaccard
from .preprocessing import preprocess_document, tokenize
from .simhash import SimHasher, hamming_distance, simhash_similarity


@dataclass
class EvalResult:
    method: str
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    runtime_sec: float

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "runtime_sec": round(self.runtime_sec, 3),
        }


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_minhash_pairs(
    df: pd.DataFrame,
    threshold: float = 0.5,
    num_perm: int = 128,
    shingle_size: int = 3,
    seed: int = 42,
) -> EvalResult:
    """
    score each row's (text_a, text_b) with MinHash-estimated Jaccard,
    predict positive if >= threshold, and compute P/R/F1 against label.
    """
    mh = MinHasher(num_perm=num_perm, seed=seed)
    start = time.time()
    tp = fp = fn = tn = 0
    for _, row in df.iterrows():
        a = preprocess_document(row["text_a"], shingle_size=shingle_size)
        b = preprocess_document(row["text_b"], shingle_size=shingle_size)
        sa = mh.signature(a)
        sb = mh.signature(b)
        sim = estimated_jaccard(sa, sb)
        pred = 1 if sim >= threshold else 0
        y = int(row["label"])
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    rt = time.time() - start
    p, r, f = _prf(tp, fp, fn)
    return EvalResult("minhash", threshold, p, r, f, tp, fp, fn, tn, rt)


def evaluate_simhash_pairs(
    df: pd.DataFrame,
    threshold: float = 0.75,
    ngram: int = 1,
    use_idf: bool = True,
) -> EvalResult:
    """
    fit idf on the union of all texts then evaluate pair similarity by
    simhash hamming.  threshold is on similarity in [0,1].
    """
    # collect tokens for idf
    a_toks = [tokenize(t) for t in df["text_a"].tolist()]
    b_toks = [tokenize(t) for t in df["text_b"].tolist()]
    corpus = a_toks + b_toks
    sh = SimHasher(ngram=ngram, use_idf=use_idf).fit(corpus)

    start = time.time()
    tp = fp = fn = tn = 0
    n = len(df)
    sigs_a = [sh.sign(t) for t in a_toks]
    sigs_b = [sh.sign(t) for t in b_toks]
    for i in range(n):
        sim = simhash_similarity(sigs_a[i], sigs_b[i])
        pred = 1 if sim >= threshold else 0
        y = int(df.iloc[i]["label"])
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    rt = time.time() - start
    p, r, f = _prf(tp, fp, fn)
    return EvalResult("simhash", threshold, p, r, f, tp, fp, fn, tn, rt)


def sweep_thresholds(
    df: pd.DataFrame,
    method: str = "minhash",
    grid: Sequence[float] | None = None,
    **kwargs,
) -> List[dict]:
    """small helper: run several thresholds, return list of rows."""
    if grid is None:
        grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    for t in grid:
        if method == "minhash":
            res = evaluate_minhash_pairs(df, threshold=t, **kwargs)
        elif method == "simhash":
            res = evaluate_simhash_pairs(df, threshold=t, **kwargs)
        else:
            raise ValueError(method)
        rows.append(res.as_row())
    return rows


def lsh_candidates_for_corpus(
    texts: Sequence[str],
    num_perm: int = 128,
    bands: int = 32,
    shingle_size: int = 3,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int, float]], dict]:
    """
    run MinHash+LSH over a corpus and return candidate pairs with
    estimated jaccard >= 0 (filter on caller side).
    """
    mh = MinHasher(num_perm=num_perm, seed=seed)
    shingle_sets = [preprocess_document(t, shingle_size=shingle_size) for t in texts]
    sigs = mh.signatures(shingle_sets)
    index = LSHIndex(num_perm=num_perm, bands=bands)
    index.build(sigs)
    pairs = index.candidate_pairs()
    results = []
    for i, j in pairs:
        sim = estimated_jaccard(sigs[i], sigs[j])
        results.append((i, j, sim))
    # sort descending by similarity
    results.sort(key=lambda x: -x[2])
    return results, index.stats()
