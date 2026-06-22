"""
LSH on MinHash signatures.

we split the signature into `bands` bands, each of `rows_per_band` rows.
hash each band to a bucket. two docs that share at least one bucket
become a candidate pair.

approx prob a pair becomes candidate as function of jaccard s:
    P(candidate) = 1 - (1 - s^r)^b
the (b, r) choice tunes the s-threshold curve.

we DO NOT use scikit/datasketch helpers. plain dict-based buckets.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def _band_hash(band_vals: np.ndarray) -> str:
    """hash a band tuple to a stable string key."""
    # use md5 of bytes for stability across runs
    return hashlib.md5(band_vals.tobytes()).hexdigest()


class LSHIndex:
    """
    Locality-sensitive hashing index for MinHash signatures.

    parameters
    ----------
    num_perm : int
        length of the signature.
    bands : int
        number of bands. must divide num_perm exactly.
    """

    def __init__(self, num_perm: int = 128, bands: int = 32):
        if num_perm % bands != 0:
            raise ValueError(
                f"num_perm ({num_perm}) must be divisible by bands ({bands})"
            )
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        # one bucket dict per band
        self.buckets: List[Dict[str, List[int]]] = [defaultdict(list) for _ in range(bands)]
        self._signatures: Dict[int, np.ndarray] = {}
        self._doc_ids: List[int] = []

    # ----- build -----
    def insert(self, doc_id: int, signature: np.ndarray) -> None:
        if signature.shape[0] != self.num_perm:
            raise ValueError("signature length mismatch")
        self._signatures[doc_id] = signature
        self._doc_ids.append(doc_id)
        for bi in range(self.bands):
            start = bi * self.rows_per_band
            end = start + self.rows_per_band
            key = _band_hash(signature[start:end])
            self.buckets[bi][key].append(doc_id)

    def build(self, signatures: np.ndarray, doc_ids: Sequence[int] | None = None) -> None:
        """bulk-insert. signatures: (N, num_perm)."""
        n = signatures.shape[0]
        if doc_ids is None:
            doc_ids = list(range(n))
        if len(doc_ids) != n:
            raise ValueError("doc_ids length mismatch")
        for i in range(n):
            self.insert(int(doc_ids[i]), signatures[i])

    # ----- query -----
    def query(self, signature: np.ndarray) -> Set[int]:
        """return doc ids that share at least one band-bucket with the query."""
        cand: Set[int] = set()
        for bi in range(self.bands):
            start = bi * self.rows_per_band
            end = start + self.rows_per_band
            key = _band_hash(signature[start:end])
            if key in self.buckets[bi]:
                cand.update(self.buckets[bi][key])
        return cand

    def candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        enumerate all unique candidate pairs across all buckets.
        used for batch / corpus mode.
        """
        pairs: Set[Tuple[int, int]] = set()
        for band_bkt in self.buckets:
            for ids in band_bkt.values():
                if len(ids) < 2:
                    continue
                # all C(k,2) inside the bucket
                ids_sorted = sorted(ids)
                for i in range(len(ids_sorted)):
                    for j in range(i + 1, len(ids_sorted)):
                        pairs.add((ids_sorted[i], ids_sorted[j]))
        return pairs

    # ----- info -----
    def stats(self) -> dict:
        bucket_sizes = []
        for band_bkt in self.buckets:
            bucket_sizes.extend(len(v) for v in band_bkt.values())
        return {
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows_per_band": self.rows_per_band,
            "num_docs": len(self._doc_ids),
            "total_buckets": sum(len(b) for b in self.buckets),
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "avg_bucket_size": (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0,
        }


def threshold_from_bands(bands: int, rows_per_band: int) -> float:
    """approx jaccard threshold s where the S-curve crosses 0.5: s ~ (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows_per_band)
