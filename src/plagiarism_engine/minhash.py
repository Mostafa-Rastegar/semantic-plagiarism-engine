"""
MinHash from scratch.

idea:
  use family of hash functions of form  h_i(x) = (a_i * x + b_i) mod p
  for each shingle hash it with all i, and keep the min per i.
  resulting vector of mins is the signature.

we DO NOT use datasketch / scikit MinHash. just numpy + python hash.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, List, Sequence, Set

import numpy as np


# big prime > 2^32, used as modulus for the universal-hash family
_LARGE_PRIME = (1 << 61) - 1  # Mersenne prime, plenty big

_MAX_HASH = (1 << 32) - 1


def _stable_hash(s: str) -> int:
    """
    stable 32-bit hash for a shingle string. python's built-in hash()
    is randomized per process which would break reproducibility, so we
    use md5 truncated.
    """
    h = hashlib.md5(s.encode("utf-8")).digest()
    # take first 4 bytes -> 32-bit unsigned int
    return int.from_bytes(h[:4], "big")


class MinHasher:
    """
    builds num_perm-length signatures.

    parameters
    ----------
    num_perm : int
        number of hash permutations (signature length). 128 or 256 typical.
    seed : int
        RNG seed used to draw the (a, b) coefficients. fixed for reproducibility.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed
        rng = random.Random(seed)
        # draw a, b coefficients
        self.a = np.array(
            [rng.randint(1, _LARGE_PRIME - 1) for _ in range(num_perm)],
            dtype=np.int64,
        )
        self.b = np.array(
            [rng.randint(0, _LARGE_PRIME - 1) for _ in range(num_perm)],
            dtype=np.int64,
        )

    # ----- core -----
    def signature(self, shingle_set: Iterable[str]) -> np.ndarray:
        """compute the MinHash signature for a set of shingles."""
        sig = np.full(self.num_perm, _MAX_HASH, dtype=np.int64)
        # if empty, just return all-max (sentinel)
        items = list(shingle_set)
        if not items:
            return sig

        # hash each shingle once to int
        x = np.array([_stable_hash(s) for s in items], dtype=np.int64)  # shape (n,)

        # apply the family of hashes -> matrix (num_perm, n)
        # broadcasted: (num_perm, 1) * (1, n) + (num_perm, 1)
        hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % _LARGE_PRIME
        # take per-row min (axis=1)
        sig = hashed.min(axis=1)
        return sig

    def signatures(self, list_of_shingle_sets: Sequence[Set[str]]) -> np.ndarray:
        """compute signatures for many docs. returns matrix (N, num_perm)."""
        n = len(list_of_shingle_sets)
        out = np.full((n, self.num_perm), _MAX_HASH, dtype=np.int64)
        for i, s in enumerate(list_of_shingle_sets):
            out[i] = self.signature(s)
        return out


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """fraction of equal entries -> Jaccard estimate."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have same length")
    if sig_a.size == 0:
        return 0.0
    return float(np.mean(sig_a == sig_b))
