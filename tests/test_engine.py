"""basic unit tests."""

import numpy as np
import pytest

from plagiarism_engine.preprocessing import (
    jaccard,
    normalize,
    preprocess_document,
    shingles,
    tokenize,
)
from plagiarism_engine.minhash import MinHasher, estimated_jaccard
from plagiarism_engine.lsh import LSHIndex, threshold_from_bands
from plagiarism_engine.simhash import (
    SimHasher,
    hamming_distance,
    simhash_similarity,
    simhash_signature,
    simhash_signature_fast,
    ngrams,
)


def test_normalize_basic():
    assert normalize("Hello, WORLD!") == "hello world"
    assert normalize("") == ""
    assert normalize(None) == ""


def test_tokenize_removes_stopwords():
    toks = tokenize("the cat sat on the mat", remove_stop=True)
    assert "the" not in toks
    assert "cat" in toks


def test_shingles_length():
    toks = ["a", "b", "c", "d"]
    s = shingles(toks, k=2)
    assert s == {"a b", "b c", "c d"}


def test_shingles_short_doc():
    s = shingles(["only", "two"], k=5)
    assert len(s) >= 1


def test_jaccard_extremes():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9


def test_minhash_identical_docs():
    mh = MinHasher(num_perm=128, seed=0)
    sh = preprocess_document("the quick brown fox jumps over the lazy dog", shingle_size=2)
    s = mh.signature(sh)
    assert estimated_jaccard(s, s) == 1.0


def test_minhash_estimate_close_to_true_jaccard():
    mh = MinHasher(num_perm=256, seed=1)
    a = preprocess_document(
        "the quick brown fox jumps over the lazy dog and runs away", shingle_size=2
    )
    b = preprocess_document(
        "the quick brown fox leaps over the lazy dog and runs away", shingle_size=2
    )
    true_j = jaccard(a, b)
    est = estimated_jaccard(mh.signature(a), mh.signature(b))
    # within 0.15 with 256 perms
    assert abs(true_j - est) < 0.2


def test_lsh_finds_duplicate():
    mh = MinHasher(num_perm=128, seed=0)
    a = preprocess_document("alpha beta gamma delta epsilon zeta eta theta", shingle_size=2)
    b = preprocess_document("alpha beta gamma delta epsilon zeta eta theta", shingle_size=2)
    c = preprocess_document("totally different content nothing in common here", shingle_size=2)
    sigs = mh.signatures([a, b, c])
    idx = LSHIndex(num_perm=128, bands=32)
    idx.build(sigs)
    pairs = idx.candidate_pairs()
    assert (0, 1) in pairs
    assert (0, 2) not in pairs


def test_lsh_threshold_formula():
    t = threshold_from_bands(32, 4)
    assert 0.0 < t < 1.0


def test_simhash_identical():
    toks = ["one", "two", "three", "four"]
    s = simhash_signature(toks)
    s2 = simhash_signature(toks)
    assert s == s2
    assert simhash_similarity(s, s2) == 1.0


def test_simhash_fast_matches_slow():
    toks = ["the", "quick", "brown", "fox", "jumps", "again", "fox"]
    s_slow = simhash_signature(toks)
    s_fast = simhash_signature_fast(toks)
    assert s_slow == s_fast


def test_simhash_different_docs_have_lower_sim():
    sh = SimHasher(ngram=1, use_idf=True)
    a = ["machine", "learning", "is", "fun"]
    b = ["machine", "learning", "is", "fun"]
    c = ["totally", "unrelated", "stuff"]
    sh.fit([a, b, c])
    sa, sb, sc = sh.sign(a), sh.sign(b), sh.sign(c)
    assert simhash_similarity(sa, sb) > simhash_similarity(sa, sc)


def test_hamming_distance():
    assert hamming_distance(0b1010, 0b0110) == 2
    assert hamming_distance(0, 0) == 0


def test_ngrams_2():
    assert ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
