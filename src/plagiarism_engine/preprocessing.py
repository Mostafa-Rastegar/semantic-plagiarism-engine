"""
preprocessing utilities.

we keep things simple: lowercase, strip punctuation, drop stopwords, tokenize,
then build word-level shingles. nothing fancy.
"""

import re
import string
import unicodedata
from typing import Iterable, List, Set


# stopwords. small lists, good enough for our experiments.
# english stopwords (NLTK-ish but trimmed by hand)
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "of", "as", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "them",
    "his", "her", "its", "their", "our", "your", "my", "me", "him",
    "what", "which", "who", "whom", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "would", "could", "may",
    "might", "must", "shall", "also", "there", "here", "where", "us",
}

# Persian/Farsi stopwords (small list)
FA_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای",
    "آن", "یک", "هم", "تا", "می", "بر", "تو", "نه", "یا", "اگر",
    "بود", "ها", "های", "شده", "شد", "خود", "ما", "شما", "آنها",
    "اما", "ولی", "چون", "وقتی", "بعد", "قبل", "روی", "زیر", "بالا",
    "پایین", "هر", "همه", "هیچ", "چه", "چی", "کجا", "کی", "چرا",
}


def _strip_diacritics(text: str) -> str:
    # remove unicode combining marks (Persian harakat etc)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """lowercase + remove diacritics + collapse whitespace."""
    if text is None:
        return ""
    text = str(text)
    text = _strip_diacritics(text)
    text = text.lower()
    # Persian arabic letter normalization (very small)
    text = text.replace("ي", "ی").replace("ك", "ک")
    # strip punctuation (keep word chars and spaces only). use unicode-safe regex.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stop: bool = True) -> List[str]:
    """split normalized text to tokens and optionally drop stopwords."""
    norm = normalize(text)
    if not norm:
        return []
    toks = norm.split(" ")
    if remove_stop:
        toks = [t for t in toks if t and t not in EN_STOPWORDS and t not in FA_STOPWORDS]
    else:
        toks = [t for t in toks if t]
    return toks


def shingles(tokens: Iterable[str], k: int = 3) -> Set[str]:
    """
    build word-level k-shingles. returns a set of joined strings.
    if doc shorter than k, fall back to single-token shingles so we still
    return something non-empty (otherwise jaccard explodes).
    """
    toks = list(tokens)
    if k < 1:
        k = 1
    if len(toks) < k:
        # short docs: shrink k
        if not toks:
            return set()
        k = min(k, len(toks))
    out = set()
    for i in range(len(toks) - k + 1):
        out.add(" ".join(toks[i : i + k]))
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    """plain Jaccard. used for baseline / sanity checks."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def preprocess_document(text: str, shingle_size: int = 3, remove_stop: bool = True) -> Set[str]:
    """one-stop: text -> set of shingles."""
    toks = tokenize(text, remove_stop=remove_stop)
    return shingles(toks, k=shingle_size)
