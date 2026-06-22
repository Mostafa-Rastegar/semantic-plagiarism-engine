"""
small dataset helpers: load text files from a folder, load pairs from csv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_corpus(folder: str | os.PathLike) -> Tuple[List[str], List[str]]:
    """
    read all .txt files under `folder` (non-recursive). returns (doc_ids, texts).
    doc_ids are file stems.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    docs = []
    ids = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in (".txt", ".md"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                # skip broken files but report
                print(f"[warn] could not read {p}: {e}")
                continue
            docs.append(txt)
            ids.append(p.stem)
    return ids, docs


def load_pairs_csv(
    path: str | os.PathLike,
    text_col_a: str,
    text_col_b: str,
    label_col: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    load a csv of (text_a, text_b, [label]) triples. drops empty rows.
    """
    df = pd.read_csv(path)
    needed = [text_col_a, text_col_b]
    if label_col is not None:
        needed.append(label_col)
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"column {c!r} not in csv (have: {list(df.columns)})")

    df = df[needed].copy()
    df.columns = ["text_a", "text_b"] + (["label"] if label_col else [])
    # drop NaN texts
    df = df.dropna(subset=["text_a", "text_b"])
    df["text_a"] = df["text_a"].astype(str)
    df["text_b"] = df["text_b"].astype(str)
    if label_col:
        df["label"] = df["label"].astype(int)
    if limit is not None and limit > 0:
        df = df.head(limit).reset_index(drop=True)
    return df.reset_index(drop=True)
