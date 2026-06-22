"""
extract false positive / false negative examples for the report's error-analysis
section. uses the best-F1 threshold for each method, then prints the first few
disagreements.

run:
    python scripts/extract_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# import path (run from repo root)
sys.path.insert(0, "src")

from plagiarism_engine.preprocessing import preprocess_document, tokenize
from plagiarism_engine.minhash import MinHasher, estimated_jaccard
from plagiarism_engine.simhash import SimHasher, simhash_similarity


def run():
    df = pd.read_csv("data/raw/quora/train.csv", nrows=5000)
    df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
    df["question1"] = df["question1"].astype(str)
    df["question2"] = df["question2"].astype(str)
    labels = df["is_duplicate"].astype(int).values

    # ----- minhash -----
    mh = MinHasher(num_perm=128, seed=42)
    sh_a = [preprocess_document(t, shingle_size=2) for t in df["question1"]]
    sh_b = [preprocess_document(t, shingle_size=2) for t in df["question2"]]
    mh_sims = []
    for a, b in zip(sh_a, sh_b):
        mh_sims.append(estimated_jaccard(mh.signature(a), mh.signature(b)))
    df["mh_sim"] = mh_sims

    # ----- simhash -----
    tok_a = [tokenize(t) for t in df["question1"]]
    tok_b = [tokenize(t) for t in df["question2"]]
    sh = SimHasher(ngram=1, use_idf=True).fit(tok_a + tok_b)
    sigs_a = [sh.sign(t) for t in tok_a]
    sigs_b = [sh.sign(t) for t in tok_b]
    sh_sims = [simhash_similarity(a, b) for a, b in zip(sigs_a, sigs_b)]
    df["sh_sim"] = sh_sims

    # ----- choose best thresholds (from metrics.csv) -----
    mh_thr = 0.20
    sh_thr = 0.60

    df["mh_pred"] = (df["mh_sim"] >= mh_thr).astype(int)
    df["sh_pred"] = (df["sh_sim"] >= sh_thr).astype(int)

    df["mh_err"] = df["mh_pred"] != labels
    df["sh_err"] = df["sh_pred"] != labels

    out = {}
    for method, pred_col, sim_col in [("minhash", "mh_pred", "mh_sim"),
                                       ("simhash", "sh_pred", "sh_sim")]:
        fp = df[(df[pred_col] == 1) & (df["is_duplicate"] == 0)].head(5)
        fn = df[(df[pred_col] == 0) & (df["is_duplicate"] == 1)].head(5)
        out[method] = {
            "false_positives": [
                {
                    "q1": r["question1"],
                    "q2": r["question2"],
                    "similarity": round(float(r[sim_col]), 3),
                }
                for _, r in fp.iterrows()
            ],
            "false_negatives": [
                {
                    "q1": r["question1"],
                    "q2": r["question2"],
                    "similarity": round(float(r[sim_col]), 3),
                }
                for _, r in fn.iterrows()
            ],
        }

    # overlap: pairs where both got wrong
    both_wrong = df[df["mh_err"] & df["sh_err"]]
    out["both_wrong_count"] = int(len(both_wrong))
    out["total_pairs"] = int(len(df))

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/error_examples.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
