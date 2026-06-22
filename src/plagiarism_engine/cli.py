"""
plagiarism_engine command-line interface.

three subcommands:
  compare  : compare two text files head-to-head
  corpus   : find near-duplicate pairs in a folder of .txt files (MinHash+LSH)
  pairs    : evaluate methods over a labeled csv (e.g. Quora) -> metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List

from . import __version__
from .dataset import load_corpus, load_pairs_csv
from .evaluation import (
    evaluate_minhash_pairs,
    evaluate_simhash_pairs,
    lsh_candidates_for_corpus,
)
from .lsh import LSHIndex, threshold_from_bands
from .minhash import MinHasher, estimated_jaccard
from .preprocessing import jaccard, preprocess_document, tokenize
from .simhash import SimHasher, simhash_similarity


# ----- shared -----

def _add_common_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shingle-size", type=int, default=3, help="word shingle size (default 3)")
    p.add_argument("--num-perm", type=int, default=128, help="MinHash signature length (default 128)")
    p.add_argument("--bands", type=int, default=32, help="LSH bands; must divide num-perm")
    p.add_argument("--ngram", type=int, default=1, help="SimHash n-gram size")
    p.add_argument("--seed", type=int, default=42)


# ----- compare -----

def cmd_compare(args: argparse.Namespace) -> int:
    fa = Path(args.file_a)
    fb = Path(args.file_b)
    if not fa.exists() or not fb.exists():
        print(f"file not found: {fa if not fa.exists() else fb}", file=sys.stderr)
        return 2

    text_a = fa.read_text(encoding="utf-8", errors="ignore")
    text_b = fb.read_text(encoding="utf-8", errors="ignore")

    # exact jaccard
    sh_a = preprocess_document(text_a, shingle_size=args.shingle_size)
    sh_b = preprocess_document(text_b, shingle_size=args.shingle_size)
    j_exact = jaccard(sh_a, sh_b)

    # minhash estimate
    mh = MinHasher(num_perm=args.num_perm, seed=args.seed)
    s_a = mh.signature(sh_a)
    s_b = mh.signature(sh_b)
    j_est = estimated_jaccard(s_a, s_b)

    # simhash similarity (idf fitted on the 2 docs)
    tok_a = tokenize(text_a)
    tok_b = tokenize(text_b)
    sh_engine = SimHasher(ngram=args.ngram, use_idf=True).fit([tok_a, tok_b])
    sig_a = sh_engine.sign(tok_a)
    sig_b = sh_engine.sign(tok_b)
    sim_sh = simhash_similarity(sig_a, sig_b)

    out = {
        "file_a": str(fa),
        "file_b": str(fb),
        "shingle_size": args.shingle_size,
        "num_perm": args.num_perm,
        "ngram": args.ngram,
        "jaccard_exact": round(j_exact, 4),
        "jaccard_minhash_estimate": round(j_est, 4),
        "simhash_similarity": round(sim_sh, 4),
        "simhash_sig_a_hex": f"{sig_a:016x}",
        "simhash_sig_b_hex": f"{sig_b:016x}",
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


# ----- corpus -----

def cmd_corpus(args: argparse.Namespace) -> int:
    ids, texts = load_corpus(args.data)
    if len(texts) < 2:
        print("need at least 2 docs in the folder", file=sys.stderr)
        return 2

    pairs, stats = lsh_candidates_for_corpus(
        texts,
        num_perm=args.num_perm,
        bands=args.bands,
        shingle_size=args.shingle_size,
        seed=args.seed,
    )
    # filter by threshold
    filtered = [(i, j, s) for (i, j, s) in pairs if s >= args.threshold]

    # write csv
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_a", "doc_b", "minhash_jaccard"])
        for i, j, s in filtered:
            w.writerow([ids[i], ids[j], f"{s:.4f}"])

    print(f"docs={len(texts)}  candidate_pairs={len(pairs)}  >=threshold({args.threshold})={len(filtered)}")
    print(f"lsh_stats={stats}")
    print(f"wrote {out_path}")
    return 0


# ----- pairs -----

def cmd_pairs(args: argparse.Namespace) -> int:
    df = load_pairs_csv(
        args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        limit=args.limit,
    )
    if "label" not in df.columns:
        print("pairs mode needs --label-col", file=sys.stderr)
        return 2

    # default threshold sweep grid (configurable via --thresholds)
    if args.thresholds:
        grid = [float(x) for x in args.thresholds.split(",")]
    else:
        grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rows = []
    print(f"running on {len(df)} pairs ...")
    for t in grid:
        r = evaluate_minhash_pairs(
            df,
            threshold=t,
            num_perm=args.num_perm,
            shingle_size=args.shingle_size,
            seed=args.seed,
        )
        rows.append(r.as_row())
        print(f"  minhash t={t:.2f}  P={r.precision:.3f}  R={r.recall:.3f}  F1={r.f1:.3f}  rt={r.runtime_sec:.2f}s")

    for t in grid:
        r = evaluate_simhash_pairs(df, threshold=t, ngram=args.ngram, use_idf=True)
        rows.append(r.as_row())
        print(f"  simhash t={t:.2f}  P={r.precision:.3f}  R={r.recall:.3f}  F1={r.f1:.3f}  rt={r.runtime_sec:.2f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out_path}")
    return 0


# ----- main -----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plagiarism-engine",
        description="Semantic duplicate & near-plagiarism detection CLI (MinHash+LSH and TF-IDF SimHash).",
    )
    p.add_argument("--version", action="version", version=f"plagiarism-engine {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # compare
    pc = sub.add_parser("compare", help="compare two text files")
    pc.add_argument("--file-a", required=True)
    pc.add_argument("--file-b", required=True)
    pc.add_argument("--output", default=None, help="optional json output path")
    _add_common_opts(pc)
    pc.set_defaults(func=cmd_compare)

    # corpus
    pcr = sub.add_parser("corpus", help="find near-dup pairs in a folder of .txt files")
    pcr.add_argument("--data", required=True, help="folder with .txt files")
    pcr.add_argument("--threshold", type=float, default=0.25, help="minhash jaccard threshold")
    pcr.add_argument("--output", default="outputs/candidates.csv")
    _add_common_opts(pcr)
    pcr.set_defaults(func=cmd_corpus)

    # pairs
    pp = sub.add_parser("pairs", help="evaluate methods on a labeled csv (Quora / Stack)")
    pp.add_argument("--pairs", required=True, help="csv path")
    pp.add_argument("--text-col-a", required=True)
    pp.add_argument("--text-col-b", required=True)
    pp.add_argument("--label-col", required=True)
    pp.add_argument("--limit", type=int, default=5000)
    pp.add_argument("--output", default="outputs/metrics.csv")
    pp.add_argument("--thresholds", default=None, help="comma-separated list, e.g. 0.2,0.4,0.6")
    _add_common_opts(pp)
    pp.set_defaults(func=cmd_pairs)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
