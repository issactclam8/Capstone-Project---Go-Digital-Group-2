# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# -----------------------------
# Helpers to build text fields
# -----------------------------
def safe_get(d: dict, key: str, default: str = "") -> str:
    v = d.get(key) if isinstance(d, dict) else None
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default


def normalize_space(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).replace("\n", " ").replace("\r", " ").split()).lower()


def build_position_text(row: pd.Series, prefer_col: str = "combined_text") -> str:
    # Prefer 'combined_text' if present; otherwise fall back to title + job_desc + industry/job_function
    if prefer_col in row and pd.notna(row[prefer_col]) and str(row[prefer_col]).strip():
        return normalize_space(str(row[prefer_col]))
    parts = []
    for col in ("title", "job_desc", "industry", "job_function"):
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    return normalize_space(" | ".join(parts))


def _concat_records(records: Optional[List[dict]], fields: List[str]) -> str:
    if not isinstance(records, list) or len(records) == 0:
        return ""
    chunks = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        vals = []
        for f in fields:
            val = safe_get(rec, f, "")
            if val:
                vals.append(val)
        if vals:
            chunks.append(", ".join(vals))
    return " | ".join(chunks)


def build_candidate_text(row: pd.Series) -> str:
    """
    Build a searchable string from candidates_cleaned:
      - educations: [achievedin, institutename, studyfield]
      - experience: [companyname, jobtitletext, roles]
    """
    edu_txt = _concat_records(row.get("educations"), ["achievedin", "institutename", "studyfield"])
    exp_txt = _concat_records(row.get("experience"), ["companyname", "jobtitletext", "roles"])
    parts = []
    if edu_txt:
        parts.append(f"edu: {edu_txt}")
    if exp_txt:
        parts.append(f"exp: {exp_txt}")
    if not parts:
        return "no_content"
    return normalize_space(" | ".join(parts))


# -----------------------------
# Core retrieval
# -----------------------------
def topn_indices_and_scores(
    X_pos: csr_matrix,
    X_cand: csr_matrix,
    topn: int,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (topn_idx, topn_scores)
      topn_idx:    shape=(num_pos, topn) candidate row indices
      topn_scores: matching cosine similarity scores
    """
    n_pos = X_pos.shape[0]
    n_cand = X_cand.shape[0]
    topn = min(topn, n_cand)

    # cosine = dot(normalized)
    X_pos = normalize(X_pos, norm="l2", axis=1, copy=False)
    X_cand = normalize(X_cand, norm="l2", axis=1, copy=False)

    idx_mat = np.zeros((n_pos, topn), dtype=np.int32)
    score_mat = np.zeros((n_pos, topn), dtype=np.float32)

    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        sim_block = X_pos[start:end].dot(X_cand.T)
        sim_np = sim_block.toarray() if hasattr(sim_block, "toarray") else np.asarray(sim_block)

        # argpartition to get candidate topn, then sort within those
        part = np.argpartition(-sim_np, kth=topn - 1, axis=1)[:, :topn]
        row_idx = np.arange(sim_np.shape[0])[:, None]
        row_scores = sim_np[row_idx, part]
        order = np.argsort(-row_scores, axis=1)

        top_idx = part[row_idx, order]
        top_scores = row_scores[row_idx, order]

        idx_mat[start:end, :] = top_idx.astype(np.int32)
        score_mat[start:end, :] = top_scores.astype(np.float32)

    return idx_mat, score_mat


def dedup_by_best_score(
    cand_ids_row: np.ndarray,
    scores_row: np.ndarray,
    topn: int,
) -> Tuple[List[int], List[float]]:
    """
    Deduplicate within a single position’s row.
    Keep the highest score per candidate_id; then return the top N by score.
    """
    best: Dict[int, float] = {}
    for cid, s in zip(cand_ids_row.tolist(), scores_row.tolist()):
        if (cid not in best) or (s > best[cid]):
            best[cid] = s
    items = sorted(best.items(), key=lambda x: -x[1])[:topn]
    out_ids = [cid for cid, _ in items]
    out_scores = [float(sc) for _, sc in items]
    return out_ids, out_scores


def main():
    parser = argparse.ArgumentParser(description="Phase 3 / Task 3.1 Retrieval (TF-IDF Baseline, v7)")
    BASE_DIR = Path(__file__).resolve().parent

    # Default paths: same folder
    parser.add_argument("--positions", default=str((BASE_DIR / "positions_canonical.parquet").resolve()))
    parser.add_argument("--candidates", default=str((BASE_DIR / "candidates_cleaned.parquet").resolve()))
    # v7: the only output
    parser.add_argument(
        "--out_scores",
        default=str((BASE_DIR / "retrieval_scores.pkl").resolve()),
        help="Output (candidate_id, score) list. Use 'None' to disable.",
    )

    parser.add_argument("--pos_id_col", default="canonical_position_id")
    parser.add_argument("--can_id_col", default="id")

    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--limit_positions", type=int, default=None, help="Run only the first N positions (for testing).")
    parser.add_argument("--pos_text_col", default="combined_text", help="Preferred position text column (default: combined_text).")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=200_000)
    parser.add_argument("--ngram", default="1,2", help="n-gram range, e.g., '1,2' or '1,3'.")
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    pos_path = Path(args.positions)
    cand_path = Path(args.candidates)

    # Allow disabling the output file by passing "None"
    out_scores_path = None if (args.out_scores == "None" or args.out_scores is None) else Path(args.out_scores)

    print("--- Running Phase 3 / Task 3.1: Retrieval (v7) ---")
    print(f"positions = {pos_path}")
    print(f"candidates = {cand_path}")

    if out_scores_path:
        print(f"out_scores = {out_scores_path}")
    else:
        print("out_scores = (disabled)")

    print(f"topn = {args.topn}, ngram={args.ngram}, min_df={args.min_df}, max_features={args.max_features}")
    print(f"[KEY] pos_id_col={args.pos_id_col} | can_id_col={args.can_id_col}")

    if not pos_path.exists():
        raise FileNotFoundError(f"Positions file not found: {pos_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {cand_path}")

    # Load data
    df_pos = pd.read_parquet(pos_path)
    df_cand = pd.read_parquet(cand_path)

    # Column checks
    if args.pos_id_col not in df_pos.columns:
        raise ValueError(f"Positions file missing column '{args.pos_id_col}'")
    if args.can_id_col not in df_cand.columns:
        raise ValueError(f"Candidates file missing column '{args.can_id_col}'")

    if args.limit_positions is not None:
        df_pos = df_pos.head(int(args.limit_positions))

    # Build text
    print("Building position texts ...")
    pos_texts = df_pos.apply(lambda r: build_position_text(r, prefer_col=args.pos_text_col), axis=1).tolist()
    print("Building candidate texts ...")
    cand_texts = df_cand.apply(build_candidate_text, axis=1).tolist()

    # Vectorize
    ngram = tuple(int(x) for x in args.ngram.split(","))
    print("Vectorizing (TF-IDF) ... this may take a while")
    vectorizer = TfidfVectorizer(
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_range=ngram,
        lowercase=False,  # we already normalized casing
        analyzer="word",
    )
    vectorizer.fit(cand_texts + pos_texts)

    # v7: removed code that saves the vectorizer

    X_cand = vectorizer.transform(cand_texts)  # (Nc, D)
    X_pos = vectorizer.transform(pos_texts)    # (Np, D)

    # Retrieval: indices + scores
    print("Computing similarities and extracting Top-N (with scores) ...")
    topn_idx, topn_scores = topn_indices_and_scores(X_pos, X_cand, topn=args.topn, batch_size=args.batch_size)

    cand_ids_all = df_cand[args.can_id_col].to_numpy()
    pos_keys = df_pos[args.pos_id_col].to_numpy()

    # Assemble output
    mapping_scores: Dict[int, List[Tuple[int, float]]] = {}

    for i, pk in enumerate(pos_keys):
        idx_row = topn_idx[i]
        scr_row = topn_scores[i]
        cand_ids_row = cand_ids_all[idx_row]

        # Deduplicate (keep highest score per candidate), then keep topn
        kept_ids, kept_scores = dedup_by_best_score(cand_ids_row, scr_row, topn=args.topn)

        if out_scores_path:
            mapping_scores[int(pk)] = [(int(cid), float(sc)) for cid, sc in zip(kept_ids, kept_scores)]

    # Save
    if out_scores_path:
        out_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_scores_path, "wb") as f:
            pickle.dump(mapping_scores, f, protocol=4)

    # Done
    if not out_scores_path:
        print("Done! (no output file configured)")
    else:
        print("Task 3.1 complete!")
        print(f"Scores output -> {out_scores_path}")

    print(f"Positions: {len(df_pos)} | Candidates: {len(df_cand)} | TopN (after dedup): <= {args.topn}")


if __name__ == "__main__":
    main()
