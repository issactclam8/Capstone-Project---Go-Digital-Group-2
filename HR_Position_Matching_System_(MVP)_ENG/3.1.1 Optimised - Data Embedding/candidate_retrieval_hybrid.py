# -*- coding: utf-8 -*-
"""
Phase 3 / Task 3.1 (Hybrid): 0.3 * TF-IDF  +  0.7 * SBERT
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_pickle(p: Path):
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def ensure_scores_from_results(results: Dict[int, List[int]], topn: int | None = None) -> Dict[int, List[Tuple[int, float]]]:
    """
    Convert rank-only results to (candidate_id, score) using Reciprocal Rank (1/rank).
    """
    out = {}
    for pid, cids in results.items():
        arr = cids[:topn] if topn else cids
        scored = [(int(cid), 1.0 / (i + 1)) for i, cid in enumerate(arr)]
        out[int(pid)] = scored
    return out


def to_dict(scores_list_map: Dict[int, List[Tuple[int, float]]]) -> Dict[int, Dict[int, float]]:
    """
    Convert {pid: [(cid, score), ...]} into {pid: {cid: score}}.
    If a cid appears multiple times, keep the larger score.
    """
    out: Dict[int, Dict[int, float]] = {}
    for pid, pairs in scores_list_map.items():
        d: Dict[int, float] = {}
        for cid, s in pairs:
            s = float(s)
            if cid in d:
                d[cid] = max(d[cid], s)
            else:
                d[cid] = s
        out[int(pid)] = d
    return out


def minmax_norm_per_query(scored: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize a single position's {cid: score} into [0, 1].
    If all values are constant, return all zeros.
    """
    if not scored:
        return {}
    vals = np.array(list(scored.values()), dtype=float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin + 1e-12:
        return {cid: 0.0 for cid in scored.keys()}
    return {cid: (float(s) - vmin) / (vmax - vmin) for cid, s in scored.items()}


def fuse_query(
    tfidf_q: Dict[int, float] | None,
    sbert_q: Dict[int, float],
    w_tfidf: float,
    w_sbert: float,
) -> Dict[int, float]:
    """
    Fuse scores for a single position.
    Both inputs are min-max normalized per query, then combined linearly.
    """
    sbert_n = minmax_norm_per_query(sbert_q)
    tfidf_n = minmax_norm_per_query(tfidf_q or {})

    all_cids = set(sbert_n.keys()) | set(tfidf_n.keys())
    fused: Dict[int, float] = {}
    for cid in all_cids:
        s = sbert_n.get(cid, 0.0)
        t = tfidf_n.get(cid, 0.0)
        fused[cid] = w_sbert * s + w_tfidf * t
    return fused


def topk_dict(d: Dict[int, float], k: int) -> List[Tuple[int, float]]:
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return items[:k]


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval score fusion")
    BASE = Path(__file__).resolve().parent

    # Default files
    parser.add_argument("--tfidf_results", default=str((BASE / "retrieval_results.pkl").resolve()))
    parser.add_argument("--tfidf_scores",  default=str((BASE / "retrieval_scores_tfidf.pkl").resolve()))
    parser.add_argument("--sbert_scores",  default=str((BASE / "retrieval_scores.pkl").resolve()))

    parser.add_argument("--out_scores",    default=str((BASE / "retrieval_scores_hybrid.pkl").resolve()))
    parser.add_argument("--out_results",   default=str((BASE / "retrieval_results_hybrid.pkl").resolve()))

    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--w_tfidf", type=float, default=0.3)
    parser.add_argument("--w_sbert", type=float, default=0.7)

    args = parser.parse_args()

    p_tfidf_results = Path(args.tfidf_results)
    p_tfidf_scores  = Path(args.tfidf_scores)
    p_sbert_scores  = Path(args.sbert_scores)

    print("--- Phase 3 / Task 3.1: Hybrid Fusion ---")
    print(f"TF-IDF results exists: {p_tfidf_results.exists()} | TF-IDF scores exists: {p_tfidf_scores.exists()}")
    print(f"SBERT scores: {p_sbert_scores}")
    print(f"Weights: TF-IDF={args.w_tfidf}  SBERT={args.w_sbert}  TopN={args.topn}")

    # Load SBERT scores (required)
    sbert_scores_list = load_pickle(p_sbert_scores)
    if sbert_scores_list is None:
        raise FileNotFoundError(f"SBERT score file not found: {p_sbert_scores}")

    # Load TF-IDF scores (optional)
    tfidf_scores_list = load_pickle(p_tfidf_scores)
    if tfidf_scores_list is None:
        # If TF-IDF scores not available, fall back to rank-only TF-IDF results (reciprocal rank)
        tfidf_results = load_pickle(p_tfidf_results)
        if tfidf_results is not None:
            tfidf_scores_list = ensure_scores_from_results(tfidf_results, topn=args.topn)
            print("[Info] TF-IDF scores not provided; derived from ranks via Reciprocal Rank.")
        else:
            tfidf_scores_list = {}

    # Convert to {pid: {cid: score}}
    sbert_scores = to_dict(sbert_scores_list)
    tfidf_scores = to_dict(tfidf_scores_list)

    # Union of all position IDs present in either set
    all_pids = set(sbert_scores.keys()) | set(tfidf_scores.keys())

    scores_hybrid: Dict[int, List[Tuple[int, float]]] = {}
    results_hybrid: Dict[int, List[int]] = {}

    for pid in all_pids:
        sbert_q = sbert_scores.get(pid, {})
        tfidf_q = tfidf_scores.get(pid, {})
        fused_q = fuse_query(tfidf_q, sbert_q, args.w_tfidf, args.w_sbert)
        topk = topk_dict(fused_q, args.topn)
        scores_hybrid[int(pid)] = topk
        results_hybrid[int(pid)] = [cid for cid, _ in topk]

    # Save outputs
    p_out_scores  = Path(args.out_scores)
    p_out_results = Path(args.out_results)
    with open(p_out_scores, "wb") as f:
        pickle.dump(scores_hybrid, f, protocol=4)
    with open(p_out_results, "wb") as f:
        pickle.dump(results_hybrid, f, protocol=4)

    # Summary
    n_pos = len(all_pids)
    avg_cands = (sum(len(v) for v in results_hybrid.values()) / max(1, n_pos))
    print(f"DONE. Hybrid scores  -> {p_out_scores}")
    print(f"DONE. Hybrid top-N   -> {p_out_results}")
    print(f"Positions={n_pos} | Avg TopN per position={avg_cands:.1f}")


if __name__ == "__main__":
    main()
