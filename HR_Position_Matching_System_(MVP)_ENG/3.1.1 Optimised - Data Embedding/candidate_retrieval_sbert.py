# -*- coding: utf-8 -*-
"""
Phase 3 / Task 3.1 (Upgrade): Candidate Retrieval with SBERT + Chroma
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ----------------------------
# Text assembly helpers
# ----------------------------
def _list_to_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        vals = [str(v) for v in x if str(v).strip()]
        return ", ".join(vals)
    return str(x)


def norm_space(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).replace("\n", " ").replace("\r", " ").split()).lower()


def build_position_text(row: pd.Series) -> str:
    """
    Build the position query text from FINAL hybrid features:
      - Base JD (prefer combined_text; fallback to title/job_desc/industry/job_function)
      - Augment with extracted features (must-have/nice-to-have/role_focus/edu/languages/tags)
    """
    base = ""
    if "combined_text" in row and isinstance(row["combined_text"], str) and row["combined_text"].strip():
        base = row["combined_text"]
    else:
        parts = []
        for col in ("title", "job_desc", "industry", "job_function"):
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                parts.append(str(row[col]))
        base = " | ".join(parts)

    segs = [f"query: {base}"]

    for k in [
        "skills_req_must_have", "skills_req_nice_to_have",
        "certs_req_must_have", "certs_req_nice_to_have",
        "role_focus_raw", "edu_level_req", "edu_major_req",
        "experience_tags_req", "languages_req",
    ]:
        if k in row:
            segs.append(f"{k}: {_list_to_str(row[k])}")

    return norm_space(" | ".join(segs))


def build_candidate_text(row: pd.Series) -> str:
    """
    Build the candidate passage from rules features:
      - Structured features (skills/certs/edu/years_exp/languages/tags)
      - Keep a stable prefix to help the model
    """
    segs = ["passage:"]
    for k in ["skills", "certifications", "edu_level", "edu_majors", "experience_tags", "languages"]:
        if k in row:
            segs.append(f"{k}: {_list_to_str(row[k])}")
    if "years_exp" in row and not pd.isna(row["years_exp"]):
        segs.append(f"years_exp: {row['years_exp']}")
    return norm_space(" | ".join(segs))


# ----------------------------
# Chroma helpers
# ----------------------------
def get_or_create_collection(client, name: str):
    try:
        col = client.get_collection(name=name)
    except Exception:
        col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col


def batched(iterable, n=512):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------------------
# Main flow
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="SBERT + Chroma Retrieval")
    BASE = Path(__file__).resolve().parent

    parser.add_argument("--positions", default=str((BASE / "positions_FINAL_hybrid_features.parquet").resolve()))
    parser.add_argument("--candidates", default=str((BASE / "candidates_rules_features.parquet").resolve()))
    parser.add_argument("--out_results", default=str((BASE / "retrieval_results.pkl").resolve()))
    parser.add_argument("--out_scores", default=str((BASE / "retrieval_scores.pkl").resolve()))
    parser.add_argument("--persist_dir", default=str((BASE / "chroma_candidates").resolve()))
    parser.add_argument("--collection", default="candidates_sbert_v1")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--limit_positions", type=int, default=None)
    parser.add_argument("--reset_index", action="store_true", help="Delete and rebuild the Chroma index")
    parser.add_argument("--batch", type=int, default=512, help="Batch size for embedding/upsert")
    args = parser.parse_args()

    pos_path = Path(args.positions)
    can_path = Path(args.candidates)
    out1 = Path(args.out_results)
    out2 = Path(args.out_scores)
    persist_dir = Path(args.persist_dir)

    print("--- Phase 3 / Task 3.1 (SBERT + Chroma) ---")
    print(f"positions = {pos_path}")
    print(f"candidates = {can_path}")
    print(f"persist_dir = {persist_dir}")
    print(f"model = {args.model}, topn = {args.topn}")

    if not pos_path.exists():
        raise FileNotFoundError(f"Positions file not found: {pos_path}")
    if not can_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {can_path}")

    # Load data
    df_pos = pd.read_parquet(pos_path)
    df_can = pd.read_parquet(can_path)

    # Column checks
    key_pos = "canonical_position_id" if "canonical_position_id" in df_pos.columns else "position_id"
    if key_pos not in df_pos.columns:
        raise ValueError("Positions file must contain canonical_position_id or position_id")
    if "id" not in df_can.columns:
        raise ValueError("Candidates file must contain 'id'")

    if args.limit_positions:
        df_pos = df_pos.head(int(args.limit_positions))

    # Build texts
    print("[TEXT] Building position texts ...")
    pos_ids = df_pos[key_pos].astype(int).tolist()
    pos_texts = [build_position_text(r) for _, r in df_pos.iterrows()]

    print("[TEXT] Building candidate texts ...")
    can_ids = df_can["id"].astype(int).tolist()
    can_texts = [build_candidate_text(r) for _, r in df_can.iterrows()]

    # Load model
    print("[EMBED] Loading SBERT model ...")
    model = SentenceTransformer(args.model)

    # Prepare Chroma
    if args.reset_index and persist_dir.exists():
        print("[INDEX] reset_index=True, removing previous index directory ...")
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)

    client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    col = get_or_create_collection(client, args.collection)

    # Upsert candidates if needed
    need_index = (col.count() == 0) or args.reset_index
    if need_index:
        print(f"[INDEX] Inserting candidate embeddings ({len(can_ids)} vectors) ...")
        total_batches = int(np.ceil(len(can_ids) / args.batch))
        for batch in tqdm(batched(list(zip(can_ids, can_texts)), n=args.batch), total=total_batches):
            ids_b = [str(cid) for cid, _ in batch]
            texts_b = [t for _, t in batch]
            embeds_b = model.encode(texts_b, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            col.add(ids=ids_b, documents=texts_b, embeddings=embeds_b)
    else:
        print(f"[INDEX] Candidate embeddings already present ({col.count()} vectors); skipping rebuild.")

    # Query per position
    print(f"[QUERY] Retrieving Top{args.topn} per position ...")
    # Pre-compute position embeddings
    pos_embeds = model.encode(pos_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    results_map: Dict[int, List[int]] = {}
    scores_map: Dict[int, List[Tuple[int, float]]] = {}

    for i, pid in tqdm(list(enumerate(pos_ids)), total=len(pos_ids)):
        q_emb = pos_embeds[i].tolist()
        q = col.query(query_embeddings=[q_emb], n_results=min(args.topn, len(can_ids)))
        # Chroma (space=cosine): distance = 1 - cosine_similarity
        got_ids = [int(x) for x in q["ids"][0]]
        dists = q["distances"][0]
        sims = [1.0 - float(d) for d in dists]

        results_map[int(pid)] = got_ids
        scores_map[int(pid)] = list(zip(got_ids, sims))

    # Save outputs
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    with open(out1, "wb") as f:
        pickle.dump(results_map, f, protocol=4)
    with open(out2, "wb") as f:
        pickle.dump(scores_map, f, protocol=4)

    print(f"Done: retrieval_results.pkl -> {out1}")
    print(f"Done: retrieval_scores.pkl  -> {out2}")
    print(f"Positions={len(pos_ids)} | Candidates={len(can_ids)} | TopN={args.topn}")
    print(f"Index dir: {persist_dir} | Collection: {args.collection}")


if __name__ == "__main__":
    main()
