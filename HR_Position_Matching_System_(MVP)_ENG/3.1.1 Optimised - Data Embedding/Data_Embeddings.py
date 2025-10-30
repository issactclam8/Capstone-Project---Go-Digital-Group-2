# -*- coding: utf-8 -*-
"""
Plan B: Feature-Augmented Embeddings (positions FINAL hybrid + candidates rules)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# -------------------------
# Helpers
# -------------------------
def _to_list(x: Any) -> List[str]:
    """Coerce value to a list of non-empty strings."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(y) for y in x if y is not None and str(y).strip() != ""]
    s = str(x).strip()
    return [s] if s else []


def _years_minmax(years_range: Any) -> tuple[str, str]:
    """Extract (min, max) years from a [min, max] list where max may be None."""
    try:
        if isinstance(years_range, (list, tuple)) and len(years_range) == 2:
            mn = "" if years_range[0] is None else str(int(years_range[0]))
            mx = "" if years_range[1] is None else str(int(years_range[1]))
            return mn, mx
    except Exception:
        pass
    return "", ""


def _trim(s: str, max_chars: int) -> str:
    if not s:
        return ""
    return s if len(s) <= max_chars else s[:max_chars]


def _join(tag: str, items: List[str]) -> str:
    return f"{tag}: " + "; ".join(items) if items else ""


def _kv(tag: str, val: str) -> str:
    return f"{tag}: {val}" if val else ""


# -------------------------
# Build augmented text
# -------------------------
def build_position_aug_text(row: pd.Series, max_chars_query: int) -> str:
    """Compose position text + structured features into a single query string."""
    # Base text: combined_text, otherwise title + job_desc
    base = ""
    if "combined_text" in row and pd.notna(row["combined_text"]) and str(row["combined_text"]).strip():
        base = str(row["combined_text"])
    else:
        parts = []
        for col in ("title", "job_desc"):
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        base = " | ".join(parts)

    years_min, years_max = _years_minmax(row.get("years_req_range"))
    skills_must = _to_list(row.get("skills_req_must_have"))
    skills_nice = _to_list(row.get("skills_req_nice_to_have"))
    certs_must = _to_list(row.get("certs_req_must_have"))
    certs_nice = _to_list(row.get("certs_req_nice_to_have"))
    role_focus = _to_list(row.get("role_focus_raw"))
    edu_level  = _to_list(row.get("edu_level_req"))
    edu_major  = _to_list(row.get("edu_major_req"))
    exp_tags   = _to_list(row.get("experience_tags_req"))
    langs_req  = _to_list(row.get("languages_req"))

    blocks = [
        f"query: {_trim(base, max_chars_query)}",
        _join("must_have_skills", skills_must),
        _join("nice_to_have_skills", skills_nice),
        _join("must_have_certs", certs_must),
        _join("nice_to_have_certs", certs_nice),
        _kv("years_min", years_min),
        _kv("years_max", years_max),
        _join("role_focus", role_focus),
        _join("edu_level_req", edu_level),
        _join("edu_major_req", edu_major),
        _join("experience_tags_req", exp_tags),
        _join("languages_req", langs_req),
    ]
    blocks = [b for b in blocks if b and b.strip()]
    return " | ".join(blocks)


def build_candidate_aug_text(row: pd.Series, max_chars_passage: int) -> str:
    """Compose candidate features into a single passage string."""
    skills = _to_list(row.get("skills"))
    certs  = _to_list(row.get("certifications"))
    edu_lv = str(row.get("edu_level")) if pd.notna(row.get("edu_level")) else ""
    edu_mj = _to_list(row.get("edu_majors"))
    exp_tg = _to_list(row.get("experience_tags"))
    langs  = _to_list(row.get("languages"))
    years  = row.get("years_exp")
    years_s = "" if years is None or (isinstance(years, float) and np.isnan(years)) else f"{years:.2f}"

    blocks = [
        "passage:",
        _join("skills", skills),
        _join("certifications", certs),
        _kv("edu_level", edu_lv),
        _join("edu_majors", edu_mj),
        _join("experience_tags", exp_tg),
        _join("languages", langs),
        _kv("years_exp", years_s),
    ]
    blocks = [b for b in blocks if b and b.strip()]
    txt = " | ".join(blocks)
    return _trim(txt, max_chars_passage)


# -------------------------
# Encode embeddings
# -------------------------
def encode_texts(
    model_name: str,
    texts: List[str],
    batch_size: int = 128,
    device: str = "auto",
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts with SentenceTransformers and (optionally) L2-normalize."""
    model = SentenceTransformer(model_name, device=None if device == "auto" else device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalize explicitly below
        show_progress_bar=True,
    )
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    return embs


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Plan B: Feature-Augmented Embeddings")
    base = Path(__file__).resolve().parent

    # Inputs
    parser.add_argument("--positions", default=str((base / "positions_FINAL_hybrid_features.parquet").resolve()))
    parser.add_argument("--candidates", default=str((base / "candidates_rules_features.parquet").resolve()))

    # Outputs
    parser.add_argument("--out_pos_text", default=str((base / "positions_aug_text.parquet").resolve()))
    parser.add_argument("--out_can_text", default=str((base / "candidates_aug_text.parquet").resolve()))
    parser.add_argument("--out_pos_emb",  default=str((base / "positions_embeddings.npy").resolve()))
    parser.add_argument("--out_can_emb",  default=str((base / "candidates_embeddings.npy").resolve()))

    # Embedding config
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--max_chars_query", type=int, default=2000)
    parser.add_argument("--max_chars_passage", type=int, default=1500)

    # Optional limits
    parser.add_argument("--limit_positions", type=int, default=None)
    parser.add_argument("--limit_candidates", type=int, default=None)

    args = parser.parse_args()

    p_pos = Path(args.positions)
    p_can = Path(args.candidates)
    if not p_pos.exists():
        raise FileNotFoundError(f"positions file not found: {p_pos}")
    if not p_can.exists():
        raise FileNotFoundError(f"candidates file not found: {p_can}")

    print("=== Feature-Augmented Embeddings ===")
    print(f"Model={args.model} device={args.device} batch={args.batch_size}")
    print(f"Positions={p_pos}")
    print(f"Candidates={p_can}")

    # Load data
    df_pos = pd.read_parquet(p_pos)
    df_can = pd.read_parquet(p_can)

    if args.limit_positions:
        df_pos = df_pos.head(int(args.limit_positions))
    if args.limit_candidates:
        df_can = df_can.head(int(args.limit_candidates))

    # Determine position id column (canonical_position_id)
    pos_id_col = "canonical_position_id" if "canonical_position_id" in df_pos.columns else "position_id"
    if pos_id_col not in df_pos.columns:
        raise ValueError("positions parquet must contain 'canonical_position_id' or 'position_id'")
    if "id" not in df_can.columns:
        raise ValueError("candidates parquet must contain 'id'")

    # Build augmented texts
    print("[BUILD] positions augmented text ...")
    pos_texts = df_pos.apply(lambda r: build_position_aug_text(r, args.max_chars_query), axis=1).tolist()
    pos_df_out = pd.DataFrame({
        "canonical_position_id": df_pos[pos_id_col].astype(int).tolist(),
        "position_id": (
            df_pos["position_id"].astype(int).tolist()
            if "position_id" in df_pos.columns
            else df_pos[pos_id_col].astype(int).tolist()
        ),
        "aug_text": pos_texts,
    })
    Path(args.out_pos_text).parent.mkdir(parents=True, exist_ok=True)
    pos_df_out.to_parquet(args.out_pos_text, index=False)
    print(f"[OK] positions_aug_text -> {args.out_pos_text} rows={len(pos_df_out)}")

    print("[BUILD] candidates augmented text ...")
    can_texts = df_can.apply(lambda r: build_candidate_aug_text(r, args.max_chars_passage), axis=1).tolist()
    can_df_out = pd.DataFrame({
        "id": df_can["id"].astype(int).tolist(),
        "aug_text": can_texts,
    })
    Path(args.out_can_text).parent.mkdir(parents=True, exist_ok=True)
    can_df_out.to_parquet(args.out_can_text, index=False)
    print(f"[OK] candidates_aug_text -> {args.out_can_text} rows={len(can_df_out)}")

    # Encode
    print("[EMB] encoding positions ...")
    pos_emb = encode_texts(args.model, pos_texts, batch_size=args.batch_size, device=args.device, normalize=True)
    np.save(args.out_pos_emb, pos_emb)
    print(f"[OK] positions_embeddings.npy -> {args.out_pos_emb} shape={pos_emb.shape}")

    print("[EMB] encoding candidates ...")
    can_emb = encode_texts(args.model, can_texts, batch_size=args.batch_size, device=args.device, normalize=True)
    np.save(args.out_can_emb, can_emb)
    print(f"[OK] candidates_embeddings.npy -> {args.out_can_emb} shape={can_emb.shape}")

    print("DONE. You can now use these embeddings for retrieval/reranking.")


if __name__ == "__main__":
    main()