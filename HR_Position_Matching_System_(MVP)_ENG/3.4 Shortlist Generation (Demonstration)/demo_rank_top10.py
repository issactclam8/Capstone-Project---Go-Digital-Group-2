# -*- coding: utf-8 -*-
"""
Run Top-K ranking for ONE position_id without CLI arguments.
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# ====== PATH ======
RESULTS_JSON = "rerank_results_FINAL.json"        
WEIGHTS_JSON = "best_weights_14_features.json"    
POSITION_ID  = "3473415"
TOPK         = 10
OUTDIR       = "out/3473415_2025-10-25"           
PAUSE_ON_END = True
# =======================================================

def resolve_path(p: str, must_exist: bool = True) -> Path:
    """
    Resolve path with this priority:
      1) Absolute path as-is (expanduser + resolve)
      2) SCRIPT_DIR / p
      3) CWD / p  （最後嘗試）
    """
    candidate = Path(p)
    # absolute
    if candidate.is_absolute():
        abs_path = candidate.expanduser().resolve()
        if must_exist and not abs_path.exists():
            raise FileNotFoundError(f"Path not found: {abs_path}")
        return abs_path

    # try relative to script folder
    abs_path = (SCRIPT_DIR / candidate).resolve()
    if abs_path.exists() or not must_exist:
        return abs_path

    # fallback: relative to current working directory
    abs_path = (Path.cwd() / candidate).resolve()
    if must_exist and not abs_path.exists():
        raise FileNotFoundError(
            f"Path not found: {abs_path}\n"
            f"Tried:\n  - {SCRIPT_DIR / candidate}\n  - {Path.cwd() / candidate}"
        )
    return abs_path

def load_weights(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or len(obj) != 14:
        raise ValueError(f"Weights JSON must have 14 keys; got {len(obj)}.")
    return {k: float(v) for k, v in obj.items()}

def load_results(path: Path) -> Dict[str, List[Dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Results JSON must be a non-empty dict of {position_id: list(...)}")
    return data

def compute_totals_for_position(position_id: str,
                                data: Dict[str, List[Dict]],
                                weights: Dict[str, float],
                                feature_order: List[str]) -> pd.DataFrame:
    if position_id not in data:
        sample_ids = list(data.keys())[:10]
        raise KeyError(f"position_id '{position_id}' not found. Sample ids: {sample_ids}")
    rows = []
    for rec in data[position_id]:
        can_id = rec.get("can_id")
        scores = rec.get("scores", {}) or {}
        contrib = {feat: float(scores.get(feat, 0.0)) * float(w) for feat, w in weights.items()}
        total = sum(contrib.values())
        row = {"position_id": position_id, "can_id": can_id, "TotalScore": total}
        for feat in feature_order:
            row[feat] = float(scores.get(feat, 0.0))
            row[f"{feat}__contrib"] = contrib[feat]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("TotalScore", ascending=False).reset_index(drop=True)
    return df

def make_compact_view(df: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    base_cols = [
        "position_id","can_id","TotalScore",
        "cross_encoder_score","text_sim_score",
        "must_have_skill_match","nice_to_have_skill_match",
        "years_match","edu_major_match","lang_match","all_skills_match",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    compact = df[base_cols].head(topk).copy()
    contrib_cols = [c for c in df.columns if c.endswith("__contrib")]
    name_map = {c: c.replace("__contrib","") for c in contrib_cols}
    top_feats = []
    for _, row in df.head(topk).iterrows():
        parts = sorted(
            [(abs(row[c]), name_map[c], row[c]) for c in contrib_cols],
            key=lambda x: x[0], reverse=True
        )[:3]
        top_feats.append(" | ".join(f"{name}: {val:+.3f}" for _, name, val in parts))
    compact["top3_feature_contribs"] = top_feats
    return compact

def main():
    results_path = resolve_path(RESULTS_JSON, must_exist=True)
    weights_path = resolve_path(WEIGHTS_JSON, must_exist=True)
    outdir_path  = resolve_path(OUTDIR, must_exist=False)
    outdir_path.mkdir(parents=True, exist_ok=True)

    print(f"▶ SCRIPT DIR   : {SCRIPT_DIR}")
    print(f"▶ RESULTS JSON : {results_path}")
    print(f"▶ WEIGHTS JSON : {weights_path}")
    print(f"▶ OUTPUT DIR   : {outdir_path}")

    weights = load_weights(weights_path)
    feature_order = list(weights.keys())
    data = load_results(results_path)

    df_full = compute_totals_for_position(str(POSITION_ID), data, weights, feature_order)
    df_view = make_compact_view(df_full, topk=TOPK)

    full_path = (outdir_path / f"top{TOPK}_full_{POSITION_ID}.csv").resolve()
    view_path = (outdir_path / f"top{TOPK}_compact_{POSITION_ID}.csv").resolve()
    df_full.to_csv(full_path, index=False)
    df_view.to_csv(view_path, index=False)

    print("\n=== Top-K (compact) preview ===")
    try:
        print(df_view.to_string(index=False))
    except Exception:
        print(df_view.head(TOPK))

    print(f"\n Saved:\n  - {full_path}\n  - {view_path}")

    if PAUSE_ON_END:
        input("\nDone. Press Enter to close...")

if __name__ == "__main__":
    main()
