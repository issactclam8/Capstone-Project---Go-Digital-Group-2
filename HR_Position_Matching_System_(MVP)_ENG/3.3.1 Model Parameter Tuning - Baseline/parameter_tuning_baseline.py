import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
import warnings
import itertools
from pathlib import Path  # NEW: use the script's directory as the base

print("--- Running Phase 3 / Task 3.3: Parameter Tuning ---")

# Use the script's directory as the base
BASE_DIR = Path(__file__).resolve().parent  # NEW

# --- File definitions ---
POS_FEATURES_FILE = BASE_DIR / "positions_features.parquet"
CAN_FEATURES_FILE = BASE_DIR / "candidates_features.parquet"
RETRIEVAL_FILE = BASE_DIR / "retrieval_results.pkl"
APPLICATIONS_SPLIT_FILE = BASE_DIR / "applications_split.parquet"

OUTPUT_JSON = BASE_DIR / "best_weights_baseline.json"

# --- Parameters ---
EVAL_K = 10  # Our metric is Recall@10

# ======================================================
# Step 1: Define the weight search space (Grid)
# ======================================================
WEIGHT_SEARCH_SPACE = {
    "skills": [1.0, 1.5, 2.0],
    "certifications": [0.5, 1.0],
    "edu_level": [1.0, 1.5, 2.0],
    "edu_majors": [0.5, 1.0, 1.5],
    "experience_tags": [0.5, 1.0],
    "languages": [0.5, 1.0],
    "years_exp": [1.0, 1.5, 2.0],
}

# ======================================================
# Step 2: Copy scoring functions from Task 3.2
# ======================================================

EDU_LEVEL_RANKING = {
    "Tertiary": 1,
    "Associate/Diploma": 2,
    "Bachelor": 3,
    "Master": 4,
    "PhD": 5,
}


def calculate_jaccard_match(jd_list, cv_list):
    set_jd = set(jd_list) if jd_list is not None else set()
    set_cv = set(cv_list) if cv_list is not None else set()
    if not set_jd and not set_cv:
        return 0.0
    intersection = len(set_jd.intersection(set_cv))
    union = len(set_jd.union(set_cv))
    if union == 0:
        return 0.0
    return intersection / union


def calculate_exp_match(jd_years_range, cv_years):
    if cv_years is None or pd.isna(cv_years):
        return 0.0
    if jd_years_range is None or (jd_years_range[0] is None and jd_years_range[1] is None):
        return 0.5
    min_req, max_req = jd_years_range[0], jd_years_range[1]
    if min_req is not None and max_req is not None:
        if min_req <= cv_years <= max_req:
            return 1.0
        if (min_req - 2) <= cv_years < min_req:
            return 0.5
        return 0.0
    if min_req is not None and max_req is None:
        if cv_years >= min_req:
            return 1.0
        if (min_req - 2) <= cv_years < min_req:
            return 0.5
        return 0.0
    if min_req is None and max_req is not None:
        if cv_years <= max_req:
            return 1.0
        return 0.0
    return 0.0


def calculate_edu_level_match(jd_edu_req_list, cv_edu_level):
    if cv_edu_level is None or pd.isna(cv_edu_level):
        return 0.0
    if jd_edu_req_list is None or len(jd_edu_req_list) == 0:
        return 0.5
    cv_rank = EDU_LEVEL_RANKING.get(cv_edu_level, 0)
    min_req_rank = min([EDU_LEVEL_RANKING.get(lvl, 0) for lvl in jd_edu_req_list])
    if cv_rank >= min_req_rank:
        return 1.0
    return 0.0


def get_rerank_scores(pos_features, can_features, weights):
    scores = {}
    scores["skills"] = calculate_jaccard_match(pos_features.get("skills_req"), can_features.get("skills"))
    scores["certifications"] = calculate_jaccard_match(
        pos_features.get("certifications_req"), can_features.get("certifications")
    )
    scores["edu_majors"] = calculate_jaccard_match(pos_features.get("edu_major_req"), can_features.get("edu_majors"))
    scores["languages"] = calculate_jaccard_match(pos_features.get("languages_req"), can_features.get("languages"))
    scores["experience_tags"] = calculate_jaccard_match(
        pos_features.get("experience_tags_req"), can_features.get("experience_tags")
    )
    scores["years_exp"] = calculate_exp_match(pos_features.get("years_req_range"), can_features.get("years_exp"))
    scores["edu_level"] = calculate_edu_level_match(pos_features.get("edu_level_req"), can_features.get("edu_level"))

    total_score = 0.0
    for feature_name, sub_score in scores.items():
        total_score += sub_score * weights.get(feature_name, 0.0)
    scores["total_score"] = total_score
    return scores


# ======================================================
# Step 3: Evaluation function (V5 fixed version)
# ======================================================

def evaluate_weights(weights, validation_set, retrieval_results, pos_dict, can_dict):
    """
    [V5 Fixed] Use int consistently for all IDs.
    """
    hits = 0
    total = len(validation_set)

    # Iterate over (pos_id_float, true_can_id_float)
    for pos_id_float, true_can_id_float in validation_set:

        # --- [FIX V5] ---
        try:
            # 1) Cast to int for position lookup
            pos_id_num = int(pos_id_float)

            # 2) Cast to int for candidate lookup/comparison
            true_can_id_num = int(true_can_id_float)
        except (ValueError, TypeError):
            continue
        # --- [End FIX V5] ---

        # Get position features (keys are int)
        pos_features = pos_dict.get(pos_id_num)
        if not pos_features:
            continue

        # Get Top-200 retrieved candidate IDs (keys are int, values are list[int])
        retrieved_can_ids = retrieval_results.get(pos_id_num)

        # (1) Retrieval check
        if retrieved_can_ids is None or true_can_id_num not in retrieved_can_ids:
            continue

        # (2) Rerank check (retrieval was successful)
        rerank_list_for_pos = []
        for can_id_num in retrieved_can_ids:
            can_features = can_dict.get(can_id_num)
            if not can_features:
                continue

            scores = get_rerank_scores(pos_features, can_features, weights)
            rerank_list_for_pos.append((can_id_num, scores["total_score"]))

        sorted_rerank_list = sorted(rerank_list_for_pos, key=lambda x: x[1], reverse=True)
        final_top_k_ids = [can_id for can_id, score in sorted_rerank_list[:EVAL_K]]

        # Check if the ground-truth candidate appears in Top-K
        if true_can_id_num in final_top_k_ids:
            hits += 1  # Hit!

    if total == 0:
        return 0.0

    return hits / total


# ======================================================
# Step 4: Run grid search
# ======================================================

def main():
    try:
        print("Loading evaluation dataset (applications_split.parquet) ...")
        df_app = pd.read_parquet(APPLICATIONS_SPLIT_FILE)
        df_val = df_app[df_app["split"] == "validation"]

        # Validation set (pos_id, can_id) is float64
        validation_set = list(zip(df_val["POSITIONID"], df_val["CANDIDATEID"]))
        print(f"  - Loaded {len(validation_set)} validation pairs.")

        print(f"Loading position features: {POS_FEATURES_FILE}")
        df_pos = pd.read_parquet(POS_FEATURES_FILE)
        # Keys are int
        pos_features_dict = df_pos.set_index("position_id").to_dict("index")

        print(f"Loading candidate features: {CAN_FEATURES_FILE}")
        df_can = pd.read_parquet(CAN_FEATURES_FILE)
        # Keys are int
        can_features_dict = df_can.set_index("id").to_dict("index")

        print(f"Loading retrieval results: {RETRIEVAL_FILE}")
        with open(RETRIEVAL_FILE, "rb") as f:
            retrieval_results = pickle.load(f)  # keys: int, values: list[int]
        print("  - All files loaded.")

        # --- Prepare grid search ---
        weight_names = list(WEIGHT_SEARCH_SPACE.keys())
        weight_value_lists = list(WEIGHT_SEARCH_SPACE.values())

        all_weight_combinations = list(itertools.product(*weight_value_lists))
        total_combinations = len(all_weight_combinations)

        print("\n--- Starting grid search ---")
        print(f"  - Will test {total_combinations} weight combinations (expected 648) ...")

        best_score = -1.0
        best_weights = None

        # --- Iterate combinations ---
        for i, weight_values in enumerate(tqdm(all_weight_combinations, desc="Searching weights")):
            current_weights = dict(zip(weight_names, weight_values))

            recall_score = evaluate_weights(
                current_weights,
                validation_set,
                retrieval_results,
                pos_features_dict,
                can_features_dict,
            )

            if recall_score > best_score:
                best_score = recall_score
                best_weights = current_weights
                # Live print to console
                print(f"\n   New best! Recall@10: {best_score:.4f}")
                print(f"     Weights: {best_weights}")

        # --- Save results ---
        print("\n--- Grid search complete ---")
        print(f"  - Best Recall@10: {best_score:.4f}")
        print(f"  - Best weight combo: {best_weights}")

        print(f"  - Saving best weights to {OUTPUT_JSON} ...")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(best_weights, f, ensure_ascii=False, indent=2)

        print(f"\n Task 3.3 done! Best weights saved to: {OUTPUT_JSON}")
        print(f"\n--- Phase 3 (Matching Engine) complete! ---")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. Please check path: {e.filename}")
    except Exception as e:
        import traceback

        print(f"Critical error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
