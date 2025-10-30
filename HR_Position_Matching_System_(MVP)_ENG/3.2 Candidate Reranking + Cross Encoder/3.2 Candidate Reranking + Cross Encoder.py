import pandas as pd
import pickle
import json
import sys
import torch
import numpy as np
import math
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from pathlib import Path  # [NEW] For absolute path handling
from sklearn.preprocessing import minmax_scale  # [NEW] For Min-Max scaling


# --- Part 1: Scoring Helper Functions (from V4 Outline) ---
# These functions calculate the 13 rule-based scores.

def calculate_fulfillment_score(required_list, candidate_list):
    """
    [Fulfillment Score Logic]
    Calculates what percentage of requirements the candidate fulfills.
    """
    if not isinstance(required_list, (list, np.ndarray)):
        required_set = set()
    else:
        required_set = set(item for item in required_list if pd.notna(item))

    if not isinstance(candidate_list, (list, np.ndarray)):
        candidate_set = set()
    else:
        candidate_set = set(item for item in candidate_list if pd.notna(item))

    if not required_set:
        return 1.0  # 100% fulfilled "no requirements"

    intersection = len(required_set.intersection(candidate_set))
    return intersection / len(required_set)


def calculate_years_exp_match(pos_req_range, can_exp):
    """
    [Gradient Years Score Logic]
    Compares position's years_req_range with candidate's years_exp.
    """
    if not isinstance(pos_req_range, list) or len(pos_req_range) != 2:
        return 1.0  # Position has no requirement
    if can_exp is None or pd.isna(can_exp) or not isinstance(can_exp, (int, float)):
        return 0.0  # Candidate has no data

    min_req, max_req = pos_req_range[0], pos_req_range[1]

    # Case A: Min requirement only (e.g., [5, None])
    if min_req is not None and max_req is None:
        if can_exp >= min_req:
            return 1.0
        elif min_req - 2 <= can_exp < min_req:
            return 0.5  # Close (within 2 years)
        else:
            return 0.0

    # Case B: Range requirement (e.g., [5, 8])
    if min_req is not None and max_req is not None:
        if min_req <= can_exp <= max_req:
            return 1.0  # Perfect fit
        elif max_req < can_exp <= max_req + 5:
            return 1.0  # Slightly overqualified (5 years) is OK
        elif min_req - 2 <= can_exp < min_req:
            return 0.5  # Close
        else:
            return 0.0

    # Case C: Max requirement only (e.g., [None, 8])
    if min_req is None and max_req is not None:
        return 1.0 if can_exp <= max_req else 0.0

    return 1.0  # No requirement (e.g., [None, None])


def calculate_jaccard_similarity(list1, list2):
    """
    [Fallback Jaccard Logic]
    Used for fallback features like edu_major, languages, etc.
    """
    if not isinstance(list1, (list, np.ndarray)):
        set1 = set()
    else:
        set1 = set(item for item in list1 if pd.notna(item))

    if not isinstance(list2, (list, np.ndarray)):
        set2 = set()
    else:
        set2 = set(item for item in list2 if pd.notna(item))

    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# [Fallback Education Level Logic]
EDU_LEVEL_RANKING = {"Tertiary": 1, "Associate/Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5}


def calculate_edu_level_match(pos_req_levels, can_level):
    """
    Matches candidate education level against position requirements.
    """
    if not isinstance(pos_req_levels, list) or not pos_req_levels:
        return 1.0  # No requirement
    if not can_level or pd.isna(can_level):
        return 0.0  # Candidate has no data

    can_rank = EDU_LEVEL_RANKING.get(can_level, 0)

    for req in pos_req_levels:
        req_rank = EDU_LEVEL_RANKING.get(req, 0)
        if can_rank >= req_rank:
            return 1.0  # Candidate meets at least one requirement
    return 0.0


def create_cv_text(can_row):
    """
    [From 3_2B] Merges education and experience fields into a single cv_text.
    """
    texts = []

    if isinstance(can_row['experience'], (list, np.ndarray)):
        for exp in can_row['experience']:
            if isinstance(exp, dict):
                title = exp.get('jobtitletext')
                roles = exp.get('roles')
                if title: texts.append(str(title))
                if roles: texts.append(str(roles))

    if isinstance(can_row['educations'], (list, np.ndarray)):
        for edu in can_row['educations']:
            if isinstance(edu, dict):
                institute = edu.get('institutename')
                field = edu.get('studyfield')
                if institute: texts.append(str(institute))
                if field: texts.append(str(field))

    if isinstance(can_row['skills'], (list, np.ndarray)) and len(can_row['skills']) > 0:
        texts.append("Skills: " + ", ".join([str(s) for s in can_row['skills'] if pd.notna(s)]))

    return " ".join(texts)


def normalize_scores_per_position(scores):
    """
    [From 3_2B] Per-Position Min-Max Scaling to 0.0-1.0
    """
    scores = np.array(scores).reshape(-1, 1)  # Needs to be 2D for minmax_scale
    if scores.size == 0 or (scores.max() == scores.min()):
        return np.full(scores.shape[0], 0.5)  # Return 0.5 if all scores are identical

    normalized = minmax_scale(scores)
    return normalized.flatten()  # Return 1D array


# --- Part 2: Main Unified Reranking Script ---

if __name__ == "__main__":

    print("--- Task 3.2 (Unified): Reranking with 14 Features ---")

    # --- Step 1: Define Absolute Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    POS_FEATURES_FILE = BASE_DIR / "positions_FINAL_hybrid_features.parquet"
    CAN_FEATURES_FILE = BASE_DIR / "candidates_rules_features.parquet"
    RETRIEVAL_PKL_FILE = BASE_DIR / "retrieval_scores_hybrid.pkl"
    OUTPUT_JSON_FILE = BASE_DIR / "rerank_results_FINAL.json"  # The final output

    # --- Step 2: Load Cross-Encoder Model (to GPU) ---
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}. Loading model to cuda...")
        device = 'cuda'
        BATCH_SIZE = 256  # Adjust based on your 4090's VRAM
    else:
        print("Warning: No GPU detected. Loading model to cpu (will be slow).")
        device = 'cpu'
        BATCH_SIZE = 32

    try:
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device, max_length=512)
        print("Cross-Encoder model loaded successfully.")
    except Exception as e:
        print(f"[Fatal Error] Could not load Cross-Encoder model: {e}")
        print("Please ensure 'sentence-transformers' and 'torch' are installed.")
        sys.exit(1)

    # --- Step 3: Load All Data Files ---
    try:
        print(f"Loading position features: {POS_FEATURES_FILE}")
        df_pos = pd.read_parquet(POS_FEATURES_FILE)
        df_pos['position_id'] = df_pos['position_id'].astype(str)
        df_pos.set_index('position_id', inplace=True)

        print(f"Loading candidate features: {CAN_FEATURES_FILE}")
        df_can = pd.read_parquet(CAN_FEATURES_FILE)
        df_can['id'] = df_can['id'].astype(str)
        df_can.set_index('id', inplace=True)

        print(f"Loading retrieval results: {RETRIEVAL_PKL_FILE}")
        with open(RETRIEVAL_PKL_FILE, "rb") as f:
            retrieval_results = pickle.load(f)

        print("All data files loaded successfully.")

    except FileNotFoundError as e:
        print(f"[Fatal Error] File not found: {e.filename}")
        print("Please check your file paths.")
        sys.exit(1)
    except ImportError:
        print("[Fatal Error] 'pyarrow' library not found.")
        print("Please run: pip install pyarrow")
        sys.exit(1)
    except Exception as e:
        print(f"[Fatal Error] Loading data failed: {e}")
        sys.exit(1)

    # --- Step 4: Start the Unified Reranking Loop ---
    print(f"Starting unified reranking for {len(retrieval_results)} positions...")

    # Final dictionary to store all results
    rerank_results_data = {}

    for pos_id, candidates_list in tqdm(retrieval_results.items(), desc="Reranking Positions"):
        pos_id = str(pos_id)

        # 1. Get Position Data
        if pos_id not in df_pos.index:
            continue
        pos_row = df_pos.loc[pos_id]
        pos_text = pos_row.get('job_desc')
        if not pos_text or pd.isna(pos_text):
            continue  # Skip positions with no job description

        # --- Pass 1: Gather Data for Batch Prediction ---
        text_pairs = []  # To store [pos_text, can_text] for Cross-Encoder
        candidate_cache = []  # To store (can_id, can_row, text_sim_score)

        for can_tuple in candidates_list:
            can_id = str(can_tuple[0])
            text_sim_score = can_tuple[1]

            if can_id not in df_can.index:
                continue
            can_row = df_can.loc[can_id]

            # Create CV text for Cross-Encoder
            can_text = create_cv_text(can_row)
            if not can_text:
                can_text = "N/A"  # Use placeholder if CV is empty

            text_pairs.append([pos_text, can_text])
            candidate_cache.append((can_id, can_row, text_sim_score))

        if not candidate_cache:
            continue  # Skip if no valid candidates found

        # --- GPU Batch Prediction (Cross-Encoder) ---
        raw_cross_scores = model.predict(
            text_pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        )

        # --- Normalization (V4 Requirement) ---
        normalized_cross_scores = normalize_scores_per_position(raw_cross_scores)

        # --- Pass 2: Calculate All 13+1 Scores ---
        pos_rerank_list = []  # Stores final list for this position

        for i, (can_id, can_row, text_sim_score) in enumerate(candidate_cache):
            # A. LLM Features
            s_must_skill = calculate_fulfillment_score(pos_row['skills_req_must_have'], can_row['skills'])
            s_nice_skill = calculate_fulfillment_score(pos_row['skills_req_nice_to_have'], can_row['skills'])
            s_must_cert = calculate_fulfillment_score(pos_row['certs_req_must_have'], can_row['certifications'])
            s_nice_cert = calculate_fulfillment_score(pos_row['certs_req_nice_to_have'], can_row['certifications'])
            s_years = calculate_years_exp_match(pos_row['years_req_range'], can_row['years_exp'])
            s_focus = calculate_fulfillment_score(pos_row['role_focus_raw'], can_row['skills'])

            # B. Fallback Features
            s_all_skills = calculate_jaccard_similarity(pos_row['skills_req'], can_row['skills'])
            s_all_certs = calculate_jaccard_similarity(pos_row['certifications_req'], can_row['certifications'])
            s_exp_tags = calculate_jaccard_similarity(pos_row['experience_tags_req'], can_row['experience_tags'])
            s_edu_major = calculate_jaccard_similarity(pos_row['edu_major_req'], can_row['edu_majors'])
            s_lang = calculate_jaccard_similarity(pos_row['languages_req'], can_row['languages'])
            s_edu_level = calculate_edu_level_match(pos_row['edu_level_req'], can_row['edu_level'])

            # C. Retrieval Score (Feature 13)
            s_text_sim = text_sim_score

            # D. Cross-Encoder Score (Feature 14)
            s_cross_encoder = normalized_cross_scores[i]

            # 5. Combine all 14 scores into one dictionary
            scores_dict = {
                "must_have_skill_match": s_must_skill,
                "nice_to_have_skill_match": s_nice_skill,
                "must_have_cert_match": s_must_cert,
                "nice_to_have_cert_match": s_nice_cert,
                "years_match": s_years,
                "role_focus_match": s_focus,
                "all_skills_match": s_all_skills,
                "all_certs_match": s_all_certs,
                "exp_tags_match": s_exp_tags,
                "edu_level_match": s_edu_level,
                "edu_major_match": s_edu_major,
                "lang_match": s_lang,
                "text_sim_score": s_text_sim,
                "cross_encoder_score": s_cross_encoder
            }

            pos_rerank_list.append({
                "can_id": can_id,
                "scores": scores_dict
            })

        # Add the results for this position to the main dictionary
        rerank_results_data[pos_id] = pos_rerank_list

    # --- Step 5: Save the Final Unified File ---
    print(f"\nUnified reranking complete. Processed {len(rerank_results_data)} positions.")
    print(f"Saving final 14-feature results to: {OUTPUT_JSON_FILE}")

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(rerank_results_data, f, indent=4)

    print(f"\n--- Task 3.2 (Unified) Finished Successfully! ---")
    print(f"Your file '{OUTPUT_JSON_FILE.name}' is ready for Task 3.3.")