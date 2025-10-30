import json
import optuna
import numpy as np
import sys
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from pathlib import Path  

# --- Absolute Paths ---
BASE_DIR = Path(__file__).resolve().parent

FINAL_RERANK_FILE = BASE_DIR / "rerank_results_FINAL.json"  
GROUND_TRUTH_FILE = BASE_DIR / "ground_truth_labels.json"  
OUTPUT_WEIGHTS_FILE = BASE_DIR / "best_weights_14_features.json"  

# --- Step 2: Define all 14 Feature Keys to be Tuned ---
WEIGHT_KEYS = [
    "must_have_skill_match",
    "nice_to_have_skill_match",
    "must_have_cert_match",
    "nice_to_have_cert_match",
    "years_match",
    "role_focus_match",
    "all_skills_match",
    "all_certs_match",
    "exp_tags_match",
    "edu_level_match",
    "edu_major_match",
    "lang_match",
    "text_sim_score",  
    "cross_encoder_score"  
]

# --- Step 3: Load Data ---
try:
    print(f"Loading 14-feature rerank data from: {FINAL_RERANK_FILE}")
    with open(FINAL_RERANK_FILE, 'r', encoding='utf-8') as f:
        # Structure: { "pos_id": [ {"can_id": "c1", "scores": {...}} ] }
        rerank_data = json.load(f)

    print(f"Loading ground truth labels from: {GROUND_TRUTH_FILE}")
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        # Structure: { "pos_id": [ {"can_id": "c1", "relevance": 3} ] }
        ground_truth_labels = json.load(f)

except FileNotFoundError as e:
    print(f"[Error] File not found: {e.filename}")
    print(
        "Please ensure rerank_results_FINAL.json and ground_truth_labels.json are in the same directory as this script.")
    sys.exit(1)

# --- Step 4: Pre-process Data for Fast Lookups ---
print("Pre-processing rerank scores for fast lookup...")
# Create a fast lookup map: { "pos_id": { "can_id": {14 scores} } }
rerank_lookup = {}
for pos_id, can_list in rerank_data.items():
    can_scores_map = {can['can_id']: can['scores'] for can in can_list}
    rerank_lookup[pos_id] = can_scores_map

# Filter Ground Truth to only include pos_ids that are in our rerank_lookup
# These are the actual positions we will use for tuning
valid_pos_ids = [
    pos_id for pos_id in ground_truth_labels
    if pos_id in rerank_lookup and ground_truth_labels[pos_id]
]
print(
    f"Data processing complete. Using {len(valid_pos_ids)} / {len(ground_truth_labels)} valid labeled positions for tuning.")


# --- Step 5: Define the Optuna "Objective Function" ---
def objective(trial):
    # 1. Suggest a new set of 14 weights from Optuna
    # A range from 0.0 (unimportant) to 5.0 (very important)
    weights = {}
    for key in WEIGHT_KEYS:
        # e.g., w_must_skill = trial.suggest_float("must_have_skill_match", 0.0, 5.0)
        weights[key] = trial.suggest_float(key, 0.0, 5.0)

    all_ndcg_scores = []  # Stores the nDCG score for each position

    # 2. Iterate through all positions in our Ground Truth
    for pos_id in valid_pos_ids:

        labels_for_pos = ground_truth_labels[pos_id]
        scores_for_pos = rerank_lookup[pos_id]

        y_true_relevance = []  # Stores the true relevance scores (e.g., [3, 0, 1, 2])
        y_pred_total_score = []  # Stores our calculated total scores (e.g., [5.8, 1.2, 3.4, 4.1])

        # 3. Iterate through all *labeled* candidates for this position
        for label_entry in labels_for_pos:
            can_id = str(label_entry['can_id'])  # Ensure ID is string
            true_relevance = label_entry['relevance']

            # 4. Check if this candidate exists in our rerank_data
            if can_id in scores_for_pos:

                # Get the 14 base scores
                base_scores = scores_for_pos[can_id]

                # --- Core Calculation: Apply weights to get Total Score ---
                total_score = 0.0
                for key in WEIGHT_KEYS:
                    # Get the score (default to 0.0 if key is missing)
                    feature_score = base_scores.get(key, 0.0)
                    total_score += weights[key] * feature_score

                # 5. Store the results
                y_true_relevance.append(true_relevance)
                y_pred_total_score.append(total_score)

        # 6. Calculate the nDCG@10 score for this single position
        if not y_true_relevance:
            continue  # Skip if this position had no valid labels

        # scikit-learn's ndcg_score expects 2D arrays
        # y_true_relevance = [3, 0, 1, 2] -> [[3, 0, 1, 2]]
        # y_pred_total_score = [5.8, 1.2, 3.4, 4.1] -> [[5.8, 1.2, 3.4, 4.1]]

        # Use k=10 to optimize for the Top 10 ranking quality
        ndcg_k = min(10, len(y_true_relevance))
        score = ndcg_score([y_true_relevance], [y_pred_total_score], k=ndcg_k)
        all_ndcg_scores.append(score)

    # 7. Calculate the "average nDCG" across all positions
    if not all_ndcg_scores:
        return 0.0  # Return 0.0 if no scores were calculated (avoids divide-by-zero)

    average_ndcg = np.mean(all_ndcg_scores)
    return average_ndcg


# --- Step 6: Run the Optuna Optimization Study ---
if __name__ == "__main__":

    if not valid_pos_ids:
        print(f"[Fatal Error] No valid labeled positions found in {GROUND_TRUTH_FILE}.")
        print("Please check your JSON file to ensure pos_ids and can_ids match rerank_results_FINAL.json.")
        sys.exit(1)

    print("\n--- Task 3.3: Optuna Parameter Tuning (14 Features) ---")

    # We want to "maximize" the nDCG score
    study = optuna.create_study(direction="maximize")

    # Run 1000 trials.
    # Start with 100 to test, then increase to 1000 or 2000 for better results.
    N_TRIALS = 1000

    # Show Optuna's logs
    optuna.logging.set_verbosity(optuna.logging.INFO)

    try:
        # Start the optimization
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n[Interrupted] Optimization stopped manually by user.")
    except Exception as e:
        print(f"\n[Error] An error occurred during optimization: {e}")
        import traceback

        traceback.print_exc()

    # --- Step 7: Display and Save the Best Results ---
    print("\n--- Optimization Complete ---")

    try:
        print(f"Best nDCG@{10} Score Achieved: {study.best_value:.6f}")

        print("\n--- Best Weights (Sorted by Importance) ---")
        best_weights = study.best_params

        # Format and print the weights
        max_key_len = max(len(key) for key in best_weights)
        for key, value in sorted(best_weights.items(), key=lambda item: item[1], reverse=True):
            print(f"  {key.ljust(max_key_len)}: {value:.6f}")

        print(f"\nSaving best weights to: {OUTPUT_WEIGHTS_FILE}")
        with open(OUTPUT_WEIGHTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(best_weights, f, indent=4)

        print("\n--- Task 3.3 Finished Successfully! ---")

    except ValueError:
        print("[Error] Optuna did not find any valid trials.")
        print("This might happen if ground_truth_labels.json is empty or all trials resulted in an error.")