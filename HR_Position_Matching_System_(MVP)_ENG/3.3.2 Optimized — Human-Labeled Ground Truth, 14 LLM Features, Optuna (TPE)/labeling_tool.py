import pandas as pd
import json
import os
import sys
import random
import numpy as np  # 

# --- File paths ---
POS_FEATURES_FILE = "positions_FINAL_hybrid_features.parquet"
CAN_FEATURES_FILE = "candidates_rules_features.parquet"
RERANK_RESULTS_FILE = "rerank_results_FINAL.json"
GROUND_TRUTH_FILE = "ground_truth_labels.json"

# Label 5 positions first
TARGET_POSITIONS_TO_LABEL = 5
# For each position, sample 15 candidates to review 
CANDIDATES_TO_SAMPLE_PER_POSITION = 15


# --- Step 2: Helper functions ---

def load_data():
    """Load all required files."""
    print("--- Loading data ---")

    # 1) Load (or create) ground truth
    if os.path.exists(GROUND_TRUTH_FILE):
        print(f"Loading existing ground truth: {GROUND_TRUTH_FILE}")
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            ground_truth_labels = json.load(f)
    else:
        print(f"{GROUND_TRUTH_FILE} not found, creating a new one.")
        ground_truth_labels = {}

    # 2) Load rerank results from Task 3.2
    print(f"Loading rerank scores: {RERANK_RESULTS_FILE}")
    with open(RERANK_RESULTS_FILE, 'r', encoding='utf-8') as f:
        rerank_data = json.load(f)

    # 3) Load parquet files
    print(f"Loading position features: {POS_FEATURES_FILE}")
    df_pos_features = pd.read_parquet(POS_FEATURES_FILE)
    df_pos_features['position_id'] = df_pos_features['position_id'].astype(str)
    df_pos_features.set_index('position_id', inplace=True)

    print(f"Loading candidate features: {CAN_FEATURES_FILE}")
    df_can_features = pd.read_parquet(CAN_FEATURES_FILE)
    df_can_features['id'] = df_can_features['id'].astype(str)
    df_can_features.set_index('id', inplace=True)

    print("--- All data loaded ---")
    return ground_truth_labels, rerank_data, df_pos_features, df_can_features


def save_ground_truth(data):
    """Persist labels to JSON."""
    with open(GROUND_TRUTH_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"\n[Progress Saved] Wrote labels for {len(data)} positions to {GROUND_TRUTH_FILE}")


def format_list_or_set(data):
    """
    [FIXED] Convert parquet-loaded list/array/None to a clean string.
    This function correctly handles empty numpy arrays.
    """

    # 1) Prioritize numpy arrays (root cause of previous errors)
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return "N/A"
        # Iterate array, cast to string, filter None/NaN
        items = [str(item) for item in data if pd.notna(item)]
        if not items:
            return "N/A"
        return ", ".join(items)

    # 2) Handle standard list / set
    if isinstance(data, (list, set)):
        if not data:  # Safe for lists/sets
            return "N/A"
        items = [str(item) for item in data if pd.notna(item)]
        if not items:
            return "N/A"
        return ", ".join(items)

    # 3) Handle None or a single NaN
    # Here, data is no longer array/list/set, so pd.isna is safe
    if data is None or pd.isna(data):
        return "N/A"

    # 4) Fallback (e.g., single string, number)
    return str(data)


def display_position(pos_row):
    """Print key info about a position to the terminal."""
    print("\n" + "=" * 80)
    print(f"Position ID:   {pos_row.name}")
    print(f"Job Title:     {pos_row['title']}")
    print("-" * 80)
    print(f" [Req] Must-have Skills: {format_list_or_set(pos_row['skills_req_must_have'])}")
    print(f" [Req] Nice-to-have Skills: {format_list_or_set(pos_row['skills_req_nice_to_have'])}")
    print(f" [Req] Years Range: {format_list_or_set(pos_row['years_req_range'])}")
    print(f" [Req] Education Level: {format_list_or_set(pos_row['edu_level_req'])}")
    print(f" [Req] Role Focus: {format_list_or_set(pos_row['role_focus_raw'])}")
    print("=" * 80 + "\n")


def display_candidate(can_row, scores, pos_row):
    """Show candidate scores and the raw data you requested."""
    print("\n" + "-" * 40 + f" Candidate ID: {can_row.name} " + "-" * 40)

    # 1) Show the 6 key scores
    print("[A] Core Scores:")
    print(f"  > Must-have Skills:   {scores['must_have_skill_match']:.2f}")
    print(f"  > Nice-to-have Skills:{scores['nice_to_have_skill_match']:.2f}")
    print(f"  > Years Match:        {scores['years_match']:.2f}")
    print(f"  > Text Similarity:    {scores['text_sim_score']:.2f}")
    print(f"  > Edu Level Match:    {scores['edu_level_match']:.2f}")
    print(f"  > Role Focus Match:   {scores['role_focus_match']:.2f}")

    # 2) Show the raw data you care about
    print("\n[B] Raw Data:")
    print(f"  > Candidate Education Level: {can_row['edu_level']}")
    print(f"  > Candidate Certifications:  {format_list_or_set(can_row['certifications'])}")
    print(f"  > Candidate Skills:          {format_list_or_set(can_row['skills'])}")
    print(f"  > Candidate Years of Exp:    {can_row['years_exp']}\n")


def sample_candidates(candidates_for_pos, num_to_sample=15):
    """
    Smart sampling: not purely random—take a mix of high/mid/low candidates.
    """
    if len(candidates_for_pos) <= num_to_sample:
        # If very few candidates, just shuffle them
        random.shuffle(candidates_for_pos)
        return candidates_for_pos

    # 1) Top by must-have skills
    sorted_by_must_skill = sorted(
        candidates_for_pos,
        key=lambda x: x['scores']['must_have_skill_match'],
        reverse=True
    )

    # 2) Top by text similarity (retrieval score)
    sorted_by_text_sim = sorted(
        candidates_for_pos,
        key=lambda x: x['scores']['text_sim_score'],
        reverse=True
    )

    # 3) Random sample
    random_sample = random.sample(candidates_for_pos, k=num_to_sample)

    # 4) Combine: 5 skill-top, 5 text-top, 5 random
    # Use a dict to deduplicate
    candidates_to_show = {}

    for can_data in sorted_by_must_skill[:5] + sorted_by_text_sim[:5] + random_sample:
        if len(candidates_to_show) < num_to_sample:
            candidates_to_show[can_data['can_id']] = can_data
        else:
            break

    # Convert dict back to list and shuffle to avoid labeling bias
    final_list = list(candidates_to_show.values())
    random.shuffle(final_list)
    return final_list


# --- Step 3: Main loop ---

def main():
    try:
        # 1) Load everything
        ground_truth_labels, rerank_data, df_pos, df_can = load_data()

    except FileNotFoundError as e:
        print(f"[FATAL] File not found: {e.filename}")
        print("Please ensure the .parquet and .json files are in the same folder.")
        sys.exit(1)
    except ImportError:
        print("[FATAL] Missing 'pyarrow' library.")
        print("In your PyCharm terminal, run: pip install pyarrow")
        sys.exit(1)

    # 2) Find positions that still need labeling
    all_pos_ids = list(rerank_data.keys())
    labeled_pos_ids = set(ground_truth_labels.keys())

    pos_ids_to_label = [pid for pid in all_pos_ids if pid not in labeled_pos_ids]
    random.shuffle(pos_ids_to_label)

    if not pos_ids_to_label:
        print("\n[Congrats] You’ve labeled all positions!")
        return

    if len(pos_ids_to_label) < TARGET_POSITIONS_TO_LABEL:
        print(f"\n[Note] Remaining unlabeled positions ({len(pos_ids_to_label)}) "
              f"is fewer than the target ({TARGET_POSITIONS_TO_LABEL}).")

    # 3) Begin labeling loop
    num_actually_labeled_pos = 0
    for pos_id in pos_ids_to_label:
        if num_actually_labeled_pos >= TARGET_POSITIONS_TO_LABEL:
            break  # Done with target

        try:
            pos_row = df_pos.loc[pos_id]
        except KeyError:
            print(f"Warning: Position {pos_id} not found in {POS_FEATURES_FILE}, skipping.")
            continue

        display_position(pos_row)

        candidates_for_pos = rerank_data[pos_id]  # list of up to 200 candidates
        candidates_to_label = sample_candidates(candidates_for_pos, CANDIDATES_TO_SAMPLE_PER_POSITION)

        pos_labels = []  # labels for this position

        print(f"--- Start labeling position {num_actually_labeled_pos + 1}/{TARGET_POSITIONS_TO_LABEL} (ID: {pos_id}) ---")
        print(f"Showing {len(candidates_to_label)} / {len(candidates_for_pos)} sampled candidates.")

        for i, can_data in enumerate(candidates_to_label):
            can_id = can_data['can_id']
            scores = can_data['scores']

            try:
                can_row = df_can.loc[can_id]
            except KeyError:
                print(f"Warning: Candidate {can_id} not found in {CAN_FEATURES_FILE}, skipping.")
                continue

            print(f"\n--- Candidate ({i + 1}/{len(candidates_to_label)}) ---")
            display_candidate(can_row, scores, pos_row)

            # 4) Get user input
            while True:
                inp = input(
                    "Enter rating (3=Perfect, 2=Good, 1=Acceptable, 0=Irrelevant, [s]=skip, [q]=finish this position): "
                ).strip().lower()

                if inp == 'q':
                    break  # finish this position early
                elif inp == 's' or inp == '':
                    print("... skipped ...")
                    break
                elif inp in ['0', '1', '2', '3']:
                    relevance = int(inp)
                    pos_labels.append({"can_id": can_id, "relevance": relevance})
                    print(f"... recorded: {can_id} -> {relevance} ...")
                    break
                else:
                    print("[Invalid] Please enter 3, 2, 1, 0, s, or q.")

            if inp == 'q':
                print(f"--- Finished position {pos_id} early ---")
                break

        # 5) Save results for this position
        if pos_labels:
            ground_truth_labels[pos_id] = pos_labels
            num_actually_labeled_pos += 1
            # Save after each position to avoid data loss
            save_ground_truth(ground_truth_labels)
        else:
            print(f"No new labels for position {pos_id}; nothing saved.")

    print("\n--- All labeling tasks complete ---")
    print(f"Labeled {num_actually_labeled_pos} new positions in total.")
    print(f"You can now inspect {GROUND_TRUTH_FILE} for results.")


if __name__ == "__main__":
    # [FIX] Removed the numpy import check here since it's already at the top
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[User Interrupt] Stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
