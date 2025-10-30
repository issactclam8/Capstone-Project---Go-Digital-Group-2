# -*- coding: utf-8 -*-
import warnings
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE_NAME = "positions_cleaned.parquet"
OUTPUT_FILE_NAME = "positions_canonical.parquet"

input_path = BASE_DIR / INPUT_FILE_NAME
output_path = BASE_DIR / OUTPUT_FILE_NAME

# Similarity threshold
SIMILARITY_THRESHOLD = 0.95

# Pandas display
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 200)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ------------------------------------------------------
# Steps 1 & 2: Load data and build 'combined_text'
# ------------------------------------------------------
def execute_steps_1_and_2():
    print(f"--- Task 1.2: Merge reposted (duplicate) positions ---")
    print(f"\n--- Step 1: Loading data ---")
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        print("Hint: Ensure 'positions_cleaned.parquet' is in the same folder as this .py, "
              "or update input_path to the correct location.")
        return None

    try:
        df = pd.read_parquet(input_path)  # requires pyarrow or fastparquet
    except ImportError:
        print("Error: Parquet engine (pyarrow/fastparquet) not installed. In your venv, run:")
        print("  python -m pip install -U pyarrow")
        print("or:")
        print("  python -m pip install -U fastparquet")
        return None

    required_cols = {"position_id", "title", "job_desc", "published_date"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {sorted(missing)}")
        return None

    if not pd.api.types.is_datetime64_any_dtype(df["published_date"]):
        print("Warning: 'published_date' is not datetime. Attempting conversion...")
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    print(f"OK: 'published_date' dtype = {df['published_date'].dtype}")

    print(f"\n--- Step 2: Create 'combined_text' ---")
    df["title"] = df["title"].fillna("")
    df["job_desc"] = df["job_desc"].fillna("")
    df["combined_text"] = (df["title"] + " " + df["job_desc"]).str.strip()

    print("OK: 'combined_text' created.")
    print("--- Steps 1 & 2 complete ---")
    return df


# ------------------------------------------------------
# Step 3: TF-IDF
# ------------------------------------------------------
def execute_step_3(df: pd.DataFrame):
    print(f"\n--- Step 3: Convert text to TF-IDF vectors ---")
    text_data = df["combined_text"]

    print("Init TfidfVectorizer(stop_words='english', max_features=5000, dtype=float32)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        dtype=np.float32
    )

    print("Running fit_transform...")
    try:
        tfidf_matrix = vectorizer.fit_transform(text_data).tocsr()
        print(f"OK: TF-IDF matrix built (shape={tfidf_matrix.shape})")
        print("--- Step 3 complete ---")
        return tfidf_matrix
    except Exception as e:
        print(f"\nError during TF-IDF transformation: {e}")
        return None


# ------------------------------------------------------
# Step 4: Similarity + Connected Components
# ------------------------------------------------------
def execute_step_4(df: pd.DataFrame, tfidf_matrix: sp.csr_matrix):
    print(f"\n--- Step 4: Compute cosine similarity and identify duplicate groups ---")

    print("Computing sparse cosine similarity via (tfidf @ tfidf.T)...")
    # O(n^2) sparse multiplication; very large data may be memory intensive
    sim_matrix = (tfidf_matrix @ tfidf_matrix.T).tocsr()

    print(f"Applying threshold >= {SIMILARITY_THRESHOLD} to build adjacency matrix...")
    adjacency = (sim_matrix >= SIMILARITY_THRESHOLD).tocsr()
    adjacency.setdiag(0)  # remove self-loops
    adjacency.eliminate_zeros()

    print("Finding connected components...")
    n_components, labels = csgraph.connected_components(
        csgraph=adjacency,
        directed=False,
        return_labels=True
    )

    print(f"\nOK! Grouped {len(df)} positions into {n_components} clusters.")
    df = df.copy()
    df["cluster_id"] = labels

    print("\n[ Cluster summary ]")
    cluster_summary = pd.Series(labels).value_counts()
    num_repost_clusters = (cluster_summary > 1).sum()
    num_repost_positions = cluster_summary[cluster_summary > 1].sum()
    print(f"  Singletons: {n_components - num_repost_clusters} clusters")
    print(f"  Reposts: {num_repost_clusters} clusters (total {num_repost_positions} rows)")

    print("--- Step 4 complete ---")
    return df, cluster_summary


# ------------------------------------------------------
# Steps 5 & 6: Assign canonical_position_id and save
# ------------------------------------------------------
def execute_steps_5_and_6(df: pd.DataFrame, cluster_summary: pd.Series):
    print(f"\n--- Step 5: Assign 'canonical_position_id' ---")

    print("Sorting by published_date, position_id (earliest becomes representative)...")
    df_sorted = df.sort_values(by=["published_date", "position_id"])

    print("Assigning representative ID via groupby('cluster_id')['position_id'].transform('first')...")
    canonical_ids = df_sorted.groupby("cluster_id")["position_id"].transform("first")
    # transform aligns by index back to the original DataFrame
    df["canonical_position_id"] = canonical_ids

    print("Assigned 'canonical_position_id'.")

    print("\n[ Preview top 5 of the largest cluster ]")
    if not cluster_summary.empty:
        example_cluster_id = cluster_summary.index[0]
        sample = (
            df[df["cluster_id"] == example_cluster_id]
            .loc[:, ["position_id", "published_date", "cluster_id", "canonical_position_id"]]
            .head(5)
        )
        try:
            print(sample.to_markdown(index=False))
        except Exception:
            print(sample)
    else:
        print("(No clusters to preview)")

    print(f"\n--- Step 6: Save results ---")
    try:
        df.to_parquet(output_path, index=False)
        print("\n" + "=" * 50)
        print("Success! Task 1.2 completed.")
        print(f"Output saved to: {output_path}")
        print("=" * 50)
    except Exception as e:
        print(f"Error while saving Parquet: {e}")


# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    df_processed = execute_steps_1_and_2()
    if df_processed is not None:
        tfidf_matrix = execute_step_3(df_processed)
        if tfidf_matrix is not None:
            df_clustered, cluster_summary = execute_step_4(df_processed, tfidf_matrix)
            if df_clustered is not None:
                execute_steps_5_and_6(df_clustered, cluster_summary)
            else:
                print("\n--- Step 4 failed ---")
        else:
            print("\n--- Step 3 failed ---")
    else:
        print("\n--- Aborting: Step 1 or 2 failed ---")

    end_time = time.time()
    print(f"\n--- Total runtime: {end_time - start_time:.2f} seconds ---")
