import json
import os
import time
from pathlib import Path
import pandas as pd

# Progress bar (tqdm is optional)
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: 'tqdm' not found. Progress bar will be disabled.")
    print("Tip: pip install tqdm")
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Core tools
try:
    from llm_extractor import extract_features_from_jd, parse_llm_output
except ImportError:
    print("Error: Could not import llm_extractor.py.")
    print("Make sure run_llm_pipeline.py and llm_extractor.py are in the same folder.")
    exit()

# Paths
BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_DIR = BASE_DIR / "ontology"
CACHE_DIR = BASE_DIR / "llm_cache"

# Inputs
INPUT_CANONICAL_POS = BASE_DIR / "positions_canonical.parquet"
SKILLS_ONTOLOGY_FILE = ONTOLOGY_DIR / "skills.json"
CERTS_ONTOLOGY_FILE = ONTOLOGY_DIR / "certifications.json"
INPUT_RULES_FEATURES_POS = BASE_DIR / "positions_rules_features.parquet"

# Output
OUTPUT_FINAL_FEATURES_NAME = "positions_FINAL_hybrid_features.parquet"

# LLM settings
MODEL_TO_USE = "qwen3:14b"

# Test mode
TEST_RUN = False
TEST_SAMPLE_SIZE = 10


def build_normalization_map(ontology_file_path: Path):
    normalization_map = {}
    if not ontology_file_path.exists():
        print(f"Warning: Ontology file not found: {ontology_file_path}")
        return None
    try:
        with open(ontology_file_path, "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        for standard_name, raw_terms in ontology_data.items():
            if isinstance(raw_terms, list):
                for term in raw_terms:
                    normalization_map[str(term).lower()] = standard_name
        print(f"Built normalization map: {ontology_file_path.name}")
        return normalization_map
    except Exception as e:
        print(f"Error reading {ontology_file_path.name}: {e}")
        return None


def normalize_feature_list(raw_list, normalization_map):
    if not isinstance(raw_list, list) or normalization_map is None:
        return set()
    normalized_set = set()
    for item in raw_list:
        if not isinstance(item, str):
            continue
        standard_name = normalization_map.get(item.lower())
        if standard_name:
            normalized_set.add(standard_name)
    return normalized_set


def run_llm_extraction(df_canonical: pd.DataFrame, cache_dir: Path):
    """
    Part 2: LLM extraction with cache.
    """
    print("\n--- Part 2: LLM extraction ---")

    id_column_name = "canonical_position_id"
    if id_column_name not in df_canonical.columns:
        print(f"Warning: '{id_column_name}' missing in positions_canonical.parquet.")
        print("Falling back to low-efficiency mode with 'position_id' (row-by-row).")
        id_column_name = "position_id"
        df_to_process = df_canonical
    else:
        df_to_process = df_canonical.drop_duplicates(subset=[id_column_name]).copy()
        if not TEST_RUN:
            print(
                f"Optimized: from {len(df_canonical)} total JDs, deduplicated to "
                f"{len(df_to_process)} unique JDs for the LLM."
            )

    print(f"Model: {MODEL_TO_USE} (API calls needed: {len(df_to_process)})")
    print(f"Cache directory: {cache_dir}")

    llm_results = []

    for _, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="LLM processing"):
        job_id = row[id_column_name]
        jd_text = row.get("combined_text")

        cache_file_path = cache_dir / f"{job_id}.json"

        # Cache hit
        if cache_file_path.exists():
            try:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    parsed_data = json.load(f)
                llm_results.append(parsed_data)
                continue
            except Exception as e:
                print(f"   [Cache Error] {job_id}: failed to read cache ({e}); re-calling API...")

        # Cache miss
        if not jd_text or not isinstance(jd_text, str) or len(jd_text) < 50:
            continue

        try:
            json_string = extract_features_from_jd(jd_text, model_name=MODEL_TO_USE)
            if not json_string:
                print(f"[Fail] {job_id}: LLM API call failed")
                continue

            parsed_data = parse_llm_output(json_string)
            if parsed_data:
                parsed_data[id_column_name] = job_id
                try:
                    with open(cache_file_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_data, f, indent=2)
                except Exception as e:
                    print(f"[Cache Error] {job_id}: failed to write cache ({e})")
                llm_results.append(parsed_data)
            else:
                print(f"[Fail] {job_id}: LLM output was not valid JSON.")
        except Exception as e:
            print(f"   [Error] {job_id}: unexpected error during processing: {e}")

    print(f"--- Part 2: Extraction complete. Success: {len(llm_results)} / {len(df_to_process)} unique JDs ---")

    if not llm_results:
        return pd.DataFrame(), id_column_name

    df_llm_raw = pd.DataFrame(llm_results)

    # Ensure expected columns exist
    expected_cols = [
        id_column_name,
        "years_min",
        "years_max",
        "must_have_skills_raw",
        "nice_to_have_skills_raw",
        "must_have_certs_raw",
        "nice_to_have_certs_raw",
        "role_focus_raw",
    ]
    for col in expected_cols:
        if col not in df_llm_raw.columns:
            df_llm_raw[col] = None

    return df_llm_raw, id_column_name


def run_normalization_and_merge(
    df_llm_raw: pd.DataFrame,
    df_rules_base: pd.DataFrame,
    df_canonical: pd.DataFrame,
    skills_map,
    certs_map,
    id_column_name: str,
):
    """
    Part 3: Normalize LLM outputs and merge with rule-based features.
    """
    print("\n--- Part 3: Normalization & merge ---")

    # Build years_req_range from min/max
    def create_year_range(row):
        y_min = row.get("years_min")
        y_max = row.get("years_max")
        if y_min is not None or y_max is not None:
            return [y_min, y_max]
        return None

    df_llm_raw["years_req_range"] = df_llm_raw.apply(create_year_range, axis=1)

    # Normalize skills
    if skills_map:
        print("   Normalizing skills...")
        df_llm_raw["skills_req_must_have"] = df_llm_raw["must_have_skills_raw"].apply(
            lambda x: normalize_feature_list(x, skills_map)
        )
        df_llm_raw["skills_req_nice_to_have"] = df_llm_raw["nice_to_have_skills_raw"].apply(
            lambda x: normalize_feature_list(x, skills_map)
        )
    else:
        df_llm_raw["skills_req_must_have"] = [set() for _ in range(len(df_llm_raw))]
        df_llm_raw["skills_req_nice_to_have"] = [set() for _ in range(len(df_llm_raw))]

    # Normalize certs
    if certs_map:
        print("   Normalizing certifications...")
        df_llm_raw["certs_req_must_have"] = df_llm_raw["must_have_certs_raw"].apply(
            lambda x: normalize_feature_list(x, certs_map)
        )
        df_llm_raw["certs_req_nice_to_have"] = df_llm_raw["nice_to_have_certs_raw"].apply(
            lambda x: normalize_feature_list(x, certs_map)
        )
    else:
        df_llm_raw["certs_req_must_have"] = [set() for _ in range(len(df_llm_raw))]
        df_llm_raw["certs_req_nice_to_have"] = [set() for _ in range(len(df_llm_raw))]

    # Role focus as a set
    if "role_focus_raw" not in df_llm_raw:
        df_llm_raw["role_focus_raw"] = None

    def to_set_if_list(item):
        if isinstance(item, list):
            return set(item)
        return set()

    df_llm_raw["role_focus_raw"] = df_llm_raw["role_focus_raw"].apply(to_set_if_list)

    # Select LLM features
    llm_features_to_merge = [
        id_column_name,
        "years_req_range",
        "skills_req_must_have",
        "skills_req_nice_to_have",
        "certs_req_must_have",
        "certs_req_nice_to_have",
        "role_focus_raw",
    ]
    df_llm_final = df_llm_raw.reindex(columns=llm_features_to_merge)

    # Merge with rule-based features
    print("   Merging rule-based features with LLM features...")
    df_final_combined = df_rules_base.copy()

    if id_column_name not in df_final_combined.columns:
        if id_column_name not in df_canonical.columns:
            print(f"Fatal: '{id_column_name}' also missing in df_canonical. Cannot merge.")
            return pd.DataFrame()
        print(f"Filling '{id_column_name}' via mapping from canonical file...")
        df_id_mapping = df_canonical[["position_id", "canonical_position_id"]].drop_duplicates()
        df_final_combined = pd.merge(df_final_combined, df_id_mapping, on="position_id", how="left")

    # Drop any existing columns that will be overwritten
    cols_to_overwrite = [c for c in llm_features_to_merge if c in df_final_combined.columns and c != id_column_name]
    if cols_to_overwrite:
        print(f"   Overwriting columns with LLM versions: {cols_to_overwrite}")
        df_final_combined = df_final_combined.drop(columns=cols_to_overwrite)

    df_final_combined = pd.merge(df_final_combined, df_llm_final, on=id_column_name, how="left")

    # Clean-up: ensure sets/lists are of the expected types
    list_cols_to_fill = [
        "skills_req_must_have",
        "skills_req_nice_to_have",
        "certs_req_must_have",
        "certs_req_nice_to_have",
        "role_focus_raw",
    ]
    for col in list_cols_to_fill:
        if col in df_final_combined.columns:
            df_final_combined[col] = df_final_combined[col].apply(lambda x: x if isinstance(x, set) else set())

    if "years_req_range" in df_final_combined.columns:
        df_final_combined["years_req_range"] = df_final_combined["years_req_range"].apply(
            lambda x: x if isinstance(x, list) else None
        )

    print("--- Part 3: Merge complete ---")
    return df_final_combined


def main():
    if TEST_RUN:
        print("=" * 50)
        print("Test mode is ON (TEST_RUN=True)")
        print(f"The pipeline will process only {TEST_SAMPLE_SIZE} unique JDs.")
        print("=" * 50)
    else:
        print("Starting [Optimized - Cache Enabled] LLM Pipeline...")

    total_start_time = time.time()

    # Prepare cache directory
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Cache directory ready: {CACHE_DIR}")
    except Exception as e:
        print(f"Fatal: Unable to create cache directory {CACHE_DIR}. Error: {e}")
        return

    # Check required files
    print("\n--- Checking required files ---")
    required_files = [INPUT_CANONICAL_POS, INPUT_RULES_FEATURES_POS, SKILLS_ONTOLOGY_FILE, CERTS_ONTOLOGY_FILE]
    all_files_found = True
    for f in required_files:
        if not f.exists():
            print(f"Error: Missing file {f}")
            all_files_found = False
    if not all_files_found:
        print("Please verify the paths in the configuration section.")
        return

    # Load ontologies
    print("\n--- Part 1: Load ontologies (normalization) ---")
    skills_map = build_normalization_map(SKILLS_ONTOLOGY_FILE)
    certs_map = build_normalization_map(CERTS_ONTOLOGY_FILE)

    # Load data
    try:
        print("\n--- Loading Parquet files ---")
        df_canonical = pd.read_parquet(INPUT_CANONICAL_POS)
        print(f"Loaded LLM input source: {INPUT_CANONICAL_POS.name} ({len(df_canonical)} rows)")

        df_rules_base = pd.read_parquet(INPUT_RULES_FEATURES_POS)
        print(f"Loaded rules base: {INPUT_RULES_FEATURES_POS.name} ({len(df_rules_base)} rows)")
    except Exception as e:
        print(f"Error reading Parquet files: {e}")
        return

    if df_canonical.empty or "combined_text" not in df_canonical.columns or "canonical_position_id" not in df_canonical.columns:
        print(f"Error: {INPUT_CANONICAL_POS.name} is empty or missing 'combined_text' / 'canonical_position_id'.")
        return
    if df_rules_base.empty:
        print(f"Error: {INPUT_RULES_FEATURES_POS.name} is empty. Nothing to merge.")
        return

    # Apply test mode sampling
    if TEST_RUN:
        print(f"\n--- Test mode: sampling {TEST_SAMPLE_SIZE} unique JDs ---")
        df_canonical_unique_sample = df_canonical.drop_duplicates(subset=["canonical_position_id"]).head(TEST_SAMPLE_SIZE)
        sample_canonical_ids = df_canonical_unique_sample["canonical_position_id"].tolist()
        df_canonical = df_canonical[df_canonical["canonical_position_id"].isin(sample_canonical_ids)]
        sample_position_ids = df_canonical["position_id"].tolist()
        df_rules_base = df_rules_base[df_rules_base["position_id"].isin(sample_position_ids)]
        print("Sample prepared:")
        print(f"     - Unique JDs (canonical_ids): {len(sample_canonical_ids)}")
        print(f"     - Related postings (position_ids): {len(df_canonical)}")
        print(f"     - Related rules features: {len(df_rules_base)}")

    # Part 2: LLM extraction
    df_llm_raw, id_column_name = run_llm_extraction(df_canonical, CACHE_DIR)

    # Part 3: Normalization & merge
    df_final_output = run_normalization_and_merge(
        df_llm_raw, df_rules_base, df_canonical, skills_map, certs_map, id_column_name
    )

    # Save results
    if TEST_RUN:
        output_file_path = BASE_DIR / f"positions_FINAL_TEST_SAMPLE_{TEST_SAMPLE_SIZE}.parquet"
    else:
        output_file_path = BASE_DIR / OUTPUT_FINAL_FEATURES_NAME

    try:
        print("\n--- Saving final output ---")
        df_final_output.to_parquet(output_file_path, index=False)
        print(f"Saved: {output_file_path.name}")
    except Exception as e:
        print(f"Error saving final Parquet: {e}")
        print("Tip: pip install pyarrow fastparquet")

    total_time_taken = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"Pipeline complete! Total time: {total_time_taken:.2f} seconds")
    if TEST_RUN:
        print(f"   (Test mode active. Check {output_file_path.name})")
    print("=" * 50)


if __name__ == "__main__":
    main()
