import pandas as pd
import os
from pathlib import Path

# Resolve paths relative to this script; fall back to current dir if not available (e.g., notebook)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    print("Warning: Could not detect script location; assuming current working directory.")
    BASE_DIR = Path(".")

INPUT_CANONICAL_POS = BASE_DIR / "positions_canonical.parquet"
CACHE_DIR = BASE_DIR / "llm_cache"

print("--- Identifying canonical_position_id that failed LLM processing ---")
print(f"Base directory: {BASE_DIR}")
print(f"Looking for Parquet file: {INPUT_CANONICAL_POS}")
print(f"Looking for cache directory: {CACHE_DIR}")

try:
    # 1) Read canonical file and collect all unique canonical_position_id
    if not INPUT_CANONICAL_POS.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_CANONICAL_POS}")

    df_canonical = pd.read_parquet(INPUT_CANONICAL_POS)
    all_canonical_ids = set(df_canonical["canonical_position_id"].unique())
    print(f"\n   Step 1: Found {len(all_canonical_ids)} unique canonical_position_id from {INPUT_CANONICAL_POS.name}.")

    # 2) Scan cache folder and collect all successfully saved canonical_position_id
    successful_ids = set()
    if CACHE_DIR.exists() and CACHE_DIR.is_dir():
        print(f"   Step 2: Scanning cache directory: {CACHE_DIR} ...")
        file_count = 0
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".json"):
                file_count += 1
                try:
                    job_id = int(filename[:-5])
                    successful_ids.add(job_id)
                except ValueError:
                    successful_ids.add(filename[:-5])
        print(f"          ... found {file_count} .json cache files.")
        print(f"          ... parsed {len(successful_ids)} unique successful_ids.")
    else:
        print(f"   Step 2: Error! Cache directory not found or not a directory: {CACHE_DIR}")

    # 3) Compare and find differences
    failed_ids = all_canonical_ids - successful_ids
    failed_count = len(failed_ids)

    print(f"\n--- Result ---")
    expected_success = len(successful_ids) if (CACHE_DIR.exists() and CACHE_DIR.is_dir()) else 0
    expected_failed_count = len(all_canonical_ids) - expected_success  # e.g., sanity check like 3878 - 3846 = 32

    if failed_count == expected_failed_count:
        print(f"Identified {failed_count} canonical_position_id that failed LLM processing.")
        if failed_count > 0:
            print("   Failed ID list:")
            failed_ids_list = sorted(list(failed_ids))
            print(failed_ids_list)

            # Optional: save failed IDs to CSV
            # try:
            #     output_csv_path = BASE_DIR / "llm_failed_ids.csv"
            #     pd.DataFrame({"failed_canonical_id": failed_ids_list}).to_csv(output_csv_path, index=False)
            #     print(f"\n   Failed ID list saved to {output_csv_path}")
            # except Exception as e:
            #     print(f"\n   Error while saving CSV: {e}")
        else:
            print("No failed JDs detected!")
    else:
        print(f"Warning: Computed failed count ({failed_count}) does not match expected ({expected_failed_count}).")
        print(f"Please verify the contents of the cache directory ({CACHE_DIR}) and ensure {INPUT_CANONICAL_POS.name} is up to date.")

except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}")
except KeyError as e:
    print(f"Error: Required column not found in {INPUT_CANONICAL_POS.name}: {e}")
    print("Ensure the file contains the 'canonical_position_id' column.")
except Exception as e:
    print(f"Unexpected error: {e}")
