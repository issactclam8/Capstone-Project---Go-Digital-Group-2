# -*- coding: utf-8 -*-
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

print("--- Running Phase 2 / Task 2.1: Prepare evaluation dataset (applications_split) ---")

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent

# Files (same folder)
INPUT_FILE = BASE_DIR / "applications-Accounting_Audit_Taxation.csv"
OUTPUT_FILE = BASE_DIR / "applications_split.parquet"

# Split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VALIDATION_RATIO

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


def main():
    try:
        # Step 1: Load and initial processing
        print(f"Loading: {INPUT_FILE} ...")
        if not INPUT_FILE.exists():
            raise FileNotFoundError

        # Encoding fallback: try utf-8-sig then latin-1
        try:
            df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(INPUT_FILE, encoding="latin-1")

        print(f"Loaded {len(df)} raw application records.")

        required_cols = {"CANDIDATEID", "POSITIONID", "APPLYDATE"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        print("  - Converting 'APPLYDATE' to datetime ...")
        df["APPLYDATE"] = pd.to_datetime(df["APPLYDATE"], errors="coerce")

        print("  - Dropping rows with nulls in 'CANDIDATEID', 'POSITIONID', or 'APPLYDATE' ...")
        initial_rows = len(df)
        df = df.dropna(subset=["CANDIDATEID", "POSITIONID", "APPLYDATE"])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"  - Removed {removed_rows} invalid rows due to nulls.")
        print(f"  - {len(df)} valid rows remain.")

        if len(df) == 0:
            raise ValueError("No usable records after cleaning; cannot split.")

        # Step 2: Deduplicate and sort
        print("  - Deduplicating on ('CANDIDATEID', 'POSITIONID'), keeping the earliest application ...")
        initial_rows = len(df)
        df = df.sort_values(by="APPLYDATE", ascending=True)
        df = df.drop_duplicates(subset=["CANDIDATEID", "POSITIONID"], keep="first")
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"  - Removed {removed_rows} duplicate applications.")
        print(f"  - {len(df)} unique application records remain.")

        print("  - Strictly sorting by 'APPLYDATE' ascending ...")
        df = df.sort_values(by="APPLYDATE", ascending=True).reset_index(drop=True)

        # Step 3: Split and label
        total_rows = len(df)
        print("  - Computing split indices and labeling 'train' / 'validation' / 'test' ...")

        train_end_index = int(total_rows * TRAIN_RATIO)
        val_end_index = int(total_rows * (TRAIN_RATIO + VALIDATION_RATIO))

        # Guardrails for very small datasets
        train_end_index = min(max(train_end_index, 0), total_rows)
        val_end_index = min(max(val_end_index, train_end_index), total_rows)

        print(f"    - Total: {total_rows}")
        print(f"    - Train: 0 → {train_end_index - 1} ({train_end_index})")
        print(f"    - Validation: {train_end_index} → {val_end_index - 1} ({val_end_index - train_end_index})")
        print(f"    - Test: {val_end_index} → {total_rows - 1} ({total_rows - val_end_index})")

        df["split"] = "test"
        if train_end_index < val_end_index:
            df.loc[train_end_index:val_end_index - 1, "split"] = "validation"
        if train_end_index > 0:
            df.loc[:train_end_index - 1, "split"] = "train"

        print("\n  - Split preview (head):")
        print(df[["APPLYDATE", "split"]].head(3))
        print("  - Split preview (tail):")
        print(df[["APPLYDATE", "split"]].tail(3))

        print("\n  - Actual proportions:")
        print(df["split"].value_counts(normalize=True).round(3))

        # Step 4: Save
        try:
            df.to_parquet(OUTPUT_FILE, index=False)
            print(f"\n Task 2.1 complete! Saved evaluation dataset to: {OUTPUT_FILE}")
        except ImportError:
            print("\nError: No Parquet engine installed (pyarrow or fastparquet). Please install one in your venv:")
            print("  python -m pip install -U pyarrow")
            print("or:")
            print("  python -m pip install -U fastparquet")

    except FileNotFoundError:
        print(f"Error: Input file not found {INPUT_FILE}")
        print("Ensure the file is in the same folder as this script, or adjust INPUT_FILE.")
    except Exception as e:
        print(f"Critical error during processing: {e}")


if __name__ == "__main__":
    main()
