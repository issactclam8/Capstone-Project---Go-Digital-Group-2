import pandas as pd

# ----------------------------------------------------
file_to_check = 'applications-Accounting_Audit_Taxation.csv'
# The field combination to check for duplicates
key_columns = ['CANDIDATEID', 'POSITIONID']
# ----------------------------------------------------

def check_duplicate_applications():
    print(f"--- Starting duplicate check for file: {file_to_check} ---")

    try:
        df = pd.read_csv(file_to_check)
    except FileNotFoundError:
        print(f"Error: File '{file_to_check}' not found. Please ensure the file path is correct.")
        return
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return

    # Validate required columns
    if not all(col in df.columns for col in key_columns):
        print(f"Error: Missing required columns. Expected {key_columns}.")
        print(f"Detected columns: {df.columns.to_list()}")
        return

    print(f"Total application records: {len(df)}")

    # --- 1. Core duplicate check ---
    # Find duplicate combinations of (CANDIDATEID, POSITIONID).
    # keep=False marks all members of a duplicate group (including the first occurrence) as True.
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    has_duplicates = duplicate_mask.any()

    if not has_duplicates:
        print("\n Result: No duplicate applications for the same job were found.")
    else:
        print("\n Result: Duplicate applications for the same job were detected!")

        # --- 2. Show all duplicate rows ---
        duplicate_rows = df[duplicate_mask]
        # For easier review, sort by CANDIDATEID, POSITIONID, and APPLYDATE
        sorted_duplicates = duplicate_rows.sort_values(by=key_columns + ['APPLYDATE'])

        print(f"\n--- Detailed duplicate records (total {len(sorted_duplicates)}) ---")
        # Display options for completeness
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        print(sorted_duplicates)

        # --- 3. Provide a duplicate summary ---
        print("\n--- Duplicate summary (combination and count) ---")
        duplicate_summary = df.groupby(key_columns).size().reset_index(name='ApplyCount')

        # Only show groups with count > 1 (i.e., true duplicates)
        print(duplicate_summary[duplicate_summary['ApplyCount'] > 1])


if __name__ == "__main__":
    check_duplicate_applications()
