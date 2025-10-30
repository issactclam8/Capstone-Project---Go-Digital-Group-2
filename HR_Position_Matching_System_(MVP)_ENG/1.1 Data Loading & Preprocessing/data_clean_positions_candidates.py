# -*- coding: utf-8 -*-
import re
import json
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# Silence some BeautifulSoup warnings
warnings.filterwarnings("ignore", module="bs4")

# Resolve paths relative to this script's folder
BASE_DIR = Path(__file__).resolve().parent

POSITION_FILE_NAME = "position-Accounting_ Audit_Taxation.csv"
CANDIDATE_FILE_NAME = "candidate.jsonl"

OUTPUT_POSITION_FILE_NAME = "positions_cleaned.parquet"
OUTPUT_CANDIDATE_FILE_NAME = "candidates_cleaned.parquet"

# Full paths
POSITION_PATH = BASE_DIR / POSITION_FILE_NAME
CANDIDATE_PATH = BASE_DIR / CANDIDATE_FILE_NAME
OUTPUT_POSITION_PATH = BASE_DIR / OUTPUT_POSITION_FILE_NAME
OUTPUT_CANDIDATE_PATH = BASE_DIR / OUTPUT_CANDIDATE_FILE_NAME


def clean_job_desc(text):
    """
    Clean the 'job_desc' field:
    1) Remove HTML tags
    2) Lowercase
    3) Remove special characters (keep letters, digits, spaces)
    4) Collapse extra spaces
    """
    if not isinstance(text, str):
        return ""

    # 1) Strip HTML (prefer lxml; fallback to html.parser)
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # 2) Lowercase
    text = text.lower()

    # 3) Non-alphanumeric -> space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4) Normalize whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_experience_list(exp_list):
    """
    Process candidate.jsonl -> 'experience':
    - Parse expfrom/expto
    - Compute duration_days
    - Lowercase text fields (jobtitletext, roles)
    """
    if not isinstance(exp_list, list):
        return []

    processed_list = []
    for job in exp_list:
        if not isinstance(job, dict):
            continue

        new_job = job.copy()

        exp_from = pd.to_datetime(new_job.get("expfrom"), errors="coerce")
        exp_to = pd.to_datetime(new_job.get("expto"), errors="coerce")

        duration_days = pd.NA
        if pd.notna(exp_from) and pd.notna(exp_to) and exp_to >= exp_from:
            duration_days = (exp_to - exp_from).days
        new_job["duration_days"] = duration_days

        if isinstance(new_job.get("jobtitletext"), str):
            new_job["jobtitletext"] = new_job["jobtitletext"].lower()
        if isinstance(new_job.get("roles"), str):
            new_job["roles"] = new_job["roles"].lower()

        processed_list.append(new_job)
    return processed_list


def process_education_list(edu_list):
    """
    Process candidate.jsonl -> 'educations':
    - Lowercase studyfield and institutename
    """
    if not isinstance(edu_list, list):
        return []

    processed_list = []
    for edu in edu_list:
        if not isinstance(edu, dict):
            continue

        new_edu = edu.copy()
        if isinstance(new_edu.get("studyfield"), str):
            new_edu["studyfield"] = new_edu["studyfield"].lower()
        if isinstance(new_edu.get("institutename"), str):
            new_edu["institutename"] = new_edu["institutename"].lower()
        processed_list.append(new_edu)
    return processed_list


def main():
    # --- 1) Positions file ---
    print(f"Processing: {POSITION_PATH}")
    if not POSITION_PATH.exists():
        print(f"Error: File not found: {POSITION_PATH}")
    else:
        try:
            # Try UTF-8 (with BOM), fallback to latin-1 to avoid decode crashes
            try:
                df_pos = pd.read_csv(POSITION_PATH, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df_pos = pd.read_csv(POSITION_PATH, encoding="latin-1")

            # Clean 'job_desc'
            if "job_desc" in df_pos.columns:
                df_pos["job_desc"] = df_pos["job_desc"].apply(clean_job_desc)
            else:
                print("Warning: 'job_desc' column missing in positions file; skipping cleaning.")

            # Lowercase some other text columns if present
            for col in ["title", "industry", "managerial_level", "job_function"]:
                if col in df_pos.columns and df_pos[col].dtype == "object":
                    df_pos[col] = df_pos[col].fillna("").str.lower()

            try:
                df_pos.to_parquet(OUTPUT_POSITION_PATH, index=False)
                print(f"Saved: {OUTPUT_POSITION_PATH}")
            except ImportError:
                print("Error: No parquet engine found (pyarrow or fastparquet). Install one in your venv:")
                print("  python -m pip install -U pyarrow")
                print("or:")
                print("  python -m pip install -U fastparquet")
        except Exception as e:
            print(f"Error while processing positions file: {e}")

    # --- 2) Candidates file ---
    print(f"\nProcessing: {CANDIDATE_PATH}")
    if not CANDIDATE_PATH.exists():
        print(f"Error: File not found: {CANDIDATE_PATH}")
    else:
        try:
            df_can = pd.read_json(CANDIDATE_PATH, lines=True)

            if "experience" in df_can.columns:
                df_can["experience"] = df_can["experience"].apply(process_experience_list)
            else:
                print("Warning: 'experience' column missing in candidate file.")

            if "educations" in df_can.columns:
                df_can["educations"] = df_can["educations"].apply(process_education_list)
            else:
                print("Warning: 'educations' column missing in candidate file.")

            try:
                df_can.to_parquet(OUTPUT_CANDIDATE_PATH, index=False)
                print(f" Saved: {OUTPUT_CANDIDATE_PATH}")
            except ImportError:
                print("Error: No parquet engine found (pyarrow or fastparquet). Install one in your venv:")
                print("  python -m pip install -U pyarrow")
                print("or:")
                print("  python -m pip install -U fastparquet")
        except ValueError as e:
            print(f"Error reading JSONL (lines=True). Ensure each line is a valid JSON object: {e}")
        except Exception as e:
            print(f"Error while processing candidate file: {e}")


if __name__ == "__main__":
    main()
