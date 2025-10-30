# -*- coding: utf-8 -*-
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

print("--- Running Task 1.4 (Standalone V3 - ordered) ---")

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent

# Inputs
INPUT_POS_CLEANED = BASE_DIR / "positions_canonical.parquet"
INPUT_CAN_CLEANED = BASE_DIR / "candidates_cleaned.parquet"
ONTOLOGY_DIR = BASE_DIR / "ontology"

# Outputs
OUTPUT_POS = BASE_DIR / "positions_rules_features.parquet"
OUTPUT_CAN = BASE_DIR / "candidates_rules_features.parquet"

# 1) Ontology load order
JSON_FILE_ORDER = [
    "skills.json",
    "certifications.json",
    "edu_levels.json",
    "edu_majors.json",
    "experience_tags.json",
    "languages.json",
]

# 2) Final Position column order
FINAL_POSITION_COLUMNS = [
    "position_id", "title", "industry", "managerial_level", "job_function",
    "published_date", "job_desc",
    "skills_req", "certifications_req", "edu_level_req", "edu_major_req",
    "experience_tags_req", "languages_req", "years_req_range",
]

# 3) Final Candidate column order
FINAL_CANDIDATE_COLUMNS = [
    "id", "educations", "experience",
    "skills", "certifications", "edu_level", "edu_majors",
    "experience_tags", "languages", "years_exp",
]

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# ======================================================
# Task 1.4: Feature extraction helpers
# ======================================================
EDU_LEVEL_RANKING = {"Tertiary": 1, "Associate/Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
YEARS_REQ_REGEX = {
    "range_to": re.compile(r"(\d{1,2})\s*to\s*(\d{1,2})\s*years?", re.IGNORECASE),
    "range_dash": re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})\s*years?", re.IGNORECASE),
    "at_least": re.compile(r"at least\s*(\d{1,2})\s*years?", re.IGNORECASE),
    "minimum": re.compile(r"minimum\s*(\d{1,2})\s*years?", re.IGNORECASE),
    "plus": re.compile(r"(\d{1,2})\+\s*years?", re.IGNORECASE),
    "exact": re.compile(r"(\d{1,2})\s*years?", re.IGNORECASE),
}

def load_ontologies(directory: Path, file_order_list):
    """Load ontologies in the exact order specified, building synonym->canonical reverse maps."""
    print(f"--- Loading ontologies in order: {directory} ---")
    if not directory.exists():
        print(f"Error: Ontology folder does not exist: {directory}")
        return None

    ontology_maps = {}
    for file_name in file_order_list:
        file_path = directory / file_name
        ontology_name = file_name.split(".")[0]
        try:
            with file_path.open("r", encoding="utf-8") as f:
                original_dict = json.load(f)
            reverse_index_map = {}
            for canonical_term, synonym_list in original_dict.items():
                if isinstance(synonym_list, list):
                    for synonym in synonym_list:
                        synonym_lower = str(synonym).lower()
                        if synonym_lower in reverse_index_map:
                            print(f"Warning: Synonym '{synonym_lower}' in {file_name} maps to multiple canonical terms.")
                        reverse_index_map[synonym_lower] = canonical_term
            ontology_maps[ontology_name] = reverse_index_map
            print(f"  - Loaded {file_name} (synonyms: {len(reverse_index_map)})")
        except FileNotFoundError:
            print(f"Error: Required ontology file not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return None
    return ontology_maps

def build_regex_from_map(reverse_index_map):
    """Build a word-boundary regex that matches any synonym in the map."""
    if not reverse_index_map:
        return re.compile(r"a^")
    all_synonyms = sorted(reverse_index_map.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(term) for term in all_synonyms) + r")\b"
    return re.compile(pattern, re.IGNORECASE)

def extract_ontology_features(text, regex_pattern, reverse_index_map):
    """Return a list of canonical terms matched in the text."""
    if not text or pd.isna(text):
        return []
    found = regex_pattern.findall(str(text).lower())
    canonical = {reverse_index_map[s] for s in found if s in reverse_index_map}
    return list(canonical) if canonical else []

def extract_years_experience(text):
    """Extract years-of-experience requirements as [min_years, max_years]."""
    if not text or pd.isna(text):
        return [None, None]
    t = str(text).lower()
    for key in ["range_to", "range_dash", "at_least", "minimum", "plus", "exact"]:
        m = YEARS_REQ_REGEX[key].search(t)
        if m:
            if key in ["range_to", "range_dash"]:
                return [int(m.group(1)), int(m.group(2))]
            if key in ["at_least", "minimum", "plus"]:
                return [int(m.group(1)), None]
            if key == "exact":
                v = int(m.group(1))
                return [v, v]
    return [None, None]

def calculate_total_experience_years_v4_fixed(experience_list):
    """Total candidate experience in years using 'duration_days' across entries."""
    if not isinstance(experience_list, (list, np.ndarray)) or len(experience_list) == 0:
        return None
    total_days = 0.0
    for job in experience_list:
        if isinstance(job, dict):
            duration = job.get("duration_days")
            if duration is not None and not pd.isna(duration):
                try:
                    val = float(duration)
                    if np.isfinite(val):
                        total_days += val
                except Exception:
                    pass
    if total_days <= 0:
        return None
    return round(total_days / 365.25, 2)

def find_highest_edu_level_v4(found_levels):
    """Pick the highest education level from an iterable of matched levels."""
    if not found_levels:
        return None

    if isinstance(found_levels, (set, list, tuple, np.ndarray)):
        iterable = found_levels
    else:
        iterable = [found_levels]

    best = None
    best_rank = 0
    for level in iterable:
        if level is None:
            continue
        lvl = str(level)
        rank = EDU_LEVEL_RANKING.get(lvl, 0)
        if rank > best_rank:
            best_rank = rank
            best = lvl
    return best

# ======================================================
# Part A: Positions
# ======================================================
def process_positions(df, ontology_regex, ontology_maps):
    print(f"\n--- Part A: Processing {len(df)} positions ---")
    keys_needed = ["skills", "certifications", "experience_tags", "edu_levels", "edu_majors", "languages"]
    for key in keys_needed:
        if key not in ontology_regex:
            print(f"  Warning: Missing ontology '{key}' for Part A. Using empty pattern.")
            ontology_regex[key] = re.compile(r"a^")
            ontology_maps[key] = {}

    df["job_desc"] = df["job_desc"].fillna("").astype(str)

    df["skills_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["skills"], ontology_maps["skills"])
    )
    df["certifications_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["certifications"], ontology_maps["certifications"])
    )
    df["experience_tags_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["experience_tags"], ontology_maps["experience_tags"])
    )
    df["edu_level_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["edu_levels"], ontology_maps["edu_levels"])
    )
    df["edu_major_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["edu_majors"], ontology_maps["edu_majors"])
    )
    df["languages_req"] = df["job_desc"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["languages"], ontology_maps["languages"])
    )
    df["years_req_range"] = df["job_desc"].apply(extract_years_experience)
    print("Part A complete.")
    return df

# ======================================================
# Part B: Candidates
# ======================================================
def process_candidates(df, ontology_regex, ontology_maps):
    print(f"\n--- Part B: Processing {len(df)} candidates ---")
    keys_needed = ["skills", "certifications", "experience_tags", "edu_levels", "edu_majors", "languages"]
    for key in keys_needed:
        if key not in ontology_regex:
            print(f"  Warning: Missing ontology '{key}' for Part B. Using empty pattern.")
            ontology_regex[key] = re.compile(r"a^")
            ontology_maps[key] = {}
            singular_key = key.rstrip("s")
            if singular_key in ontology_regex:
                print(f"  -> Using singular ontology '{singular_key}' instead.")
                ontology_regex[key] = ontology_regex[singular_key]
                ontology_maps[key] = ontology_maps[singular_key]
            elif key == "experience_tags" and "experience" in ontology_regex:
                print(f"  -> Using 'experience' instead.")
                ontology_regex[key] = ontology_regex["experience"]
                ontology_maps[key] = ontology_maps["experience"]

    print("  - Preparing 'cv_text' and 'edu_text' ...")

    def create_combined_text_v4(row):
        edu_texts, exp_texts = [], []
        edu_list = row.get("educations")
        exp_list = row.get("experience")

        if isinstance(edu_list, (list, np.ndarray)):
            for edu in edu_list:
                if isinstance(edu, dict):
                    edu_texts.append(str(edu.get("studyfield", "")))
                    edu_texts.append(str(edu.get("institutename", "")))

        if isinstance(exp_list, (list, np.ndarray)):
            for exp in exp_list:
                if isinstance(exp, dict):
                    exp_texts.append(str(exp.get("jobtitletext", "")))
                    exp_texts.append(str(exp.get("roles", "")))

        edu_full_text = " ".join(filter(None, edu_texts))
        exp_full_text = " ".join(filter(None, exp_texts))
        cv_full_text = f"{edu_full_text} {exp_full_text}".strip()
        return pd.Series([cv_full_text, edu_full_text])

    df[["cv_text", "edu_text"]] = df.apply(create_combined_text_v4, axis=1)

    df["skills"] = df["cv_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["skills"], ontology_maps["skills"])
    )
    df["certifications"] = df["cv_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["certifications"], ontology_maps["certifications"])
    )
    df["experience_tags"] = df["cv_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["experience_tags"], ontology_maps["experience_tags"])
    )
    df["edu_majors"] = df["edu_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["edu_majors"], ontology_maps["edu_majors"])
    )
    df["languages"] = df["cv_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["languages"], ontology_maps["languages"])
    )

    found_levels = df["edu_text"].apply(
        lambda x: extract_ontology_features(x, ontology_regex["edu_levels"], ontology_maps["edu_levels"])
    )
    df["edu_level"] = found_levels.apply(find_highest_edu_level_v4)

    df["years_exp"] = df["experience"].apply(calculate_total_experience_years_v4_fixed)

    df = df.drop(columns=["cv_text", "edu_text"])
    print("Part B complete.")
    return df

# ======================================================
# Main
# ======================================================
def main():
    # 1) Load ontologies
    ontology_maps = load_ontologies(ONTOLOGY_DIR, JSON_FILE_ORDER)
    if not ontology_maps:
        print("Error: Ontology load failed. Aborting.")
        return

    # 1b) Build regex patterns
    ontology_regex = {name: build_regex_from_map(rmap) for name, rmap in ontology_maps.items()}

    # 2) Positions
    try:
        df_pos = pd.read_parquet(INPUT_POS_CLEANED)  # requires pyarrow/fastparquet
        print(f"\nLoaded cleaned {INPUT_POS_CLEANED.name} ({len(df_pos)} rows).")

        df_pos_features = process_positions(df_pos.copy(), ontology_regex, ontology_maps)

        print(f"Reordering columns for {OUTPUT_POS.name} ...")
        missing_cols_pos = [c for c in FINAL_POSITION_COLUMNS if c not in df_pos_features.columns]
        if missing_cols_pos:
            print(f"Warning: Missing columns {missing_cols_pos}; saving available columns.")
            final_cols_pos_present = [c for c in FINAL_POSITION_COLUMNS if c in df_pos_features.columns]
            df_pos_features = df_pos_features[final_cols_pos_present]
        else:
            df_pos_features = df_pos_features[FINAL_POSITION_COLUMNS]
            print("Position column order set.")

        df_pos_features.to_parquet(OUTPUT_POS, index=False)
        print(f"\n Saved position features: {OUTPUT_POS}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {INPUT_POS_CLEANED}")
        print("Please run Task 1.2 to generate positions_canonical.parquet first.")
    except ImportError:
        print("Error: Missing parquet engine (pyarrow/fastparquet). Install in your venv:")
        print("  python -m pip install -U pyarrow")
        print("or:")
        print("  python -m pip install -U fastparquet")
    except Exception as e:
        print(f"Fatal error while processing Positions: {e}")

    # 3) Candidates
    print(f"\n--- Part B: Loading cleaned {INPUT_CAN_CLEANED.name} ---")
    try:
        df_can = pd.read_parquet(INPUT_CAN_CLEANED)
        print(f"Loaded cleaned {INPUT_CAN_CLEANED.name} ({len(df_can)} rows).")

        df_can_features = process_candidates(df_can.copy(), ontology_regex, ontology_maps)

        print(f"Reordering columns for {OUTPUT_CAN.name} ...")
        missing_cols_can = [c for c in FINAL_CANDIDATE_COLUMNS if c not in df_can_features.columns]
        if missing_cols_can:
            print(f"Warning: Missing columns {missing_cols_can}; saving available columns.")
            final_cols_can = [c for c in FINAL_CANDIDATE_COLUMNS if c in df_can_features.columns]
            df_can_features = df_can_features[final_cols_can]
        else:
            df_can_features = df_can_features[FINAL_CANDIDATE_COLUMNS]
            print("Candidate column order set.")

        df_can_features.to_parquet(OUTPUT_CAN, index=False)
        print(f"\n Saved candidate features: {OUTPUT_CAN}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {INPUT_CAN_CLEANED}")
        print("Please run Task 1.1 to generate candidates_cleaned.parquet first.")
    except Exception as e:
        print(f"Fatal error while processing Candidates: {e}")

    print("\n" + "=" * 50)
    print("--- Task 1.4 complete ---")
    print(f"Outputs: {OUTPUT_POS}, {OUTPUT_CAN}")
    print("=" * 50)

if __name__ == "__main__":
    main()
