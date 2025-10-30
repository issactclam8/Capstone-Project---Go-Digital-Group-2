# ontology_extract_terms_updated.py
# -*- coding: utf-8 -*-
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Paths are resolved relative to this script
BASE_DIR = Path(__file__).resolve().parent

POS_PATH = BASE_DIR / "positions_cleaned.parquet"
CAN_PATH = BASE_DIR / "candidates_cleaned.parquet"
OUT_DIR = BASE_DIR / "ontology"
OUT_FILE = OUT_DIR / "term_positions_candidates_updated.txt"

# TF-IDF parameters
NGRAM_RANGE = (1, 3)     # unigrams, bigrams, trigrams
MAX_FEATURES = 5000      # keep the top 5,000 n-grams
STOP_WORDS = "english"   # remove English stopwords

warnings.filterwarnings("ignore", module="sklearn")


def setup_directory():
    """Ensure the ontology/ directory exists (sibling to this script)."""
    if not OUT_DIR.exists():
        print(f"Creating directory: {OUT_DIR}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_corpus():
    """
    Load positions and candidates and build a unified text corpus.
    - positions_cleaned.parquet: use 'job_desc'
    - candidates_cleaned.parquet: collect 'roles' from each item in 'experience' (list)
    - candidates_cleaned.parquet: collect 'studyfield' from each item in 'educations' (list)
    """
    # positions
    print(f"Loading {POS_PATH} ...")
    if not POS_PATH.exists():
        print(f"Error: Not found {POS_PATH}")
        return None
    try:
        df_pos = pd.read_parquet(POS_PATH)  # requires pyarrow or fastparquet
    except ImportError:
        print("Error: Parquet engine (pyarrow/fastparquet) not installed. Install in your venv:")
        print("  python -m pip install -U pyarrow")
        print("or:")
        print("  python -m pip install -U fastparquet")
        return None
    except Exception as e:
        print(f"Error reading {POS_PATH.name}: {e}")
        return None

    if "job_desc" not in df_pos.columns:
        print("Warning: 'job_desc' missing in positions file. Using empty strings.")
        pos_text = pd.Series([""] * len(df_pos))
    else:
        pos_text = df_pos["job_desc"].fillna("").astype(str)

    # candidates
    print(f"Loading {CAN_PATH} ...")
    if not CAN_PATH.exists():
        print(f"Error: Not found {CAN_PATH}")
        return None
    try:
        df_can = pd.read_parquet(CAN_PATH)
    except ImportError:
        print("Error: Parquet engine (pyarrow/fastparquet) not installed. Install in your venv:")
        print("  python -m pip install -U pyarrow")
        print("or:")
        print("  python -m pip install -U fastparquet")
        return None
    except Exception as e:
        print(f"Error reading {CAN_PATH.name}: {e}")
        return None

    # Collect roles from 'experience'
    roles_list = []
    if "experience" in df_can.columns:
        for exp_list in df_can["experience"].dropna():
            if isinstance(exp_list, list):
                for job in exp_list:
                    if isinstance(job, dict) and isinstance(job.get("roles"), str):
                        roles_list.append(job["roles"])
    else:
        print("Warning: 'experience' column missing in candidates file. Skipping roles collection.")

    roles_text = pd.Series(roles_list, dtype="string").fillna("").astype(str)

    # Collect studyfield from 'educations'
    studyfield_list = []
    if "educations" in df_can.columns:
        for edu_list in df_can["educations"].dropna():
            if isinstance(edu_list, list):
                for edu in edu_list:
                    if isinstance(edu, dict) and isinstance(edu.get("studyfield"), str):
                        studyfield_list.append(edu["studyfield"])
    else:
        print("Warning: 'educations' column missing in candidates file. Skipping studyfield collection.")

    studyfield_text = pd.Series(studyfield_list, dtype="string").fillna("").astype(str)

    print(f"Loaded {len(pos_text)} job descriptions (JD).")
    print(f"Loaded {len(roles_text)} resume experience entries (CV roles).")
    print(f"Loaded {len(studyfield_text)} education majors (CV studyfield).")

    # Build corpus
    corpus = pd.concat([pos_text, roles_text, studyfield_text], ignore_index=True)
    corpus = corpus.astype(str).fillna("")

    if len(corpus) == 0 or all(s.strip() == "" for s in corpus[:100]):
        print("Warning: Corpus is empty or nearly all whitespace; 'empty vocabulary' may occur later.")
    print(f"Total corpus size: {len(corpus)} documents.")
    return corpus


def extract_and_rank_terms(corpus: pd.Series):
    """
    Use TF-IDF to select up to 5,000 representative n-grams,
    then count total frequency across the corpus and sort.
    """
    print(f"\n--- Running TfidfVectorizer ngram_range={NGRAM_RANGE}, max_features={MAX_FEATURES} ---")
    tfidf_vec = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        stop_words=STOP_WORDS,
        max_features=MAX_FEATURES
    )

    try:
        tfidf_vec.fit(corpus)
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            print("Error: The corpus is empty or stopwords removed all tokens.")
            return None
        raise

    top_terms = tfidf_vec.get_feature_names_out()
    print(f"Identified {len(top_terms)} candidate terms (n-grams).")

    print("\n--- Counting term frequencies with CountVectorizer ---")
    count_vec = CountVectorizer(vocabulary=top_terms)
    term_counts = count_vec.fit_transform(corpus)
    term_freq = term_counts.sum(axis=0).A1

    results_df = pd.DataFrame({"term": top_terms, "frequency": term_freq})
    results_df = results_df.sort_values("frequency", ascending=False, kind="mergesort").reset_index(drop=True)
    print("Finished frequency tally and sorting.")
    return results_df


def save_terms(results_df: pd.DataFrame):
    """Write the sorted term list to ontology/term_positions_candidates_updated.txt"""
    print(f"\n--- Saving candidate term list to: {OUT_FILE} ---")
    try:
        with OUT_FILE.open("w", encoding="utf-8") as f:
            for term in results_df["term"]:
                f.write(f"{term}\n")
        print("\n" + "=" * 50)
        print(f"✅ Success! Exported: {OUT_FILE}")
        print("Next step: Proceed to Step 3 – manual curation.")
        print("=" * 50)

        print("\n[ Top 20 by frequency — preview ]")
        head20 = results_df.head(20)
        try:
            print(head20.to_markdown(index=False))
        except Exception:
            print(head20.to_string(index=False))
    except Exception as e:
        print(f"Error while saving file: {e}")


if __name__ == "__main__":
    setup_directory()
    corpus = load_corpus()

    if corpus is not None:
        results_df = extract_and_rank_terms(corpus)
        if results_df is not None:
            save_terms(results_df)
        else:
            print("Term extraction failed. Aborting.")
    else:
        print("Failed to load corpus. Aborting.")
