import pandas as pd
import numpy as np

# Display settings for readability
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 200)

print("--- Inspecting 'positions_cleaned.parquet' ---")
df_pos = pd.read_parquet('positions_cleaned.parquet')

# 1) Structure and dtypes
print("\n[Positions Info]")
print(df_pos.info())

# 2) Header and first rows
print("\n[Positions Head]")
print(df_pos.head())

# 3) Random samples of cleaned job_desc
print("\n[Job Description samples (random 5)]")
print(df_pos['job_desc'].sample(5))

print("\n" + "=" * 50 + "\n")

print("--- Inspecting 'candidates_cleaned.parquet' ---")
df_can = pd.read_parquet('candidates_cleaned.parquet')

# 1) Structure and dtypes
print("\n[Candidates Info]")
print(df_can.info())

# 2) Header
print("\n[Candidates Head]")
print(df_can.head())

# 3) Check 'experience' field results
print("\n[Experience samples (random up to 5 with data)]")
non_empty_exp = df_can[df_can['experience'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
if not non_empty_exp.empty:
    print(non_empty_exp['experience'].sample(min(5, len(non_empty_exp))))
else:
    print("No examples found with data in the 'experience' field.")

# 4) Check 'educations' field results
print("\n[Educations samples (random up to 5 with data)]")
non_empty_edu = df_can[df_can['educations'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
if not non_empty_edu.empty:
    print(non_empty_edu['educations'].sample(min(5, len(non_empty_edu))))
else:
    print("No examples found with data in the 'educations' field.")
