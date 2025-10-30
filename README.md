# Capstone-Project---Go-Digital-Group-2

1. Project Objectives
We develop a recruiter-facing Retrieval→Rerank system for Accounting / Audit / Taxation (A/A/T) roles to reduce screening time and increase hit-rate. The pipeline first retrieves Top-200 candidates via hybrid lexical–semantic retrieval, then produces a ranked Top-10 using a 14-component weighted additive model plus a Cross-Encoder signal. The approach balances efficiency and accuracy, and exposes explainable component-level attributions per candidate.

Key innovations:
Hybrid feature extraction: rules/dictionary coverage + LLM extraction (hierarchical requirements)
Hybrid retrieval: 0.3 TF-IDF + 0.7 SBERT with per-query min–max normalization; HNSW via ChromaDB
Explainable reranker: 14 components (6 LLM + 6 Rules + 1 Retrieval + 1 Cross-Encoder)
Optuna tuning: TPE optimization on a curated 75-pair benchmark (objective: nDCG@10)

2. Dataset Description
Sources (scoped): Original raw data were positions.jsonl, candidate.jsonl, applications.csv. To keep the MVP focused and under 10k entries, we restricted the study to Accounting / Audit / Taxation roles.
Positions: position-Accounting_ Audit_Taxation (csv) → canonicalized.
Applications: applications-Accounting_Audit_Taxation (csv) → proxy positives.
Candidates: candidate.jsonl (unchanged).
2.1 Dataset Characteristics and Scale
1. Scale: 8,433 original JD records → 3,878 canonical unique positions after TF-IDF dedup (cosine sim > 0.95). 32 invalid entries dropped (format/critical fields missing).
2. Unstructured text: JD & CV content is largely free text (bilingual ZH/EN), requiring robust NLP cleaning and normalization.
3. Duplicates/reposts: frequent near-identical JDs under different IDs → dedup canonicalization.
4. Class imbalance: for a given position, positives (actual applicants) are sparse vs the large candidate pool.
5. Proxy label rationale: applications approximate relevance where hire/interview labels are absent.

Ref: folder 1.1/data_check_positions_candidates.py.

Data Preprocessing Steps
1. Ingestion & cleaning: robust encoding (UTF-8 w/ BOM → Latin-1 fallback), HTML stripping (BeautifulSoup), special-char removal, lower-casing (preserve acronyms e.g., CPA/ACCA), date parsing and employment duration calculations.
2. Deduplication (positions): TF-IDF over title + JD → cosine similarity; cluster at >0.95 and assign a canonical position_id per cluster (8,433 → 3,878).
3. Ontology build: six JSON dictionaries — skills, certifications, edu_level, edu_major, languages, experience_tags — with bilingual aliases and fuzzy mapping (edit-distance + synonym tables). Top n-grams (≈5k) seeded the term lists; curated iteratively.
4. Split: chronological 70/15/15 on application timestamps to avoid leakage and simulate deployment.

Ref: folder 1.2/position_deduplication_final.py
 
Ref: folder 1.3/Ontology Building; folder 1.3/Ontology Building/skills.json

3. Methodology
Phase 1: Data Foundation and Feature Engineering
This phase focuses on converting unstructured text in job descriptions and CVs into structured, comparable features. The key innovation is a hybrid feature extraction approach that combines the strengths of two complementary engines.

Rules-Based Extraction Engine
The rules engine is the foundation, running case-insensitive keyword matching with word-boundary detection against six ontologies. For each document it outputs seven feature sets: 
skills_req,
certifications,
edu_level_req,
edu_major_req,
experience_tags,
languages_req,
years_req_range

It is extremely fast (thousands of documents per second), offers 100% coverage within the ontology, is deterministic and reproducible, and every feature is traceable to its source match. Its limitations are that it cannot distinguish must-have from nice-to-have requirements, lacks semantic understanding, and is confined to predefined terms.

Ref: folder 1.4/ baseline_rules_extraction.py.

LLM-Based Extraction Engine
To address those gaps, we use qwen3:14b for structured, multilingual (EN/ZH) extraction with JSON-constrained outputs. A concise prompt (system role, task, JSON schema, few-shot examples, and JD input) guides the model to extract six components:
years_min and years_max (numerical range for required experience, e.g., "5+ years" → years_min: 5, years_max: null), 
must_have_skills_raw (skills explicitly marked as mandatory by phrases like "must have", "required", "essential"), 
nice_to_have_skills_raw (skills marked as preferred by phrases like "is a plus", "preferred", "an advantage"), 
must_have_certs_raw (mandatory certifications), 
nice_to_have_certs_raw (preferred certifications)
role_focus_raw (2-3 keywords summarizing the core function, e.g., ["Financial Reporting", "Consolidation"])

The LLM's key contributions include distinguishing importance levels (must-have vs. nice-to-have), contextual inference and understanding semantic context, flexibility in handling varied phrasing, and semantic summarization to identify core job functions beyond simple keyword matching.
  
Ref: folder 1.5/llm_extractor.py; folder 1.5/ run_llm_pipeline.py; 

Design Trade-offs for MVP
We apply the LLM primarily to positions (JDs) and keep candidates on rules plus computed years. JDs typically state hierarchical requirements explicitly, yielding high ROI for LLM extraction, while CVs are heterogeneous and rarely label priority. This focus captures the most impactful hierarchy signals while keeping cost and latency under control for the MVP.

Normalization & Merging
After extraction, LLM raw terms are fuzzy-matched to the ontology (e.g., “proficient in excel” → “MS Excel”) and merged with the Rules output. Each position has 12 content features—six normalized, hierarchical LLM features and six Rules features—saved as positions_FINAL_hybrid_features.parquet.

LLM (normalized)
years_req_range: [LLM.years_min, LLM.years_max]
skills_req_must_have: {...}
skills_req_nice_to_have: {...}
certs_req_must_have: {...}
certs_req_nice_to_have: {...}
role_focus_raw: {...} (kept from LLM as an inferred feature)
Rules Engine (fallback / full-coverage)
skills_req: {...} (use Rules output directly)
certifications_req: {...}
edu_level_req: {...}
edu_major_req: {...}
experience_tags_req: {...}
languages_req: {...}

Ref: folder 1.5/positions_FINAL_hybrid_features.parquet.

Phase 2: Evaluation Framework
Without explicit hire/shortlist labels, we use application records as proxy positives: if a candidate applied, it signals some degree of relevance. This signal is imperfect but pragmatic for MVPs and widely used in production HR systems.
To avoid leakage and reflect real use, we apply a chronological split by application date: 70% train, 15% validation, and 15% test (old → new). The model is trained on history and evaluated on future data.
We assess retrieval with Recall@K (ensuring most actual applicants appear in the Top-200) and optimize reranking with nDCG@10, a position-aware, normalized ranking metric suited to graded relevance.
For fine-tuning, we built a small human-labeled benchmark: 5 representative positions × 15 candidates = 75 pairs, each scored 0–3 for relevance. This set serves as the Optuna objective for weight tuning.

Phase 3: Core Matching Engine
Stage 1: Candidate Retrieval with Hybrid Fusion
This stage narrows the full candidate pool to a focused Top-200 per position, creating a high-recall set for reranking. A pure TF-IDF baseline was fast and precise on exact keywords, but missed semantic matches and struggled with mixed English/Chinese text. We therefore adopt a hybrid fusion that combines TF-IDF (precision on hard domain terms) with SBERT (semantic breadth and multilingual understanding), yielding materially better Recall@200 and a stronger starting slate for Stage 2.

TF-IDF Retrieval Component (Weight: 0.3)
We compute cosine similarity on TF-IDF vectors to capture explicit overlaps on domain-critical terminology (e.g., “Big 4”, “HKICPA”, “Consolidation”, “SAP”). This component emphasizes high-precision hits and helps ensure profiles containing hard constraints or regulated credentials are not missed. Assigning it a 0.3 weight preserves this precision signal without letting exact-match idiosyncrasies dominate overall ranking, especially where phrasing varies across JDs and CVs.

SBERT Semantic Retrieval Component (Weight: 0.7)
We use the bi-encoder paraphrase-multilingual-MiniLM-L12-v2 (768-dim) to produce multilingual embeddings (EN/中文). To increase informativeness, we build feature-augmented texts: for positions—title, JD, industry, must-/nice-to-have skills and certs, role focus, education level, languages; for candidates—skills, certs, education level and majors, experience tags, languages, and years of experience. SBERT captures conceptual similarity (e.g., “Financial Reporting” ~ “Statutory Reporting”/“Month-End Closing”; “會計” ~ “Accountant”), improving recall where exact wording differs. We weight SBERT at 0.7 to reflect its superior semantic coverage in bilingual content.

Fusion Strategy
For each position, we compute TF-IDF and SBERT scores independently, apply per-query min–max normalization to [0,1], and combine them linearly: Hybrid = 0.3 · TFIDF_norm + 0.7 · SBERT_norm. Candidates are ranked by the hybrid score and the Top-200 are selected. Embeddings are precomputed and indexed in ChromaDB (HNSW) for low-latency ANN search with persistent storage. This ensemble consistently outperforms either method alone on validation Recall@200. Key artifacts: retrieval_scores_hybrid.pkl (scores) and retrieval_results_hybrid.pkl (Top-K lists).

Ref: folder 3.1.1/candidate_retrieval_hybrid.py; 

Stage 2: Candidate Reranking
We rerank the Top-200 candidates with a 14-component weighted additive model to produce a precise Top-10. All component scores si are normalized to 0, 1 before weighting; missing scores default to 0; weights wi are learned with Optuna (objective: nDCG@10).

TotalScore = Σ wᵢ·sᵢ 
(w_must_skill * must_have_skill_match)
+ (w_nice_skill * nice_to_have_skill_match)
+ (w_must_cert * must_have_cert_match)
+ (w_nice_cert * nice_to_have_cert_match)
+ (w_years * years_match)
+ (w_focus * role_focus_match)
+ (w_all_skills * all_skills_match)			# fallback
+ (w_all_certs * all_certs_match)			# fallback
+ (w_exp_tags * exp_tags_match)
+ (w_edu_level * edu_level_match)
+ (w_edu_major * edu_major_match)
+ (w_lang * lang_match)
+ (w_text_sim * text_sim_score)			# Stage-1 hybrid score
+ (w_cross_enc * cross_encoder_score) 	

The first six signals are LLM-derived (e.g., Jaccard on must-have skills, graded years-within-range, and semantic role-focus overlap). The next six come from the Rules engine as comprehensive fallbacks (skills/certs coverage, experience tags like “Big 4”, binary education level/major, and languages). We also include the Stage-1 hybrid retrieval score (text_sim_score) to preserve coarse relevance.

The cross-encoder jointly encodes the position–candidate pair ([CLS] position [SEP] candidate [SEP]) with full cross-attention (e.g., BERT/RoBERTa), yielding a fine-grained relevance score that captures token-level interactions (e.g., “5 years in Big 4” ↔ “Senior Auditor, Big 4 experience”). Because it is slower than bi-encoders, we apply it only to the Top-200.

Ref: folder 3.2/Candidate Reranking + Cross Encoder.py; 

Stage 3: Parameter Tuning
3.1 Baseline – 7 features
The baseline uses a Grid Search over 7 rules-only features (skills, certifications, edu_level, edu_majors, experience_tags, languages, years_exp) and optimizes Recall@10 on the validation set. The grid enumerates 648 discrete weight combinations with no learning across trials. The best setting is saved to best_weights_baseline.json.

Ref: folder 3.3.1/parameter_tuning_baseline.py;

3.2 Optimized — Human-Labeled Ground Truth, 14 LLM Features, Optuna

We optimize the 14-component reranker with Optuna (TPE), learning continuous weights in 0.0, 5.0. For each trial, the Total Score is computed for every labeled position–candidate pair and evaluated with nDCG@10, averaged across positions. The sampler iteratively focuses on promising regions, typically reaching near-optimal weights within ~100–200 trials; we cap at 1,000 to allow further refinement under longer budgets. Final weights are saved as best_weights_14_features.json.
For clarity and reproducibility:
Benchmark format: 75 labeled pairs (e.g., 5 positions × 15 candidates), with graded relevance 0–3.
Score normalization: all component scores si are normalized to [0,1] before weighting; missing scores default to 0.
Metric definition: nDCG@10 uses 𝑘 = min (10, # labeled candidates for that position ) k=min(10, #labeled candidates for that position) so positions with fewer labels are handled correctly.
Inputs required:
rerank_results_FINAL.json (per-pair 14 scores),
ground_truth_labels.json (mapping position_id → [{can_id, relevance}]).
Reproducibility & efficiency: fix a random seed; enable an Optuna pruner (e.g., Median or ASHA) to stop weak trials early; set direction="maximize"; optionally persist the study to SQLite to resume interrupted runs.

Ref: folder 3.3.2/labeling_tool.py;

Ref: folder 3.3.2/parameter_tuning_optimized.py;
Stage 4: End-to-End Shortlist Generation (Demonstration) 
Using the learned weights in best_weights_14_features.json and the per-candidate scores in rerank_results_FINAL.json, the demo computes a TotalScore = Σ wᵢ·sᵢ for each candidate under a given position (e.g., 3473415), sorts by TotalScore, and outputs a Top-10 shortlist. The script (demo_rank_top10.py) also surfaces lightweight explainability by reporting each candidate’s top contributing features (e.g., years_match, all_skills_match, cross_encoder_score), alongside the overall score. Deliverables are a compact Top-10 table (CSV/JSON) and an optional full table with per-feature contributions—sufficient for recruiters to review the curated list and understand, at a glance, why each candidate ranks where they do.

Ref: folder 3.4/demo_rank_top10.py;

4. Analysis and Discussion
4.1 Feature Importance and Insights
Optuna-learned weights (with all component scores normalized to 0,1) indicate which signals drive match quality in A/A/T.

Reading The Weights
Breadth of skills and appropriate years of experience dominate (both 4.99), signalling recruiters prefer versatile candidates with the right tenure. Among already-qualified profiles (thanks to retrieval), nice-to-have skills become a strong differentiator (4.36). High weight on languages (3.65) reflects HK’s EN/中文 context. Major outranking education level suggests domain fit matters more than degree seniority. Cross-encoder and role focus add semantic nuance without overpowering explicit signals.
Caveats: Some features are correlated (e.g., all_skills vs must_have_skill). Since two top weights sit at the upper bound, widening the search range (e.g., [0, 6] or [0, 10]) is a useful robustness check. Practical importance also depends on activation frequency; consider reporting mean contribution E[wisi]on validation shortlists.

4.2 Model Performance Results
We compare the Optimized Model against a rules-only baseline.
The Optimized Model achieved an nDCG@10 score of 0.735, representing a more than 45-fold increase over the baseline's score of 0.016.

Business Impact
An nDCG@10 of 0.735 means recruiters receive a Top-10 that already concentrates the best matches, shifting effort from trawling 100–200 CVs to making informed interview decisions—directly reducing screening time and lifting hit rate.

4.3 What Drove the Lift
Hierarchical Features via LLM: LLM extraction separates must-have from nice-to-have skills/certs and surfaces role focus and years ranges, letting the model weight differentiators (e.g., nice-to-have skills) higher once basics are satisfied.
Hybrid Retrieval (TF-IDF + SBERT): TF-IDF preserves hard domain terms (“Big 4”, “HKICPA”, “Consolidation”); SBERT captures semantic and bilingual variants (“Financial Reporting” ↔ “Statutory Reporting”, “會計” ↔ “Accountant”). Per-query min–max normalization and a 0.3/0.7 blend improve Recall@200 (≈0.45 → ≈0.78 on validation) versus either method alone.
Cross-Encoder Signal: Pairwise encoding with full token cross-attention provides a precise relevance score (e.g., “5 years in Big 4” ↔ “Senior Auditor with Big 4”). It complements bi-encoder retrieval and explicit features (weight 1.64).
Bayesian Weight Tuning (Optuna): Instead of a discrete grid (648 combos), Optuna (TPE) explores continuous weights 0,5, learns from prior trials, and typically converges in ~100–200 trials, delivering the 0.735 nDCG@10.

4.4 Delivery, Explainability & Rationale
For each position, we retrieve a Top-200 via hybrid fusion, compute 14 component scores per candidate, apply the learned weights, and rank a Top-10 shortlist. The Explainability panel surfaces per-component contributions, matched/missing must-haves, and years-of-experience fit. Design-wise, a two-stage (retrieve → rerank) architecture balances latency and accuracy; a weighted additive model remains interpretable; Rules ensure coverage while LLM adds hierarchical nuance; the domain ontology grounds industry terms; and Optuna tunes robust weights under scarce labels.
