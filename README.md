# LLM-enabled analysis of influenza vaccine hesitancy

This repository contains the publicly shareable code, prompts, and supporting
data for the study *LLM-Enabled Digital Analysis of Influenza Vaccine
Hesitancy: Decoupling Delay and Refusal via Semantic Attribution*.

## Analysis

- `analysis/01_stance_analysis.py`: stance distributions, temporal trends, STL
  decomposition, and anomalous-month detection.
- `analysis/02_attribution_analysis.py`: attribution distributions,
  delayed-vaccination versus refusal comparisons, temporal patterns, and
  misclassification sensitivity analysis.
- `analysis/03_sensitivity_analyses.py`: event exclusions, first-person,
  platform-composition, author-level, and external-comparison analyses.
- `analysis/04_model_evaluation.py`: screening, stance, attribution, and human
  annotation agreement metrics.
- `analysis/reason_codebook.csv`: three-level attribution framework, reason
  codes, category names, and operational definitions.

## Data

- `data/monthly_stance_summary.csv`: monthly counts and percentages for the
  four vaccination-stance categories from January 2016 to December 2025.
- `data/attribution_analysis_human_annotated_dataset.xlsx`: development and
  test splits, stance categories, two independent human annotations, and
  adjudicated attribution labels for 1,000 records.

## Preprocessing

- `preprocessing/01_pre_dedup.py`: basic time and text preparation.
- `preprocessing/02_dedup.py`: URL and exact title-body deduplication.
- `preprocessing/03_structural_filtering.py`: structural text filtering.
- `preprocessing/04_rule_based_relevance.py`: rule-based relevance screening.
- `preprocessing/05_rule_based_public_post_screening.py`: rule-based
  individual user-generated-content screening.

## Prompts

- `prompts/01_relevance_screening.md`: relevance-screening instruction.
- `prompts/02_public_post_screening.md`: author/source-screening instruction.
- `prompts/03_stance_classification.md`: four-category stance-classification
  instruction.
- `prompts/04_open_ended_reason_extraction.md`: open-ended reason-extraction
  instruction.
- `prompts/05_attribution_coding/v1.md`: development attribution-coding
  instruction.
- `prompts/05_attribution_coding/v2.md`: final attribution-coding instruction.
- `prompts/06_first_person_decision_filter.md`: first-person vaccination
  decision-filtering instruction.
