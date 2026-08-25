# Influenza Vaccine Hesitancy on Chinese Social Media

This repository contains code, prompts, model configurations, and analysis scripts associated with a longitudinal study of influenza vaccine hesitancy expressed on Chinese social media from 2016 to 2025.

## Repository contents

- `preprocessing/`: data cleaning and screening code used before model-based analysis.
- `prompts/`: full prompts used for relevance screening, public-post screening, stance classification, reason extraction/coding, and first-person sensitivity analysis.
- `model_configuration/`: model identifiers, inference settings, decoding parameters, and failure-handling rules.
- `analysis/`: statistical analysis and sensitivity-analysis scripts used to generate reported results.
- `docs/`: workflow documentation, data dictionaries, and reproducibility notes.

## Data availability and privacy

Raw social-media records are not included in this repository because they may contain user-generated text, account identifiers, URLs, and other platform-derived information subject to privacy and data-access restrictions. No authentication credentials, cookies, tokens, proprietary provider endpoints, or other access secrets are included.

Where feasible, aggregated outputs and non-identifying example inputs may be provided to support reproducibility of the analysis workflow.

## Reproducibility scope

The repository is intended to document the computational workflow from cleaned inputs through LLM-based measurement and statistical analysis. It does not provide unrestricted redistribution of the original social-media corpus.

## Citation

Citation information will be added after the manuscript and repository version are finalized.
