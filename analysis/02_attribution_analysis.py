"""Post-label statistical analysis of vaccine-hesitancy attributions.

The input contains one to three final reason codes for each vaccine-hesitant
post. Edit the file locations below before running the script.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# File locations
INPUT_FILE = Path("data/attribution_labels.csv")
CODEBOOK_FILE = Path("reason_codebook.csv")
ANOMALOUS_MONTH_FILE = Path("outputs/stance/positive_anomalous_months.csv")
OUTPUT_DIR = Path("outputs/attribution")


def summarize_mentions(frame, reference, code_column, name_column, level, total):
    """Number and rate of posts mentioning each reason."""
    counts = (
        frame.groupby(code_column)["master_id"]
        .nunique()
        .rename("n")
        .reset_index()
    )
    summary = reference[[code_column, name_column]].drop_duplicates().merge(
        counts, on=code_column, how="left"
    )
    summary["n"] = summary["n"].fillna(0).astype(int)
    summary["denominator"] = total
    summary["mention_rate_percent"] = summary["n"] / total * 100
    summary.insert(0, "level", level)
    return summary.rename(columns={code_column: "code", name_column: "reason"})


def fdr_bh(p_values):
    """Benjamini-Hochberg false-discovery-rate correction."""
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1)
    return adjusted


# 1. Read the final attribution labels and reason hierarchy
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["year_month"] = df["date"].dt.to_period("M")
df["specific_reasons"] = df["reason_labels"].apply(ast.literal_eval)

codebook = pd.read_csv(CODEBOOK_FILE)
second_reference = codebook[
    ["second_level", "second_level_name"]
].drop_duplicates()
first_reference = codebook[
    ["first_level", "first_level_name"]
].drop_duplicates()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Create one row for each reason mentioned in each post
mentions = (
    df[
        [
            "master_id",
            "date",
            "year",
            "month",
            "year_month",
            "stance_label",
            "specific_reasons",
        ]
    ]
    .explode("specific_reasons")
    .rename(columns={"specific_reasons": "specific_reason"})
)
mentions = mentions.merge(codebook, on="specific_reason", how="left")

# A post is counted once at each level even if several of its specific reasons
# belong to the same second-level category or first-level domain.
specific_mentions = mentions.drop_duplicates(["master_id", "specific_reason"])
second_mentions = mentions.drop_duplicates(["master_id", "second_level"])
first_mentions = mentions.drop_duplicates(["master_id", "first_level"])


# 3. Overall attribution distribution at three hierarchical levels
overall_attribution = pd.concat(
    [
        summarize_mentions(
            first_mentions,
            first_reference,
            "first_level",
            "first_level_name",
            "first_level",
            len(df),
        ),
        summarize_mentions(
            second_mentions,
            second_reference,
            "second_level",
            "second_level_name",
            "second_level",
            len(df),
        ),
        summarize_mentions(
            specific_mentions,
            codebook,
            "specific_reason",
            "specific_reason_name",
            "specific_reason",
            len(df),
        ),
    ],
    ignore_index=True,
)

reasons_per_post = (
    df["specific_reasons"]
    .str.len()
    .value_counts()
    .sort_index()
    .rename_axis("number_of_specific_reasons")
    .reset_index(name="n")
)
reasons_per_post["percent"] = reasons_per_post["n"] / len(df) * 100


# 4. Delayed vaccination versus vaccine refusal
comparison_reference = second_reference[
    second_reference["second_level"] != "D0"
].copy()
comparison_counts = (
    second_mentions.groupby(["second_level", "stance_label"])["master_id"]
    .nunique()
    .unstack(fill_value=0)
    .reset_index()
    .rename(columns={3: "delayed_n", 4: "refusal_n"})
)
comparison = comparison_reference.merge(
    comparison_counts, on="second_level", how="left"
)

delayed_total = (df["stance_label"] == 3).sum()
refusal_total = (df["stance_label"] == 4).sum()
comparison["delayed_denominator"] = delayed_total
comparison["refusal_denominator"] = refusal_total
comparison["delayed_mention_rate_percent"] = (
    comparison["delayed_n"] / delayed_total * 100
)
comparison["refusal_mention_rate_percent"] = (
    comparison["refusal_n"] / refusal_total * 100
)
comparison["difference_pp_refusal_minus_delayed"] = (
    comparison["refusal_mention_rate_percent"]
    - comparison["delayed_mention_rate_percent"]
)
comparison["prevalence_ratio_refusal_vs_delayed"] = (
    comparison["refusal_mention_rate_percent"]
    / comparison["delayed_mention_rate_percent"]
)

p_values = []
for row in comparison.itertuples():
    table = [
        [row.delayed_n, delayed_total - row.delayed_n],
        [row.refusal_n, refusal_total - row.refusal_n],
    ]
    # Pearson chi-square test without Yates continuity correction.
    p_values.append(chi2_contingency(table, correction=False).pvalue)

comparison["chi_square_p_value"] = p_values
comparison["fdr_adjusted_p_value"] = fdr_bh(p_values)
comparison = comparison.sort_values(
    "difference_pp_refusal_minus_delayed", ascending=False
)


# 5. Annual second-level attribution structure and annual top three
categories = comparison_reference["second_level"].tolist()
years = sorted(df["year"].unique())
annual_index = pd.MultiIndex.from_product(
    [years, categories], names=["year", "second_level"]
)
annual_counts = (
    second_mentions[second_mentions["second_level"] != "D0"]
    .groupby(["year", "second_level"])["master_id"]
    .nunique()
    .reindex(annual_index, fill_value=0)
    .rename("n")
    .reset_index()
)
annual_denominators = df.groupby("year").size().rename("vaccine_hesitant_posts")
annual = annual_counts.merge(second_reference, on="second_level", how="left")
annual = annual.merge(annual_denominators, on="year", how="left")
annual["mention_rate_percent"] = (
    annual["n"] / annual["vaccine_hesitant_posts"] * 100
)
annual["annual_rank"] = (
    annual.groupby("year")["n"].rank(method="first", ascending=False).astype(int)
)
annual_top3 = annual[annual["annual_rank"] <= 3].sort_values(
    ["year", "annual_rank"]
)


# 6. Deviations during hesitancy-rate anomalies identified by both methods
anomaly_results = pd.read_csv(ANOMALOUS_MONTH_FILE)
anomaly_results["date"] = pd.to_datetime(anomaly_results["date"])
anomalous_months = (
    anomaly_results.loc[
        (anomaly_results["outcome"] == "vaccine_hesitancy_rate")
        & (anomaly_results["detection_pattern"] == "both_methods"),
        "date",
    ]
    .dt.to_period("M")
    .tolist()
)

deviation_rows = []
for anomalous_month in anomalous_months:
    focal_posts = df[df["year_month"] == anomalous_month]
    # Pool all posts from the same calendar month in the other years.
    baseline_posts = df[
        (df["month"] == anomalous_month.month)
        & (df["year"] != anomalous_month.year)
    ]
    focal_mentions = second_mentions[
        second_mentions["year_month"] == anomalous_month
    ]
    baseline_mentions = second_mentions[
        (second_mentions["month"] == anomalous_month.month)
        & (second_mentions["year"] != anomalous_month.year)
    ]

    focal_counts = focal_mentions["second_level"].value_counts()
    baseline_counts = baseline_mentions["second_level"].value_counts()

    for category in categories:
        focal_n = focal_counts.get(category, 0)
        baseline_n = baseline_counts.get(category, 0)
        focal_rate = focal_n / len(focal_posts) * 100
        baseline_rate = baseline_n / len(baseline_posts) * 100
        deviation_rows.append(
            {
                "anomalous_month": str(anomalous_month),
                "second_level": category,
                "focal_n": focal_n,
                "focal_denominator": len(focal_posts),
                "focal_mention_rate_percent": focal_rate,
                "baseline_n": baseline_n,
                "baseline_denominator": len(baseline_posts),
                "same_month_baseline_rate_percent": baseline_rate,
                "deviation_pp": focal_rate - baseline_rate,
            }
        )

anomalous_month_deviations = pd.DataFrame(deviation_rows).merge(
    second_reference, on="second_level", how="left"
)


# 7. Save aggregate source tables
overall_attribution.to_csv(
    OUTPUT_DIR / "overall_attribution_distribution.csv", index=False
)
reasons_per_post.to_csv(OUTPUT_DIR / "number_of_reasons_per_post.csv", index=False)
comparison.to_csv(OUTPUT_DIR / "delayed_vs_refusal.csv", index=False)
annual.to_csv(OUTPUT_DIR / "annual_second_level_structure.csv", index=False)
annual_top3.to_csv(OUTPUT_DIR / "annual_top3.csv", index=False)
anomalous_month_deviations.to_csv(
    OUTPUT_DIR / "anomalous_month_deviations.csv", index=False
)
