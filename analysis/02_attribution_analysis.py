import ast

import numpy as np
import pandas as pd
from scipy.stats import norm


def mention_summary(mentions, reference, code, name, denominator):
    counts = mentions.groupby(code)["master_id"].nunique().rename("n")
    result = reference[[code, name]].drop_duplicates().set_index(code).join(counts)
    result["n"] = result["n"].fillna(0).astype(int)
    result["denominator"] = denominator
    result["mention_percent"] = result["n"] / denominator * 100
    return result.reset_index()


def group_comparison(delayed_n, refusal_n, delayed_total, refusal_total):
    delayed = delayed_n / delayed_total
    refusal = refusal_n / refusal_total
    difference = refusal - delayed
    difference_se = np.sqrt(
        delayed * (1 - delayed) / delayed_total
        + refusal * (1 - refusal) / refusal_total
    )

    ratio = refusal / delayed
    log_ratio_se = np.sqrt(
        (1 - refusal) / (refusal_total * refusal)
        + (1 - delayed) / (delayed_total * delayed)
    )
    z = norm.ppf(0.975)

    return pd.Series({
        "delayed_mention_percent": delayed * 100,
        "refusal_mention_percent": refusal * 100,
        "difference_pp": difference * 100,
        "difference_ci_low": (difference - z * difference_se) * 100,
        "difference_ci_high": (difference + z * difference_se) * 100,
        "prevalence_ratio": ratio,
        "prevalence_ratio_ci_low": np.exp(np.log(ratio) - z * log_ratio_se),
        "prevalence_ratio_ci_high": np.exp(np.log(ratio) + z * log_ratio_se),
    })


def misclassification_analysis(comparison, performance, simulations=10000):
    rng = np.random.default_rng(2026)
    results = []

    for row in comparison.itertuples():
        error = performance.loc[performance["second_level"] == row.second_level].iloc[0]
        sensitivity = rng.beta(error.tp + 0.5, error.fn + 0.5, simulations)
        specificity = rng.beta(error.tn + 0.5, error.fp + 0.5, simulations)
        denominator = sensitivity + specificity - 1
        valid = denominator > 0

        delayed = row.delayed_n / row.delayed_total
        refusal = row.refusal_n / row.refusal_total
        delayed_raw = (delayed + specificity[valid] - 1) / denominator[valid]
        refusal_raw = (refusal + specificity[valid] - 1) / denominator[valid]
        truncated = ((delayed_raw < 0) | (delayed_raw > 1)
                     | (refusal_raw < 0) | (refusal_raw > 1))
        adjusted_delayed = np.clip(delayed_raw, 0, 1)
        adjusted_refusal = np.clip(refusal_raw, 0, 1)
        adjusted_difference = adjusted_refusal - adjusted_delayed
        adjusted_ratio = adjusted_refusal / adjusted_delayed

        results.append({
            "second_level": row.second_level,
            "adjusted_delayed_median_percent": np.median(adjusted_delayed) * 100,
            "adjusted_delayed_interval_low": np.quantile(adjusted_delayed, 0.025) * 100,
            "adjusted_delayed_interval_high": np.quantile(adjusted_delayed, 0.975) * 100,
            "adjusted_refusal_median_percent": np.median(adjusted_refusal) * 100,
            "adjusted_refusal_interval_low": np.quantile(adjusted_refusal, 0.025) * 100,
            "adjusted_refusal_interval_high": np.quantile(adjusted_refusal, 0.975) * 100,
            "adjusted_difference_median_pp": np.median(adjusted_difference) * 100,
            "adjusted_difference_interval_low": np.quantile(adjusted_difference, 0.025) * 100,
            "adjusted_difference_interval_high": np.quantile(adjusted_difference, 0.975) * 100,
            "adjusted_ratio_median": np.nanmedian(adjusted_ratio),
            "adjusted_ratio_interval_low": np.nanquantile(adjusted_ratio, 0.025),
            "adjusted_ratio_interval_high": np.nanquantile(adjusted_ratio, 0.975),
            "direction_retained_percent": np.mean(
                adjusted_difference * (refusal - delayed) > 0
            ) * 100,
            "invalid_simulation_percent": np.mean(~valid) * 100,
            "truncated_simulation_percent": np.mean(truncated) * 100,
        })

    return pd.DataFrame(results)


data = pd.read_csv("attribution_labels.csv")
codebook = pd.read_csv("reason_codebook.csv")
performance = pd.read_csv("attribution_performance.csv")
anomaly_table = pd.read_csv("positive_anomalous_months.csv")

data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["year_month"] = data["date"].dt.to_period("M")
data["specific_reasons"] = data["reason_labels"].apply(ast.literal_eval)

mentions = (
    data[["master_id", "year", "month", "year_month", "stance_label",
          "specific_reasons"]]
    .explode("specific_reasons")
    .rename(columns={"specific_reasons": "specific_reason"})
    .merge(codebook, on="specific_reason", how="left")
)

specific_mentions = mentions.drop_duplicates(["master_id", "specific_reason"])
second_mentions = mentions.drop_duplicates(["master_id", "second_level"])
first_mentions = mentions.drop_duplicates(["master_id", "first_level"])

overall_attribution = {
    "first_level": mention_summary(
        first_mentions, codebook, "first_level", "first_level_name", len(data)
    ),
    "second_level": mention_summary(
        second_mentions, codebook, "second_level", "second_level_name", len(data)
    ),
    "specific_reason": mention_summary(
        specific_mentions, codebook, "specific_reason", "specific_reason_name",
        len(data)
    ),
}

delayed_total = (data["stance_label"] == 3).sum()
refusal_total = (data["stance_label"] == 4).sum()
comparison = (
    second_mentions.groupby(["second_level", "stance_label"])["master_id"]
    .nunique().unstack(fill_value=0).rename(columns={3: "delayed_n", 4: "refusal_n"})
    .reset_index()
)
comparison = comparison[comparison["second_level"] != "D0"].copy()
comparison["delayed_total"] = delayed_total
comparison["refusal_total"] = refusal_total
comparison = pd.concat([
    comparison,
    comparison.apply(
        lambda x: group_comparison(x["delayed_n"], x["refusal_n"],
                                   delayed_total, refusal_total), axis=1
    ),
], axis=1)

misclassification_results = misclassification_analysis(comparison, performance)

annual = (
    second_mentions[second_mentions["second_level"] != "D0"]
    .groupby(["year", "second_level"])["master_id"].nunique().rename("n")
    .reset_index()
)
annual = annual.merge(data.groupby("year").size().rename("denominator"), on="year")
annual["mention_percent"] = annual["n"] / annual["denominator"] * 100
annual["rank"] = annual.groupby("year")["n"].rank(
    method="first", ascending=False
)
annual_top_three = annual[annual["rank"] <= 3]

anomaly_table["month"] = pd.to_datetime(anomaly_table["month"]).dt.to_period("M")
focal_months = anomaly_table.loc[
    (anomaly_table["outcome"] == "hesitancy")
    & (anomaly_table["detection_pattern"] == "both_methods"), "month"
]

deviations = []
categories = comparison["second_level"]
for focal_month in focal_months:
    focal_posts = data[data["year_month"] == focal_month]
    baseline_posts = data[
        (data["month"] == focal_month.month) & (data["year"] != focal_month.year)
    ]
    focal = second_mentions[second_mentions["year_month"] == focal_month]
    baseline = second_mentions[
        (second_mentions["month"] == focal_month.month)
        & (second_mentions["year"] != focal_month.year)
    ]

    for category in categories:
        focal_n = focal.loc[focal["second_level"] == category, "master_id"].nunique()
        baseline_n = baseline.loc[
            baseline["second_level"] == category, "master_id"
        ].nunique()
        deviations.append({
            "anomalous_month": focal_month,
            "second_level": category,
            "focal_n": focal_n,
            "focal_denominator": len(focal_posts),
            "baseline_n": baseline_n,
            "baseline_denominator": len(baseline_posts),
            "deviation_pp": (
                focal_n / len(focal_posts) - baseline_n / len(baseline_posts)
            ) * 100,
        })

anomalous_month_deviations = pd.DataFrame(deviations)
