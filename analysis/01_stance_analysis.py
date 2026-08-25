from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL


# File locations
INPUT_FILE = Path("data/stance_labels.csv")
OUTPUT_DIR = Path("outputs/stance")

# Parameters reported in the statistical methods
STL_PERIOD = 12
STL_SEASONAL = 13
STL_TREND = 25
HAMPEL_WINDOW = 13
ANOMALY_THRESHOLD = 3

# Stance coding used in the shared analysis file
STANCE_NAMES = {
    1: "pro_vaccination",
    2: "no_explicit_stance",
    3: "delayed_vaccination",
    4: "vaccine_refusal",
}


def component_strength(component, remainder):
    """Strength of an STL component."""
    return max(
        0,
        1 - np.var(remainder, ddof=1) / np.var(component + remainder, ddof=1),
    )


def seasonal_mann_kendall(values):
    """Seasonal Mann-Kendall test with calendar month as the season."""
    s_total = 0
    variance_total = 0
    comparable_pairs = 0

    for month in range(1, 13):
        x = values[values.index.month == month].to_numpy()
        n = len(x)

        for i in range(n - 1):
            differences = x[i + 1 :] - x[i]
            s_total += np.sign(differences).sum()
            comparable_pairs += len(differences)

        _, tie_counts = np.unique(x, return_counts=True)
        tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
        variance_total += (n * (n - 1) * (2 * n + 5) - tie_term) / 18

    if s_total > 0:
        z_value = (s_total - 1) / np.sqrt(variance_total)
    elif s_total < 0:
        z_value = (s_total + 1) / np.sqrt(variance_total)
    else:
        z_value = 0

    return {
        "tau": s_total / comparable_pairs,
        "z": z_value,
        "p_value": 2 * norm.sf(abs(z_value)),
    }


def seasonal_theil_sen(values):
    """Median of pairwise slopes calculated within calendar months."""
    slopes = []

    for month in range(1, 13):
        x = values[values.index.month == month]
        years = x.index.year.to_numpy()
        observations = x.to_numpy()

        for i in range(len(x) - 1):
            slopes.extend(
                (observations[i + 1 :] - observations[i])
                / (years[i + 1 :] - years[i])
            )

    return float(np.median(slopes))


def anomaly_scores(remainder):
    """Robust z-score and local Hampel score for an STL remainder series."""
    series_median = remainder.median()
    series_mad = (remainder - series_median).abs().median()
    robust_z = 0.6745 * (remainder - series_median) / series_mad

    rolling_median = remainder.rolling(
        window=HAMPEL_WINDOW, center=True, min_periods=1
    ).median()
    rolling_mad = remainder.rolling(
        window=HAMPEL_WINDOW, center=True, min_periods=1
    ).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    hampel = (remainder - rolling_median).abs() / (1.4826 * rolling_mad)

    positive = (remainder > 0) & (
        (robust_z > ANOMALY_THRESHOLD) | (hampel > ANOMALY_THRESHOLD)
    )
    pattern = np.select(
        [
            (robust_z > ANOMALY_THRESHOLD) & (hampel > ANOMALY_THRESHOLD),
            robust_z > ANOMALY_THRESHOLD,
            hampel > ANOMALY_THRESHOLD,
        ],
        ["both_methods", "robust_z_only", "hampel_only"],
        default="",
    )

    return robust_z, hampel, positive, pattern


# 1. Read the final stance labels
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["hesitant"] = df["stance_label"].isin([3, 4])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Overall stance distribution and vaccine hesitancy
stance_summary = (
    df["stance_label"]
    .value_counts()
    .reindex(STANCE_NAMES)
    .rename_axis("stance_label")
    .reset_index(name="n")
)
stance_summary["stance"] = stance_summary["stance_label"].map(STANCE_NAMES)
stance_summary["percent"] = stance_summary["n"] / len(df) * 100
stance_summary = stance_summary[["stance_label", "stance", "n", "percent"]]

hesitancy_summary = pd.DataFrame(
    {
        "measure": ["vaccine_hesitancy"],
        "n": [df["hesitant"].sum()],
        "percent": [df["hesitant"].mean() * 100],
    }
)

stance_summary.to_csv(OUTPUT_DIR / "overall_stance_distribution.csv", index=False)
hesitancy_summary.to_csv(OUTPUT_DIR / "overall_hesitancy.csv", index=False)


# 3. Annual stance distribution
annual_counts = pd.crosstab(df["year"], df["stance_label"]).reindex(
    columns=STANCE_NAMES
)
annual_counts.columns = [STANCE_NAMES[x] for x in annual_counts.columns]
annual_counts["total_posts"] = annual_counts.sum(axis=1)
annual_counts["hesitant_posts"] = (
    annual_counts["delayed_vaccination"] + annual_counts["vaccine_refusal"]
)
annual_counts["hesitancy_rate_percent"] = (
    annual_counts["hesitant_posts"] / annual_counts["total_posts"] * 100
)

annual_percentages = annual_counts[list(STANCE_NAMES.values())].div(
    annual_counts["total_posts"], axis=0
) * 100
annual_percentages.columns = [f"{x}_percent" for x in annual_percentages.columns]

annual_summary = annual_counts.join(annual_percentages).reset_index()
annual_summary.to_csv(OUTPUT_DIR / "annual_stance_distribution.csv", index=False)


# 4. Monthly posting volume and vaccine hesitancy
monthly = (
    df.set_index("date")
    .resample("MS")
    .agg(total_posts=("master_id", "size"), hesitant_posts=("hesitant", "sum"))
)
monthly["hesitancy_rate"] = monthly["hesitant_posts"] / monthly["total_posts"]
monthly["hesitancy_rate_percent"] = monthly["hesitancy_rate"] * 100


# 5. STL decomposition of transformed monthly series
monthly["log_posting_volume"] = np.log(monthly["total_posts"])
monthly["logit_hesitancy_rate"] = np.log(
    monthly["hesitancy_rate"] / (1 - monthly["hesitancy_rate"])
)

volume_stl = STL(
    monthly["log_posting_volume"],
    period=STL_PERIOD,
    seasonal=STL_SEASONAL,
    trend=STL_TREND,
    robust=True,
).fit()

hesitancy_stl = STL(
    monthly["logit_hesitancy_rate"],
    period=STL_PERIOD,
    seasonal=STL_SEASONAL,
    trend=STL_TREND,
    robust=True,
).fit()

monthly["volume_stl_trend"] = volume_stl.trend
monthly["volume_stl_seasonal"] = volume_stl.seasonal
monthly["volume_stl_remainder"] = volume_stl.resid
monthly["hesitancy_stl_trend"] = hesitancy_stl.trend
monthly["hesitancy_stl_seasonal"] = hesitancy_stl.seasonal
monthly["hesitancy_stl_remainder"] = hesitancy_stl.resid


# 6. Trend and seasonality statistics
volume_mk = seasonal_mann_kendall(monthly["total_posts"])
hesitancy_mk = seasonal_mann_kendall(monthly["hesitancy_rate_percent"])

time_series_statistics = pd.DataFrame(
    [
        {
            "series": "monthly_posting_volume",
            "trend_strength": component_strength(volume_stl.trend, volume_stl.resid),
            "seasonal_strength": component_strength(
                volume_stl.seasonal, volume_stl.resid
            ),
            "seasonal_mk_tau": volume_mk["tau"],
            "seasonal_mk_z": volume_mk["z"],
            "seasonal_mk_p_value": volume_mk["p_value"],
            "seasonal_theil_sen_slope_per_year": seasonal_theil_sen(
                monthly["total_posts"]
            ),
        },
        {
            "series": "monthly_hesitancy_rate_percent",
            "trend_strength": component_strength(
                hesitancy_stl.trend, hesitancy_stl.resid
            ),
            "seasonal_strength": component_strength(
                hesitancy_stl.seasonal, hesitancy_stl.resid
            ),
            "seasonal_mk_tau": hesitancy_mk["tau"],
            "seasonal_mk_z": hesitancy_mk["z"],
            "seasonal_mk_p_value": hesitancy_mk["p_value"],
            "seasonal_theil_sen_slope_per_year": seasonal_theil_sen(
                monthly["hesitancy_rate_percent"]
            ),
        },
    ]
)
time_series_statistics.to_csv(
    OUTPUT_DIR / "time_series_statistics.csv", index=False
)


# 7. Positive anomalous months in both STL remainder series
for prefix in ["volume", "hesitancy"]:
    z_score, hampel_score, positive_anomaly, detection_pattern = anomaly_scores(
        monthly[f"{prefix}_stl_remainder"]
    )
    monthly[f"{prefix}_robust_z_score"] = z_score
    monthly[f"{prefix}_hampel_score"] = hampel_score
    monthly[f"{prefix}_positive_anomaly"] = positive_anomaly
    monthly[f"{prefix}_detection_pattern"] = detection_pattern

anomaly_tables = []
for prefix, outcome in [
    ("volume", "public_post_volume"),
    ("hesitancy", "vaccine_hesitancy_rate"),
]:
    table = monthly.loc[monthly[f"{prefix}_positive_anomaly"]].reset_index()
    table = table[
        [
            "date",
            f"{prefix}_robust_z_score",
            f"{prefix}_hampel_score",
            f"{prefix}_detection_pattern",
        ]
    ].rename(
        columns={
            f"{prefix}_robust_z_score": "robust_z_score",
            f"{prefix}_hampel_score": "hampel_score",
            f"{prefix}_detection_pattern": "detection_pattern",
        }
    )
    table["outcome"] = outcome
    anomaly_tables.append(table)

anomalous_months = pd.concat(anomaly_tables, ignore_index=True)
anomalous_months = anomalous_months[
    ["outcome", "date", "robust_z_score", "hampel_score", "detection_pattern"]
]

monthly.reset_index().to_csv(OUTPUT_DIR / "monthly_stl_results.csv", index=False)
anomalous_months.to_csv(OUTPUT_DIR / "positive_anomalous_months.csv", index=False)
