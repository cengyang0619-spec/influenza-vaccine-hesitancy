import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL


def component_strength(component, remainder):
    return max(0, 1 - np.var(remainder, ddof=1) /
               np.var(component + remainder, ddof=1))


def seasonal_trend(series):
    s = 0
    variance = 0
    pairs = 0
    slopes = []

    for month in range(1, 13):
        x = series[series.index.month == month]
        values = x.to_numpy()
        years = x.index.year.to_numpy()
        n = len(values)
        pairs += n * (n - 1) // 2

        for i in range(n - 1):
            differences = values[i + 1:] - values[i]
            s += np.sign(differences).sum()
            slopes.extend(differences / (years[i + 1:] - years[i]))

        _, counts = np.unique(values, return_counts=True)
        ties = sum(c * (c - 1) * (2 * c + 5) for c in counts)
        variance += (n * (n - 1) * (2 * n + 5) - ties) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(variance)
    elif s < 0:
        z = (s + 1) / np.sqrt(variance)
    else:
        z = 0

    slopes = np.sort(slopes)
    rank_offset = norm.ppf(0.975) * np.sqrt(variance)
    lower = int(np.floor((len(slopes) - rank_offset) / 2))
    upper = int(np.ceil((len(slopes) + rank_offset) / 2))

    return {
        "tau": s / pairs,
        "p_value": 2 * norm.sf(abs(z)),
        "slope": np.median(slopes),
        "slope_ci_low": slopes[lower],
        "slope_ci_high": slopes[upper],
    }


def anomaly_scores(remainder):
    median = remainder.median()
    mad = (remainder - median).abs().median()
    robust_z = 0.67448975 * (remainder - median) / mad

    rolling_median = remainder.rolling(13, center=True, min_periods=1).median()
    rolling_mad = remainder.rolling(13, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    hampel = 0.67448975 * (remainder - rolling_median) / rolling_mad

    pattern = np.select(
        [(robust_z > 3) & (hampel > 3), robust_z > 3, hampel > 3],
        ["both_methods", "robust_z_only", "hampel_only"],
        default="",
    )
    return robust_z, hampel, pattern


data = pd.read_csv("stance_labels.csv")
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["hesitant"] = data["stance_label"].isin([3, 4])

stance_distribution = (
    data["stance_label"].value_counts().reindex([1, 2, 3, 4])
    .rename("n").to_frame()
)
stance_distribution["percent"] = stance_distribution["n"] / len(data) * 100

annual_stance = pd.crosstab(data["year"], data["stance_label"]).reindex(
    columns=[1, 2, 3, 4], fill_value=0
)
annual_stance["total"] = annual_stance.sum(axis=1)
annual_stance["hesitant"] = annual_stance[3] + annual_stance[4]
annual_stance["hesitancy_percent"] = (
    annual_stance["hesitant"] / annual_stance["total"] * 100
)

monthly = (
    data.set_index("date").resample("MS")
    .agg(total=("master_id", "size"), hesitant=("hesitant", "sum"))
)
monthly["hesitancy_proportion"] = monthly["hesitant"] / monthly["total"]
monthly["hesitancy_percent"] = monthly["hesitancy_proportion"] * 100

volume_model = STL(
    np.log(monthly["total"]), period=12, seasonal=13, trend=25, robust=True
).fit()

proportion = monthly["hesitancy_proportion"]
hesitancy_model = STL(
    np.log(proportion / (1 - proportion)),
    period=12, seasonal=13, trend=25, robust=True
).fit()

monthly["volume_trend"] = volume_model.trend
monthly["volume_seasonal"] = volume_model.seasonal
monthly["volume_remainder"] = volume_model.resid
monthly["hesitancy_trend"] = hesitancy_model.trend
monthly["hesitancy_seasonal"] = hesitancy_model.seasonal
monthly["hesitancy_remainder"] = hesitancy_model.resid

volume_trend = seasonal_trend(monthly["total"])
hesitancy_trend = seasonal_trend(monthly["hesitancy_percent"])

time_series_statistics = pd.DataFrame([
    {
        "series": "monthly_post_volume",
        "trend_strength": component_strength(volume_model.trend,
                                               volume_model.resid),
        "seasonal_strength": component_strength(volume_model.seasonal,
                                                  volume_model.resid),
        **volume_trend,
    },
    {
        "series": "monthly_hesitancy_percent",
        "trend_strength": component_strength(hesitancy_model.trend,
                                               hesitancy_model.resid),
        "seasonal_strength": component_strength(hesitancy_model.seasonal,
                                                  hesitancy_model.resid),
        **hesitancy_trend,
    },
])

anomalies = []
for outcome in ["volume", "hesitancy"]:
    robust_z, hampel, pattern = anomaly_scores(monthly[f"{outcome}_remainder"])
    result = pd.DataFrame({
        "outcome": outcome,
        "month": monthly.index,
        "robust_z": robust_z,
        "hampel": hampel,
        "detection_pattern": pattern,
    })
    anomalies.append(result[result["detection_pattern"] != ""])

anomalous_months = pd.concat(anomalies, ignore_index=True)
