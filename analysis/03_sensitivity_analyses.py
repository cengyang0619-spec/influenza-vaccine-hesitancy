import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.seasonal import STL


def annual_summary(frame):
    result = frame.groupby("year").agg(total=("master_id", "size"),
                                          hesitant=("hesitant", "sum"))
    result["hesitancy_proportion"] = result["hesitant"] / result["total"]
    return result


def direct_standardization(frame, strata, years):
    reference = frame.groupby(strata).size().rename("reference_n").reset_index()
    reference["weight"] = reference["reference_n"] / reference["reference_n"].sum()

    cells = (
        frame.groupby(["year"] + strata)
        .agg(n=("master_id", "size"), hesitant=("hesitant", "sum"))
        .reset_index()
    )
    cells["cell_proportion"] = cells["hesitant"] / cells["n"]
    cells = cells.merge(reference[strata + ["weight"]], on=strata)
    return (
        cells.assign(weighted=lambda x: x["weight"] * x["cell_proportion"])
        .groupby("year")["weighted"].sum()
        .reindex(years)
        .rename("standardized_hesitancy_proportion")
    )


def author_weighted_series(frame, period, top_accounts):
    rows = []
    for value, subset in frame.groupby(period):
        identified = subset.dropna(subset=["author_account"])
        without_top = subset[~subset["author_account"].isin(top_accounts)]
        account_rates = identified.groupby("author_account")["hesitant"].mean()
        missing = subset[subset["author_account"].isna()]["hesitant"]
        missing_as_singletons = pd.concat([account_rates, missing], ignore_index=True)

        rows.append({
            period: value,
            "crude_proportion": subset["hesitant"].mean(),
            "top_1_percent_excluded_proportion": without_top["hesitant"].mean(),
            "account_equal_proportion": account_rates.mean(),
            "account_equal_missing_as_singletons_proportion":
                missing_as_singletons.mean(),
        })
    return pd.DataFrame(rows)


def moving_block_spearman(x, y, block_length=8, simulations=2000):
    values = np.column_stack([x, y])
    n = len(values)
    starts = np.arange(n - block_length + 1)
    rng = np.random.default_rng(2026)
    bootstrap = []

    for _ in range(simulations):
        indices = []
        while len(indices) < n:
            start = rng.choice(starts)
            indices.extend(range(start, start + block_length))
        sample = values[indices[:n]]
        bootstrap.append(spearmanr(sample[:, 0], sample[:, 1]).statistic)

    return {
        "n_weeks": n,
        "spearman_rho": spearmanr(values[:, 0], values[:, 1]).statistic,
        "interval_low": np.quantile(bootstrap, 0.025),
        "interval_high": np.quantile(bootstrap, 0.975),
    }


data = pd.read_csv("stance_labels_with_metadata.csv")
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.to_period("M")
data["hesitant"] = data["stance_label"].isin([3, 4])

annual = annual_summary(data)

event_sensitivity = {
    "all_years": annual,
    "exclude_2016": annual.drop(index=2016),
    "exclude_2018": annual.drop(index=2018),
    "exclude_2016_and_2018": annual.drop(index=[2016, 2018]),
}

without_july_2018 = data[~(
    (data["date"].dt.year == 2018) & (data["date"].dt.month == 7)
)]
event_sensitivity["exclude_July_2018"] = annual_summary(without_july_2018)

first_person_annual = annual_summary(data[data["first_person"]])

years_2020_2025 = list(range(2020, 2026))
recent = data[data["year"].isin(years_2020_2025)]
platform_counts = recent.groupby(["year", "platform"]).size().unstack(fill_value=0)
platform_counts = platform_counts.reindex(years_2020_2025, fill_value=0)
stable_platforms = platform_counts.columns[
    (platform_counts.min() >= 100)
    & (platform_counts.columns != "Other indexed web sources")
].tolist()
platform_data = recent[recent["platform"].isin(stable_platforms)]
platform_standardized = direct_standardization(
    platform_data, ["platform"], years_2020_2025
)

identified_counts = (
    data.dropna(subset=["author_account"])
    .groupby("author_account").size()
    .sort_values(ascending=False)
)
top_n = math.ceil(len(identified_counts) * 0.01)
top_accounts = set(identified_counts.head(top_n).index)
author_annual = author_weighted_series(data, "year", top_accounts)
author_monthly = author_weighted_series(data, "month", top_accounts)

weekly = pd.read_csv("weekly_external_validation.csv")
weekly["hesitancy_proportion"] = weekly["n_hesitant"] / weekly["n_total"]
weekly["volume_remainder"] = STL(
    np.log1p(weekly["n_total"]), period=52, robust=True
).fit().resid
weekly["hesitancy_remainder"] = STL(
    weekly["hesitancy_proportion"], period=52, robust=True
).fit().resid
weekly["south_ili_remainder"] = STL(
    weekly["south_ili_percent"], period=52, robust=True
).fit().resid
weekly["north_ili_remainder"] = STL(
    weekly["north_ili_percent"], period=52, robust=True
).fit().resid

external_validation = pd.DataFrame([
    {"social_signal": "post_volume", "external_signal": region,
     "analysis": "raw", **moving_block_spearman(weekly["n_total"], weekly[column])}
    for region, column in {
        "south_ili": "south_ili_percent", "north_ili": "north_ili_percent"
    }.items()
] + [
    {"social_signal": "hesitancy_proportion", "external_signal": region,
     "analysis": "raw",
     **moving_block_spearman(weekly["hesitancy_proportion"], weekly[column])}
    for region, column in {
        "south_ili": "south_ili_percent", "north_ili": "north_ili_percent"
    }.items()
] + [
    {"social_signal": signal, "external_signal": region,
     "analysis": "STL_remainder",
     **moving_block_spearman(weekly[signal], weekly[f"{region}_remainder"])}
    for signal in ["volume_remainder", "hesitancy_remainder"]
    for region in ["south_ili", "north_ili"]
])
