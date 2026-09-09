import re

import pandas as pd


def clean_text(value):
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


data = pd.read_csv("raw_records.csv")
data["发布时间"] = pd.to_datetime(data["发布时间"])
data["标题"] = data["标题"].map(clean_text)
data["正文"] = data["正文"].map(clean_text)
