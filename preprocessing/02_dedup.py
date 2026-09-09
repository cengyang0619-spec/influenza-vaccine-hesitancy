import pandas as pd


data = pd.read_csv("preprocessed_records.csv")
data["duplicate_reason"] = ""

url = data["URL"].str.strip()
url_duplicate = url.ne("") & url.duplicated(keep="first")
data.loc[url_duplicate, "duplicate_reason"] = "duplicate_URL"

remaining = data["duplicate_reason"] == ""
text_duplicate = data.loc[remaining, ["标题", "正文"]].duplicated(keep="first")
data.loc[text_duplicate.index[text_duplicate], "duplicate_reason"] = "duplicate_title_and_body"

deduplicated_records = data[data["duplicate_reason"] == ""].copy()
duplicate_records = data[data["duplicate_reason"] != ""].copy()
