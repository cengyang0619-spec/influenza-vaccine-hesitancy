import re

import pandas as pd


meaningless_short_phrases = {
    "转发", "哈哈", "嗯", "哦", "图片", "视频", "来了", "收到", "呵呵", "好的", "知道了"
}
meaningful_short_phrases = {"甲流", "疫苗", "发烧", "流感", "高热", "咳嗽", "接种"}
template_placeholders = {
    "转发微博", "转发", "分享图片", "分享视频", "图片", "视频", "网页链接",
    "原图", "无法解析的图片", "查看图片", "查看原图"
}

invisible = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
url = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
valid_character = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
symbol = re.compile(r"[^\w\s\u4e00-\u9fff]")
laughter = re.compile(r"^[哈哈啊呵嗯哦哼]+$")


def clean_text(value):
    text = "" if pd.isna(value) else str(value)
    text = invisible.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(text):
    return re.sub(r"[\s\W_]+", "", text)


def link_only(text):
    remainder = compact(url.sub(" ", text).replace("网页链接", " "))
    return bool(url.search(text) or "网页链接" in text) and remainder == ""


def garbled(text):
    characters = [x for x in text if not x.isspace()]
    if len(characters) < 6:
        return False
    valid = sum(bool(valid_character.match(x)) for x in characters)
    symbols = sum(bool(symbol.match(x)) for x in characters)
    abnormal = max(len(characters) - valid - symbols, 0) + text.count("�")
    return valid / len(characters) <= 0.35 and abnormal / len(characters) >= 0.55


def symbol_noise(text):
    characters = [x for x in text if not x.isspace()]
    if len(characters) < 5:
        return False
    valid = sum(bool(valid_character.match(x)) for x in characters)
    symbols = sum(bool(symbol.match(x)) for x in characters)
    return valid / len(characters) <= 0.20 and symbols / len(characters) >= 0.75


def classify_structure(title, body):
    text = f"{clean_text(title)} {clean_text(body)}".strip()
    short = compact(text)

    if text == "":
        return "empty_text"
    if link_only(text):
        return "link_only"
    if garbled(text):
        return "garbled_text"
    if symbol_noise(text):
        return "symbol_noise"
    if short not in meaningful_short_phrases and len(short) <= 3 and (
        short in meaningless_short_phrases or laughter.fullmatch(short)
    ):
        return "meaningless_short_text"
    if short in template_placeholders:
        return "template_placeholder"
    return None


data = pd.read_csv("deduplicated_records.csv")
data["structure_exclusion_reason"] = [
    classify_structure(title, body)
    for title, body in zip(data["标题"], data["正文"])
]

structure_clean_records = data[data["structure_exclusion_reason"].isna()].copy()
structure_excluded_records = data[data["structure_exclusion_reason"].notna()].copy()
