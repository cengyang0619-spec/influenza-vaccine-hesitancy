from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# -----------------------------
# Configurable parameters
# -----------------------------
CHUNKSIZE = 50000
ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "gb18030", "gbk"]

SHORT_TEXT_LEN_THRESHOLD = 3
GARBLED_MIN_LEN = 6
GARBLED_MAX_VALID_RATIO = 0.35
GARBLED_MIN_ABNORMAL_RATIO = 0.55
SYMBOL_NOISE_MIN_LEN = 5
SYMBOL_NOISE_MAX_VALID_RATIO = 0.20
SYMBOL_NOISE_MIN_SYMBOL_RATIO = 0.75

MEANINGLESS_SHORT_PHRASES = {
    "转发",
    "哈哈",
    "嗯",
    "哦",
    "图片",
    "视频",
    "来了",
    "收到",
    "呵呵",
    "好的",
    "知道了",
}

SHORT_MEANINGFUL_WHITELIST = {
    "甲流",
    "疫苗",
    "发烧",
    "流感",
    "高热",
    "咳嗽",
    "接种",
}

TEMPLATE_PLACEHOLDERS = {
    "转发微博",
    "转发",
    "分享图片",
    "分享视频",
    "图片",
    "视频",
    "网页链接",
    "原图",
    "无法解析的图片",
    "查看图片",
}

# -----------------------------
# Regex patterns
# -----------------------------
INVISIBLE_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
LINK_TOKEN_RE = re.compile(r"(?:网页链接)", re.IGNORECASE)
VALID_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
PUNCT_OR_SYMBOL_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)
LAUGHTER_LIKE_RE = re.compile(r"^[哈啊呵嗯哦哼]+$")
ONLY_PUNCT_SPACE_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = INVISIBLE_CHARS_RE.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_core_text(title: str, body: str) -> str:
    if title and body:
        return f"{title} {body}"
    return title or body or ""


def compact_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def ratio(part: int, whole: int) -> float:
    return float(part) / float(whole) if whole > 0 else 0.0


def is_link_only(core_text: str) -> bool:
    if not core_text:
        return False
    without_urls = URL_RE.sub(" ", core_text)
    without_link_tokens = LINK_TOKEN_RE.sub(" ", without_urls)
    remainder = compact_text(without_link_tokens)
    has_link_marker = bool(URL_RE.search(core_text) or LINK_TOKEN_RE.search(core_text))
    return has_link_marker and remainder == ""


def is_garbled_text(core_text: str) -> bool:
    chars = [ch for ch in core_text if not ch.isspace()]
    total = len(chars)
    if total < GARBLED_MIN_LEN:
        return False

    valid_count = sum(1 for ch in chars if VALID_CHAR_RE.match(ch))
    symbol_count = sum(1 for ch in chars if PUNCT_OR_SYMBOL_RE.match(ch))
    replacement_count = core_text.count("�")
    abnormal_count = max(total - valid_count - symbol_count, 0) + replacement_count

    valid_ratio = ratio(valid_count, total)
    abnormal_ratio = ratio(abnormal_count, total)
    return valid_ratio <= GARBLED_MAX_VALID_RATIO and abnormal_ratio >= GARBLED_MIN_ABNORMAL_RATIO


def is_symbol_noise(core_text: str) -> bool:
    chars = [ch for ch in core_text if not ch.isspace()]
    total = len(chars)
    if total < SYMBOL_NOISE_MIN_LEN:
        return False

    valid_count = sum(1 for ch in chars if VALID_CHAR_RE.match(ch))
    symbol_count = sum(1 for ch in chars if PUNCT_OR_SYMBOL_RE.match(ch))
    valid_ratio = ratio(valid_count, total)
    symbol_ratio = ratio(symbol_count, total)

    if ONLY_PUNCT_SPACE_RE.fullmatch(core_text):
        return True
    return valid_ratio <= SYMBOL_NOISE_MAX_VALID_RATIO and symbol_ratio >= SYMBOL_NOISE_MIN_SYMBOL_RATIO


def is_meaningless_short(core_text: str) -> bool:
    compact = compact_text(core_text)
    if not compact:
        return False
    if compact in SHORT_MEANINGFUL_WHITELIST:
        return False
    if len(compact) <= SHORT_TEXT_LEN_THRESHOLD and compact in MEANINGLESS_SHORT_PHRASES:
        return True
    if len(compact) <= SHORT_TEXT_LEN_THRESHOLD and LAUGHTER_LIKE_RE.fullmatch(compact):
        return True
    return False


def is_template_placeholder(core_text: str) -> bool:
    compact = compact_text(core_text)
    if not compact:
        return False
    if compact in TEMPLATE_PLACEHOLDERS:
        return True
    if compact in {
        "网页链接",
        "原图",
        "查看原图",
        "查看图片",
    }:
        return True
    return False


def classify_structure_noise(title_value: object, body_value: object) -> Optional[str]:
    title = clean_text(title_value)
    body = clean_text(body_value)
    core_text = build_core_text(title, body)

    # 1) Empty text
    if not title and not body:
        return "empty_text"
    if not core_text:
        return "empty_text"

    # 2) Link-only text
    if is_link_only(core_text):
        return "link_only"

    # 3) Garbled text
    if is_garbled_text(core_text):
        return "garbled_text"

    # 4) Symbol-stacked noise
    if is_symbol_noise(core_text):
        return "symbol_noise"

    # 5) Very short meaningless text
    if is_meaningless_short(core_text):
        return "meaningless_short_text"

    # 6) Template placeholders
    if is_template_placeholder(core_text):
        return "template_placeholder"

    return None


def detect_encoding(csv_path: Path) -> str:
    last_error: Optional[Exception] = None
    for enc in ENCODING_CANDIDATES:
        try:
            pd.read_csv(
                csv_path,
                encoding=enc,
                nrows=50,
                on_bad_lines="skip",
                engine="c",
            )
            return enc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Unable to decode {csv_path} with candidates {ENCODING_CANDIDATES}") from last_error


def count_data_rows_fast(csv_path: Path) -> int:
    # Fast line-based counter for progress total; assumes one record per line after header.
    # For datasets with embedded newlines inside quoted fields, this can overcount.
    line_count = 0
    with csv_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            line_count += block.count(b"\n")
    return max(line_count - 1, 0)


def resolve_text_columns(df_columns: list[str]) -> tuple[str, str]:
    stripped_to_raw = {col.strip().lstrip("\ufeff"): col for col in df_columns}
    title_key = "标题"
    body_key = "正文"
    if title_key not in stripped_to_raw or body_key not in stripped_to_raw:
        raise KeyError(f"Missing required columns. Available columns: {df_columns}")
    return stripped_to_raw[title_key], stripped_to_raw[body_key]


def main() -> None:
    parser = argparse.ArgumentParser(description="清除空文本、乱码、符号噪声和模板占位内容。")
    parser.add_argument("--input", required=True, help="输入 CSV 文件。")
    parser.add_argument("--output-dir", default="structure_cleaning_output", help="输出目录。")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_path = output_dir / "keep_structure.csv"
    drop_path = output_dir / "drop_structure.csv"
    keep_tmp_path = output_dir / "keep_structure.csv.tmp"
    drop_tmp_path = output_dir / "drop_structure.csv.tmp"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for tmp in (keep_tmp_path, drop_tmp_path):
        if tmp.exists():
            tmp.unlink()

    encoding = detect_encoding(input_path)
    header_df = pd.read_csv(
        input_path,
        encoding=encoding,
        nrows=0,
        on_bad_lines="skip",
        engine="c",
    )
    title_col, body_col = resolve_text_columns(list(header_df.columns))
    output_columns = list(header_df.columns) + ["structure_keep_flag", "structure_drop_reason"]
    estimated_total_rows = count_data_rows_fast(input_path)
    print(f"Estimated rows for progress bar: {estimated_total_rows}")

    keep_count = 0
    drop_count = 0
    total_count = 0
    reason_counter: Counter[str] = Counter()
    chunk_count = 0
    wrote_keep_header = False
    wrote_drop_header = False
    reader = pd.read_csv(
        input_path,
        encoding=encoding,
        chunksize=CHUNKSIZE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
        engine="c",
    )

    with tqdm(
        total=estimated_total_rows if estimated_total_rows > 0 else None,
        desc="Structure cleaning",
        unit="rows",
        dynamic_ncols=True,
    ) as pbar:
        for chunk in reader:
            chunk_count += 1
            reasons = [classify_structure_noise(t, b) for t, b in zip(chunk[title_col], chunk[body_col])]
            reason_series = pd.Series(reasons, index=chunk.index, dtype="object")
            keep_mask = reason_series.isna()
            drop_mask = ~keep_mask

            keep_chunk = chunk.loc[keep_mask].copy()
            drop_chunk = chunk.loc[drop_mask].copy()

            keep_chunk["structure_keep_flag"] = 1
            keep_chunk["structure_drop_reason"] = ""
            drop_chunk["structure_keep_flag"] = 0
            drop_chunk["structure_drop_reason"] = reason_series.loc[drop_mask].astype(str).values

            if not keep_chunk.empty:
                keep_chunk.to_csv(
                    keep_tmp_path,
                    mode="a",
                    index=False,
                    header=not wrote_keep_header,
                    encoding="utf-8-sig",
                )
                wrote_keep_header = True

            if not drop_chunk.empty:
                drop_chunk.to_csv(
                    drop_tmp_path,
                    mode="a",
                    index=False,
                    header=not wrote_drop_header,
                    encoding="utf-8-sig",
                )
                wrote_drop_header = True

            keep_rows = len(keep_chunk)
            drop_rows = len(drop_chunk)
            chunk_rows = len(chunk)

            keep_count += keep_rows
            drop_count += drop_rows
            total_count += chunk_rows
            reason_counter.update(reason_series.loc[drop_mask].astype(str).tolist())

            pbar.update(chunk_rows)
            pbar.set_postfix(chunks=chunk_count, keep=keep_count, drop=drop_count)

    if total_count != keep_count + drop_count:
        raise RuntimeError(
            "Consistency check failed: keep + drop != dedup_total "
            f"({keep_count} + {drop_count} != {total_count})"
        )

    # Ensure output files exist even if a side is empty.
    if not wrote_keep_header:
        pd.DataFrame(columns=output_columns).to_csv(keep_tmp_path, index=False, encoding="utf-8-sig")
    if not wrote_drop_header:
        pd.DataFrame(columns=output_columns).to_csv(drop_tmp_path, index=False, encoding="utf-8-sig")

    keep_tmp_path.replace(keep_path)
    drop_tmp_path.replace(drop_path)

    print("\n=== Structure Cleaning Summary ===")
    print(f"Input encoding: {encoding}")
    print(f"Input file: {input_path}")
    print(f"Total rows: {total_count}")
    print(f"Rows kept: {keep_count}")
    print(f"Rows dropped: {drop_count}")
    print("Drop reason counts:")
    if reason_counter:
        for reason, count in reason_counter.most_common():
            print(f"  {reason}: {count}")
    else:
        print("  (none)")
    print(f"Kept records: {keep_path}")
    print(f"Removed records: {drop_path}")
    print(f"Consistency check: {'PASS' if total_count == keep_count + drop_count else 'FAIL'}")


if __name__ == "__main__":
    main()
