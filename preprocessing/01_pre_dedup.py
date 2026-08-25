import argparse
import re
from pathlib import Path

import pandas as pd


# Required original columns
COL_TIME_RAW = "发布时间"
COL_TITLE = "标题"
COL_BODY = "正文"

# New columns
COL_TIME_STD = "标准时间"
COL_TEXT_DEDUP = "去重用文本"

# Regex patterns
RE_INVISIBLE = re.compile("[" + chr(0x200B) + "-" + chr(0x200D) + chr(0xFEFF) + "]")
RE_WHITESPACE = re.compile(r"\s+")
RE_LINEBREAK_TAB = re.compile(r"[\r\n\t]+")
RE_LINK = re.compile(r"(?:https?://|www\.)[^\s]+", flags=re.IGNORECASE)
RE_REPEAT_PUNCT = re.compile(r"([!?！？。.，,;；:：~～\-—])\1+")

PLATFORM_WORDS = [
    "网页链接",
    "展开全文",
    "全文",
    "收起全文",
]


def read_csv_chunks(path: Path, chunksize: int = 100000):
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            reader = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", chunksize=chunksize)
            first = next(iter(reader))
            # Re-open after probe
            reader = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", chunksize=chunksize)
            return enc, reader
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to read input CSV with supported encodings.")


def basic_clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.strip()
    s = s.str.replace(RE_LINEBREAK_TAB, " ", regex=True)
    s = s.str.replace(RE_INVISIBLE, "", regex=True)
    s = s.str.replace(RE_WHITESPACE, " ", regex=True).str.strip()
    return s


def merge_title_body(title: str, body: str) -> str:
    if not title and not body:
        return ""
    if title and not body:
        return title
    if body and not title:
        return body
    if title == body:
        return title
    if title in body:
        return body
    if body in title:
        return title
    return f"{title} {body}"


def light_normalize_series(s: pd.Series) -> pd.Series:
    s = s.str.replace(RE_LINK, " ", regex=True)
    for w in PLATFORM_WORDS:
        s = s.str.replace(w, " ", regex=False)
    s = s.str.lower()
    s = s.str.replace(RE_REPEAT_PUNCT, r"\1", regex=True)
    s = s.str.replace(RE_WHITESPACE, " ", regex=True).str.strip()
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="标准化时间和文本，为后续去重生成规范字段。")
    parser.add_argument("--input", required=True, help="输入 CSV 文件。")
    parser.add_argument("--output", default="preprocessed_records.csv", help="输出 CSV 文件。")
    parser.add_argument("--chunksize", type=int, default=100000, help="分块读取行数。")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than 0")

    if output_path.exists():
        output_path.unlink()
    if temp_path.exists():
        temp_path.unlink()

    enc, reader = read_csv_chunks(input_path, chunksize=args.chunksize)

    total_rows = 0
    std_time_na = 0
    dedup_text_empty = 0
    first_write = True

    required = [COL_TIME_RAW, COL_TITLE, COL_BODY]

    for i, chunk in enumerate(reader, start=1):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Add exactly two columns
        chunk[COL_TIME_STD] = pd.to_datetime(chunk[COL_TIME_RAW], errors="coerce")

        title = basic_clean_series(chunk[COL_TITLE])
        body = basic_clean_series(chunk[COL_BODY])
        merged = pd.Series(
            (merge_title_body(t, b) for t, b in zip(title.tolist(), body.tolist())),
            index=chunk.index,
            dtype="object",
        )
        chunk[COL_TEXT_DEDUP] = light_normalize_series(merged)

        # Stats
        total_rows += len(chunk)
        std_time_na += int(chunk[COL_TIME_STD].isna().sum())
        dedup_text_empty += int((chunk[COL_TEXT_DEDUP].fillna("").str.strip() == "").sum())

        # Write to temp file safely
        chunk.to_csv(
            temp_path,
            mode="w" if first_write else "a",
            index=False,
            encoding="utf-8-sig",
            header=first_write,
        )
        first_write = False

        if i % 5 == 0:
            print(f"Processed chunks: {i}, rows: {total_rows}")

    # Atomic finalize: only after complete success
    temp_path.replace(output_path)

    print(f"Encoding used: {enc}")
    print(f"Output file: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with null {COL_TIME_STD}: {std_time_na}")
    print(f"Rows with empty {COL_TEXT_DEDUP}: {dedup_text_empty}")


if __name__ == "__main__":
    main()
