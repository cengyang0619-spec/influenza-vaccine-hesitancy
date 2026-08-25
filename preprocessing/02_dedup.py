import argparse

import pandas as pd
from pathlib import Path
from time import perf_counter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def u(*codes):
    return "".join(chr(c) for c in codes)


# Column names
COL_URL = "URL"
COL_TITLE = u(0x6807, 0x9898)
COL_BODY = u(0x6B63, 0x6587)

# Chunk size for incremental processing
CHUNK_SIZE = 50_000


def detect_csv_format(path: Path) -> tuple[str, str]:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        for engine in ["c", "python"]:
            try:
                pd.read_csv(
                    path,
                    encoding=enc,
                    engine=engine,
                    on_bad_lines="skip",
                    nrows=5,
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                )
                return enc, engine
            except UnicodeDecodeError:
                break
            except Exception:
                continue
    raise ValueError("Unable to read input CSV with supported encodings/engines.")


def append_csv_chunk(df: pd.DataFrame, path: Path, wrote_header: bool) -> bool:
    if df.empty:
        return wrote_header

    mode = "a" if wrote_header else "w"
    header = not wrote_header
    encoding = "utf-8" if wrote_header else "utf-8-sig"
    df.to_csv(path, mode=mode, index=False, header=header, encoding=encoding)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="按 URL 及标题—正文组合对 CSV 记录去重。")
    parser.add_argument("--input", required=True, help="输入 CSV 文件。")
    parser.add_argument("--output-dir", default="deduplication_output", help="输出目录。")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_path = output_dir / "deduplicated_records.csv"
    dup_path = output_dir / "duplicate_records.csv"
    keep_tmp_path = output_dir / "deduplicated_records.csv.tmp"
    dup_tmp_path = output_dir / "duplicate_records.csv.tmp"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Remove stale temp files
    if keep_tmp_path.exists():
        keep_tmp_path.unlink()
    if dup_tmp_path.exists():
        dup_tmp_path.unlink()

    enc, engine = detect_csv_format(input_path)

    header_df = pd.read_csv(
        input_path,
        encoding=enc,
        engine=engine,
        on_bad_lines="skip",
        nrows=0,
    )
    header_df.columns = [str(c).strip() for c in header_df.columns]
    all_cols = list(header_df.columns)

    required = [COL_TITLE, COL_BODY]
    missing = [c for c in required if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use python engine for chunked processing to avoid C-engine OOM on very wide/long text rows.
    reader = pd.read_csv(
        input_path,
        encoding=enc,
        engine="python",
        on_bad_lines="skip",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    seen_urls: set[str] = set()
    seen_title_body_hash: set[int] = set()

    input_rows = 0
    step1_removed = 0
    step2_removed = 0
    final_kept = 0

    wrote_keep_header = False
    wrote_dup_header = False

    start_time = perf_counter()
    chunk_idx = 0
    chunk_iter = tqdm(reader, desc="Dedup", unit="chunk") if tqdm is not None else reader

    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        input_rows += len(chunk)

        # Step 1: URL exact dedup (keep first non-empty URL globally)
        step1_keep = pd.Series(True, index=chunk.index)
        if COL_URL in chunk.columns:
            url_norm = chunk[COL_URL].str.strip()
            has_url = url_norm.ne("")
            first_in_chunk = ~url_norm.duplicated(keep="first")
            not_seen_global = pd.Series(
                [u not in seen_urls if u else True for u in url_norm],
                index=chunk.index,
            )
            step1_keep = (~has_url) | (has_url & first_in_chunk & not_seen_global)

            new_urls = url_norm[step1_keep & has_url].unique().tolist()
            seen_urls.update(new_urls)

        step1_removed += int((~step1_keep).sum())

        # Step 2: title + body exact dedup (global, incremental)
        s1_keep_df = chunk.loc[step1_keep]
        key_df = s1_keep_df[[COL_TITLE, COL_BODY]]
        key_hash = pd.util.hash_pandas_object(key_df, index=False, categorize=False).to_numpy(dtype="uint64", copy=False)

        hash_series = pd.Series(key_hash, index=s1_keep_df.index)
        first_in_chunk = ~hash_series.duplicated(keep="first")
        not_seen_global = pd.Series(
            [int(h) not in seen_title_body_hash for h in key_hash],
            index=s1_keep_df.index,
        )
        step2_keep = first_in_chunk & not_seen_global

        step2_removed += int((~step2_keep).sum())
        seen_title_body_hash.update(int(h) for h in key_hash[step2_keep.to_numpy()])

        final_keep_mask = pd.Series(False, index=chunk.index)
        final_keep_mask.loc[s1_keep_df.index] = step2_keep.to_numpy()

        keep_chunk = chunk.loc[final_keep_mask]
        dup_chunk = chunk.loc[~final_keep_mask]

        final_kept += len(keep_chunk)

        wrote_keep_header = append_csv_chunk(keep_chunk, keep_tmp_path, wrote_keep_header)
        wrote_dup_header = append_csv_chunk(dup_chunk, dup_tmp_path, wrote_dup_header)

        elapsed = perf_counter() - start_time
        if tqdm is not None:
            chunk_iter.set_postfix(rows=input_rows, kept=final_kept, elapsed_s=f"{elapsed:.1f}")
        elif chunk_idx % 10 == 0:
            rate = input_rows / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] chunks={chunk_idx}, rows={input_rows}, kept={final_kept}, "
                f"elapsed={elapsed:.1f}s, rate={rate:.0f} rows/s",
                flush=True,
            )

    # Ensure output files exist with headers even if empty
    if not wrote_keep_header:
        pd.DataFrame(columns=all_cols).to_csv(keep_tmp_path, index=False, encoding="utf-8-sig")
    if not wrote_dup_header:
        pd.DataFrame(columns=all_cols).to_csv(dup_tmp_path, index=False, encoding="utf-8-sig")

    keep_tmp_path.replace(keep_path)
    dup_tmp_path.replace(dup_path)

    print(f"Encoding used: {enc}")
    print("CSV engine: python (forced for stability)")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Input rows: {input_rows}")
    print(f"Step1_URL_removed: {step1_removed}")
    print(f"Step2_title_body_removed: {step2_removed}")
    print(f"Total_removed: {input_rows - final_kept}")
    print(f"Final_kept: {final_kept}")
    print(f"Saved kept file: {keep_path}")
    print(f"Saved duplicate file: {dup_path}")


if __name__ == "__main__":
    main()
