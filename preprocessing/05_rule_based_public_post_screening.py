from __future__ import annotations

"""Conservative rule-based screening of obvious non-public posts.

This stage removes only records with strong, concordant evidence of a
non-public source, such as an institutional notice, a news report, health
education content, or commercial promotion. Every other record is retained
as a candidate for the subsequent LLM-based public-post screening stage.
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


CSV_ENCODING = "utf-8-sig"
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_OUTPUT_PREFIX = "public_post_rule_screening"

FIELD_CANDIDATES = {
    "author": ["作者名称", "作者", "author_name", "author", "昵称", "用户名", "发布者"],
    "title": ["标题", "title", "headline", "subject"],
    "body": ["正文", "内容", "body", "text", "全文", "帖子内容"],
    "summary": ["摘要", "summary", "简介", "导语", "描述"],
    "source": ["来源网站", "来源", "source", "网站", "平台"],
    "media_type": ["媒体类型", "media_type", "类型", "内容类型"],
    "publisher_type": ["发布者性质", "发布者类型", "publisher_type", "账号类型"],
}

INSTITUTION_AUTHOR_TERMS = [
    "医院",
    "卫生院",
    "妇幼",
    "疾控",
    "cdc",
    "接种门诊",
    "社区卫生服务中心",
    "卫生服务站",
    "门诊部",
    "保健院",
    "卫生健康",
    "卫健",
    "医联体",
    "医共体",
]

SCHOOL_AUTHOR_TERMS = [
    "幼儿园",
    "小学",
    "中学",
    "学校",
    "教育局",
    "教育集团",
    "校园",
    "托育",
]

MEDIA_AUTHOR_TERMS = [
    "新闻",
    "日报",
    "晚报",
    "广播",
    "电视台",
    "融媒体",
    "传媒",
    "报社",
    "记者",
    "资讯",
]

PROFESSIONAL_AUTHOR_TERMS = [
    "医生",
    "医师",
    "药师",
    "护士",
    "健康科普",
    "疫苗科普",
    "健康讲堂",
    "营养师",
    "健康号",
]

COMMERCIAL_AUTHOR_TERMS = [
    "大药房",
    "药店",
    "药房",
    "门店",
    "旗舰店",
    "生物",
    "医药",
    "药业",
    "商贸",
    "商城",
]

EXPLICIT_NONPUBLIC_PUBLISHER_TERMS = [
    "机构",
    "官方",
    "政府",
    "媒体",
    "学校",
    "医疗机构",
    "企业",
    "商家",
]

NOTICE_TERMS = [
    "通知",
    "公告",
    "接种安排",
    "门诊安排",
    "接种时间",
    "接种地点",
    "接种对象",
    "接种费用",
    "预约方式",
    "预约电话",
    "请携带",
    "请居民",
    "请家长",
    "具体安排如下",
    "免费接种",
]

NEWS_TERMS = [
    "本报讯",
    "记者从",
    "通讯员",
    "据悉",
    "获悉",
    "发布会",
    "通报",
    "原标题",
    "本文转载",
    "来源：",
    "专家表示",
]

SCIENCE_TERMS = [
    "健康科普",
    "健康提示",
    "科普课堂",
    "知识讲堂",
    "医生提醒",
    "专家建议",
    "健康宣教",
    "接种禁忌",
    "适用人群",
    "注意事项",
    "常见问题",
]

MARKETING_TERMS = [
    "优惠",
    "团购",
    "套餐",
    "促销",
    "限时",
    "私信咨询",
    "点击咨询",
    "扫码咨询",
    "咨询热线",
    "欢迎到店",
    "名额有限",
    "代预约",
]

PERSONAL_DECISION_PATTERNS = [
    re.compile(r"我.{0,15}(打了|打完|去打|准备打|预约|约了|不打|犹豫|担心|后悔)"),
    re.compile(r"我家.{0,12}(孩子|宝宝|家人|老人).{0,12}(打|接种|预约|不打)"),
    re.compile(r"(带孩子去打|给孩子打|陪家人打|我们家.{0,12}(接种|不打|预约))"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="以保守规则排除明显非公众发文，其余记录交由后续 LLM 筛选。"
    )
    parser.add_argument("--input", required=True, help="输入文件路径，支持 CSV、XLSX 或 XLS。")
    parser.add_argument("--output-dir", default="public_post_rule_output", help="输出目录。")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="输出文件名前缀。")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE, help="CSV 分块读取行数。")
    return parser


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "").replace("\u200b", " ")
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def choose_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return ""


def resolve_fields(columns: list[str]) -> dict[str, str]:
    field_map = {
        field: choose_column(columns, candidates)
        for field, candidates in FIELD_CANDIDATES.items()
    }
    if not field_map["title"] and not field_map["body"]:
        raise KeyError("未识别到标题或正文字段。")
    return field_map


def get_value(row: pd.Series, column: str) -> str:
    return normalize_text(row.get(column, "")) if column else ""


def has_personal_decision_expression(text: str) -> bool:
    return any(pattern.search(text) for pattern in PERSONAL_DECISION_PATTERNS)


def classify_record(row: pd.Series, field_map: dict[str, str]) -> tuple[str, str]:
    author = get_value(row, field_map["author"])
    title = get_value(row, field_map["title"])
    body = get_value(row, field_map["body"])
    summary = get_value(row, field_map["summary"])
    source = get_value(row, field_map["source"])
    media_type = get_value(row, field_map["media_type"])
    publisher_type = get_value(row, field_map["publisher_type"])

    content = " ".join(part for part in [title, body, summary] if part)
    author_context = " ".join(
        part for part in [author, source, media_type, publisher_type] if part
    )

    # Personal decision language prevents a rule-only exclusion. Ambiguous cases
    # are deliberately delegated to the subsequent semantic screening stage.
    if has_personal_decision_expression(content):
        return "llm_review", ""

    institution_author = contains_any(author_context, INSTITUTION_AUTHOR_TERMS)
    school_author = contains_any(author_context, SCHOOL_AUTHOR_TERMS)
    media_author = contains_any(author_context, MEDIA_AUTHOR_TERMS)
    professional_author = contains_any(author_context, PROFESSIONAL_AUTHOR_TERMS)
    commercial_author = contains_any(author_context, COMMERCIAL_AUTHOR_TERMS)
    explicit_nonpublic = contains_any(publisher_type, EXPLICIT_NONPUBLIC_PUBLISHER_TERMS)

    notice_content = contains_any(content, NOTICE_TERMS)
    news_content = contains_any(content, NEWS_TERMS)
    science_content = contains_any(content, SCIENCE_TERMS)
    marketing_content = contains_any(content, MARKETING_TERMS)

    if commercial_author and marketing_content:
        return "rule_exclude", "commercial_promotion"
    if institution_author and notice_content:
        return "rule_exclude", "institutional_notice"
    if school_author and notice_content:
        return "rule_exclude", "school_notice"
    if media_author and news_content:
        return "rule_exclude", "news_or_media_content"
    if professional_author and science_content:
        return "rule_exclude", "professional_health_education"
    if explicit_nonpublic and notice_content:
        return "rule_exclude", "nonpublic_service_notice"
    if explicit_nonpublic and news_content:
        return "rule_exclude", "nonpublic_news_content"
    if explicit_nonpublic and marketing_content:
        return "rule_exclude", "nonpublic_commercial_content"

    return "llm_review", ""


def classify_dataframe(df: pd.DataFrame, field_map: dict[str, str]) -> pd.DataFrame:
    results = [classify_record(row, field_map) for _, row in df.iterrows()]
    output = df.copy()
    output["public_post_rule_status"] = [status for status, _ in results]
    output["public_post_rule_reason"] = [reason for _, reason in results]
    return output


def append_csv(df: pd.DataFrame, path: Path, write_header: bool) -> bool:
    if df.empty:
        return write_header
    df.to_csv(
        path,
        mode="a" if write_header else "w",
        index=False,
        header=not write_header,
        encoding=CSV_ENCODING if not write_header else "utf-8",
    )
    return True


def output_paths(output_dir: Path, prefix: str, suffix: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{prefix}_all{suffix}",
        output_dir / f"{prefix}_rule_excluded{suffix}",
        output_dir / f"{prefix}_llm_candidates{suffix}",
    )


def process_csv(input_path: Path, output_dir: Path, prefix: str, chunksize: int) -> Counter[str]:
    header = pd.read_csv(input_path, nrows=0, dtype=str, encoding=CSV_ENCODING)
    field_map = resolve_fields(list(header.columns))
    all_path, excluded_path, candidates_path = output_paths(output_dir, prefix, ".csv")
    for path in [all_path, excluded_path, candidates_path]:
        if path.exists():
            path.unlink()

    wrote_all = wrote_excluded = wrote_candidates = False
    counts: Counter[str] = Counter()
    output_columns = list(header.columns) + ["public_post_rule_status", "public_post_rule_reason"]

    reader = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        encoding=CSV_ENCODING,
        on_bad_lines="error",
    )
    for chunk in reader:
        classified = classify_dataframe(chunk, field_map)
        excluded = classified[classified["public_post_rule_status"] == "rule_exclude"]
        candidates = classified[classified["public_post_rule_status"] == "llm_review"]
        counts.update(classified["public_post_rule_status"])
        counts.update(classified.loc[excluded.index, "public_post_rule_reason"])
        wrote_all = append_csv(classified, all_path, wrote_all)
        wrote_excluded = append_csv(excluded, excluded_path, wrote_excluded)
        wrote_candidates = append_csv(candidates, candidates_path, wrote_candidates)

    for path, wrote in [
        (all_path, wrote_all),
        (excluded_path, wrote_excluded),
        (candidates_path, wrote_candidates),
    ]:
        if not wrote:
            pd.DataFrame(columns=output_columns).to_csv(path, index=False, encoding=CSV_ENCODING)
    return counts


def process_excel(input_path: Path, output_dir: Path, prefix: str) -> Counter[str]:
    df = pd.read_excel(input_path, dtype=str).fillna("")
    field_map = resolve_fields(list(df.columns))
    classified = classify_dataframe(df, field_map)
    excluded = classified[classified["public_post_rule_status"] == "rule_exclude"]
    candidates = classified[classified["public_post_rule_status"] == "llm_review"]
    all_path, excluded_path, candidates_path = output_paths(output_dir, prefix, ".xlsx")
    classified.to_excel(all_path, index=False)
    excluded.to_excel(excluded_path, index=False)
    candidates.to_excel(candidates_path, index=False)
    counts: Counter[str] = Counter(classified["public_post_rule_status"])
    counts.update(excluded["public_post_rule_reason"])
    return counts


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize 必须大于 0")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        counts = process_csv(input_path, output_dir, args.output_prefix, args.chunksize)
    elif suffix in {".xlsx", ".xls"}:
        counts = process_excel(input_path, output_dir, args.output_prefix)
    else:
        raise ValueError("仅支持 CSV、XLSX 或 XLS 文件。")

    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Rule-excluded records: {counts.get('rule_exclude', 0)}")
    print(f"Candidates for LLM review: {counts.get('llm_review', 0)}")
    print("Rule-exclusion reasons:")
    for reason, count in sorted(counts.items()):
        if reason not in {"rule_exclude", "llm_review", ""}:
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
