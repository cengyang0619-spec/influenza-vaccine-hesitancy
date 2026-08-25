from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

# =============================
# 1) 配置区
# =============================
TITLE_CANDIDATES = ["标题", "title", "Title"]
BODY_CANDIDATES = ["正文", "内容", "content", "Content", "body", "Body", "text", "Text"]

CHUNK_SIZE = 100_000
CSV_ENCODING = "utf-8-sig"

# =============================
# 2) 规则词表
# =============================
FLU_EXPLICIT_TERMS = [
    "流感疫苗",
    "流感预防针",
    "流感针",
    "鼻喷流感疫苗",
    "三价流感疫苗",
    "四价流感疫苗",
    "流感裂解疫苗",
    "流感亚单位疫苗",
    "流感减毒活疫苗",
]

FLU_TERMS = ["流感", "甲流", "乙流", "influenza", "flu"]
VACCINE_TERMS = ["疫苗", "预防针"]
ACTION_TERMS = [
    "打",
    "接种",
    "注射",
    "预约",
    "到苗",
    "到货",
    "缺货",
    "价格",
    "费用",
    "副作用",
    "不良反应",
    "保护效果",
    "保护率",
    "适用人群",
    "要不要打",
    "有必要打吗",
    "哪里打",
    "社区医院",
    "门诊",
]

DROP_ANIMAL_TERMS = [
    "禽流感", "h5n1", "h7n9", "鸡", "鸭", "鹅", "养殖", "养鸡", "养鸭", "养殖场", "畜牧", "兽医", "猪场",
    "宠物疫苗", "犬疫苗", "猫疫苗", "动物疫苗",
]
DROP_ECOMMERCE_TERMS = [
    "秒杀", "优惠券", "领券", "下单", "拼团", "团购", "直播间", "橱窗", "小黄车", "店铺", "客服", "现货", "发货",
    "包邮", "淘宝", "京东", "拼多多", "抖音小店", "链接",
]
DROP_STOCK_TERMS = ["涨停", "跌停", "概念股", "板块", "资金流入", "主力", "建仓", "k线", "盘中", "市值", "股价", "财报"]
DROP_HIB_TERMS = ["流感嗜血杆菌", "hib", "b型流感嗜血杆菌", "嗜血杆菌疫苗"]
DROP_DISEASE_TERMS = ["甲流", "乙流", "流感高发", "流感症状", "发烧", "咳嗽", "奥司他韦", "医院爆满", "检测", "感染", "退烧", "门诊"]
DROP_OTHER_VACCINE_TERMS = [
    "新冠疫苗", "hpv", "宫颈癌疫苗", "乙肝疫苗", "肺炎疫苗", "狂犬疫苗", "麻腮风", "水痘疫苗", "带状疱疹疫苗",
]
DROP_GENERAL_HEALTH_TERMS = ["增强免疫力", "多喝水", "勤洗手", "戴口罩", "多锻炼", "少去人多地方"]

BRAND_TERMS = ["华兰", "科兴", "长生", "金迪克", "国药", "四价流感", "三价流感", "鼻喷流感"]
VACCINATION_CONTEXT_TERMS = ["打", "接种", "预约", "到苗", "门诊", "价格", "缺货", "副作用", "儿童", "老人", "适合", "效果"]

BORDERLINE_WEAK_PRODUCT_TERMS = ["三价", "四价", "鼻喷", "预防针"]
BORDERLINE_NOTICE_TERMS = ["通知", "提醒", "门诊接种安排", "接种安排", "科普", "新闻", "公告"]
BORDERLINE_CONTEXT_DEPENDENT_TERMS = ["这个我肯定不打", "去年打了今年", "去年打了，今年", "到了但没去", "再看看"]

# =============================
# 3) 文本处理
# =============================
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def has_any(text: str, terms: Iterable[str]) -> bool:
    return any(t in text for t in terms)


def find_text_columns(columns: list[str]) -> tuple[str, str]:
    normalized_map = {unicodedata.normalize("NFKC", c).strip().lower(): c for c in columns}

    title_col = None
    for c in TITLE_CANDIDATES:
        key = unicodedata.normalize("NFKC", c).strip().lower()
        if key in normalized_map:
            title_col = normalized_map[key]
            break

    body_col = None
    for c in BODY_CANDIDATES:
        key = unicodedata.normalize("NFKC", c).strip().lower()
        if key in normalized_map:
            body_col = normalized_map[key]
            break

    if title_col is None or body_col is None:
        raise KeyError(
            f"无法识别标题/正文列。可用列: {columns}; 标题候选: {TITLE_CANDIDATES}; 正文候选: {BODY_CANDIDATES}"
        )

    return title_col, body_col


# =============================
# 4) strong_keep_signal
# =============================
EXPLICIT_FLU_PHRASE_RE = re.compile(
    r"(流感疫苗|鼻喷流感疫苗|三价流感疫苗|四价流感疫苗|流感预防针|流感针)"
)

ACTION_BOUND_RE = re.compile(
    r"((打|接种|注射)流感疫苗|流感疫苗(预约|副作用|不良反应|价格|费用|保护效果|保护率|到苗|缺货|门诊))"
)


def strong_keep_signal(text_all: str) -> bool:
    if EXPLICIT_FLU_PHRASE_RE.search(text_all):
        return True

    if ACTION_BOUND_RE.search(text_all):
        return True

    if has_any(text_all, FLU_TERMS) and has_any(text_all, VACCINE_TERMS) and has_any(text_all, ACTION_TERMS):
        return True

    return False


# =============================
# 5) 分类函数
# =============================
DECISION_PATTERNS = [
    re.compile(r"今年要不要打流感疫苗"),
    re.compile(r"孩子要不要接种流感疫苗"),
    re.compile(r"去年打了今年还要打吗"),
    re.compile(r"流感疫苗有没有必要"),
    re.compile(r"流感疫苗效果怎么样"),
    re.compile(r"流感疫苗副作用大不大"),
]

FLU_VACCINE_OBJECT_RE = re.compile(r"(流感疫苗|流感预防针|流感针|鼻喷流感疫苗|三价流感疫苗|四价流感疫苗)")


DROP_TRIGGERS = [
    "DROP_ANIMAL_CONTEXT",
    "DROP_ECOMMERCE_CONTEXT",
    "DROP_STOCK_CONTEXT",
    "DROP_HIB_CONTEXT",
    "DROP_DISEASE_ONLY",
    "DROP_OTHER_VACCINE_ONLY",
    "DROP_GENERAL_HEALTH_ONLY",
]

KEEP_TRIGGERS = [
    "KEEP_EXPLICIT_FLU_VACCINE",
    "KEEP_FLU_VACCINE_ACTION_BOUND",
    "KEEP_VACCINATION_DECISION",
    "KEEP_BRAND_WITH_VACCINATION_CONTEXT",
]

BORDERLINE_TRIGGERS = [
    "BORDERLINE_WEAK_FLU_VACCINE_LINK",
    "BORDERLINE_WEAK_PRODUCT_FORM",
    "BORDERLINE_BRAND_ONLY",
    "BORDERLINE_NOTICE_OR_NEWS",
    "BORDERLINE_CONTEXT_DEPENDENT",
    "BORDERLINE_MIXED_SIGNAL",
    "BORDERLINE_DEFAULT",
]


def classify_row(title_value: object, body_value: object) -> tuple[str, str]:
    title = normalize_text(title_value)
    body = normalize_text(body_value)
    text_all = f"{title} {body}".strip()

    strong_keep = strong_keep_signal(text_all)
    has_flu_object = FLU_VACCINE_OBJECT_RE.search(text_all) is not None

    # 第1步: drop_rule (仅在 strong_keep_signal=False 时生效)
    if not strong_keep:
        if has_any(text_all, DROP_ANIMAL_TERMS):
            return "drop_rule", "DROP_ANIMAL_CONTEXT"
        if has_any(text_all, DROP_ECOMMERCE_TERMS):
            return "drop_rule", "DROP_ECOMMERCE_CONTEXT"
        if has_any(text_all, DROP_STOCK_TERMS):
            return "drop_rule", "DROP_STOCK_CONTEXT"
        if has_any(text_all, DROP_HIB_TERMS):
            return "drop_rule", "DROP_HIB_CONTEXT"
        if has_any(text_all, DROP_DISEASE_TERMS) and not has_flu_object:
            return "drop_rule", "DROP_DISEASE_ONLY"
        if has_any(text_all, DROP_OTHER_VACCINE_TERMS) and not has_flu_object:
            return "drop_rule", "DROP_OTHER_VACCINE_ONLY"
        if has_any(text_all, DROP_GENERAL_HEALTH_TERMS) and not has_flu_object:
            return "drop_rule", "DROP_GENERAL_HEALTH_ONLY"

    # 第2步: keep_rule
    if has_any(text_all, FLU_EXPLICIT_TERMS):
        return "keep_rule", "KEEP_EXPLICIT_FLU_VACCINE"

    if ACTION_BOUND_RE.search(text_all):
        return "keep_rule", "KEEP_FLU_VACCINE_ACTION_BOUND"

    if any(p.search(text_all) for p in DECISION_PATTERNS):
        return "keep_rule", "KEEP_VACCINATION_DECISION"

    if has_any(text_all, BRAND_TERMS) and has_any(text_all, VACCINATION_CONTEXT_TERMS):
        return "keep_rule", "KEEP_BRAND_WITH_VACCINATION_CONTEXT"

    # 第3步: borderline_rule
    weak_related = has_any(text_all, FLU_TERMS) or has_any(text_all, VACCINE_TERMS) or has_any(text_all, BRAND_TERMS)
    noise_related = (
        has_any(text_all, DROP_ANIMAL_TERMS)
        or has_any(text_all, DROP_ECOMMERCE_TERMS)
        or has_any(text_all, DROP_STOCK_TERMS)
        or has_any(text_all, DROP_HIB_TERMS)
        or has_any(text_all, DROP_DISEASE_TERMS)
        or has_any(text_all, DROP_OTHER_VACCINE_TERMS)
        or has_any(text_all, DROP_GENERAL_HEALTH_TERMS)
    )

    if has_any(text_all, FLU_TERMS) and ("疫苗" in text_all or "打针" in text_all):
        return "borderline_rule", "BORDERLINE_WEAK_FLU_VACCINE_LINK"

    if has_any(text_all, BORDERLINE_WEAK_PRODUCT_TERMS) and not has_any(text_all, FLU_EXPLICIT_TERMS):
        return "borderline_rule", "BORDERLINE_WEAK_PRODUCT_FORM"

    if has_any(text_all, BRAND_TERMS) and not has_any(text_all, VACCINATION_CONTEXT_TERMS):
        return "borderline_rule", "BORDERLINE_BRAND_ONLY"

    if has_any(text_all, BORDERLINE_NOTICE_TERMS) and not has_flu_object:
        return "borderline_rule", "BORDERLINE_NOTICE_OR_NEWS"

    if has_any(text_all, BORDERLINE_CONTEXT_DEPENDENT_TERMS):
        return "borderline_rule", "BORDERLINE_CONTEXT_DEPENDENT"

    if weak_related and noise_related:
        return "borderline_rule", "BORDERLINE_MIXED_SIGNAL"

    return "borderline_rule", "BORDERLINE_DEFAULT"


# =============================
# 6) I/O
# =============================
def ensure_output_paths(input_file: Path, output_dir: Path) -> tuple[Path, Path, Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_file.suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise ValueError(f"仅支持 csv/xlsx 输入，当前文件: {input_file}")

    keep_path = output_dir / f"keep_rule{suffix}"
    borderline_path = output_dir / f"borderline_rule{suffix}"
    drop_path = output_dir / f"drop_rule{suffix}"
    return keep_path, borderline_path, drop_path, suffix


def print_group_counts(counter: Counter[str]) -> None:
    print("\n=== Trigger Counts ===")
    print("Drop rules:")
    for name in DROP_TRIGGERS:
        print(f"  {name}: {counter.get(name, 0)}")

    print("Keep rules:")
    for name in KEEP_TRIGGERS:
        print(f"  {name}: {counter.get(name, 0)}")

    print("Borderline rules:")
    for name in BORDERLINE_TRIGGERS:
        print(f"  {name}: {counter.get(name, 0)}")


def validate_result(df: pd.DataFrame, total_input: int) -> None:
    allowed_labels = {"keep_rule", "borderline_rule", "drop_rule"}
    if not set(df["rule_label"].unique()).issubset(allowed_labels):
        bad = set(df["rule_label"].unique()) - allowed_labels
        raise RuntimeError(f"rule_label 存在非法取值: {bad}")

    if len(df) != total_input:
        raise RuntimeError(f"总行数校验失败: output={len(df)} input={total_input}")


def classify_dataframe(df: pd.DataFrame, title_col: str, body_col: str) -> tuple[pd.DataFrame, Counter[str]]:
    outcomes = [classify_row(t, b) for t, b in zip(df[title_col], df[body_col])]
    labels = [o[0] for o in outcomes]
    triggers = [o[1] for o in outcomes]

    out_df = df.copy()
    out_df["rule_label"] = labels
    out_df["rule_trigger"] = triggers

    return out_df, Counter(triggers)


def process_csv(input_file: Path, keep_path: Path, borderline_path: Path, drop_path: Path) -> None:
    for p in (keep_path, borderline_path, drop_path):
        if p.exists():
            p.unlink()

    header_df = pd.read_csv(input_file, nrows=0, dtype=str, encoding=CSV_ENCODING)
    title_col, body_col = find_text_columns(list(header_df.columns))

    total_input = 0
    keep_count = 0
    borderline_count = 0
    drop_count = 0
    trigger_counter: Counter[str] = Counter()

    wrote_keep = False
    wrote_borderline = False
    wrote_drop = False

    for chunk in pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=CHUNK_SIZE,
        encoding=CSV_ENCODING,
        on_bad_lines="error",
    ):
        total_input += len(chunk)
        chunk_out, counter = classify_dataframe(chunk, title_col, body_col)
        trigger_counter.update(counter)

        keep_df = chunk_out.loc[chunk_out["rule_label"] == "keep_rule"]
        borderline_df = chunk_out.loc[chunk_out["rule_label"] == "borderline_rule"]
        drop_df = chunk_out.loc[chunk_out["rule_label"] == "drop_rule"]

        keep_count += len(keep_df)
        borderline_count += len(borderline_df)
        drop_count += len(drop_df)

        if not keep_df.empty:
            keep_df.to_csv(keep_path, mode="a", index=False, header=not wrote_keep, encoding="utf-8-sig")
            wrote_keep = True
        if not borderline_df.empty:
            borderline_df.to_csv(
                borderline_path,
                mode="a",
                index=False,
                header=not wrote_borderline,
                encoding="utf-8-sig",
            )
            wrote_borderline = True
        if not drop_df.empty:
            drop_df.to_csv(drop_path, mode="a", index=False, header=not wrote_drop, encoding="utf-8-sig")
            wrote_drop = True

    if total_input != keep_count + borderline_count + drop_count:
        raise RuntimeError(
            f"总行数校验失败: keep({keep_count}) + borderline({borderline_count}) + drop({drop_count}) != input({total_input})"
        )

    # 如果某集合为空，也要输出仅包含列头文件
    all_cols = list(header_df.columns) + ["rule_label", "rule_trigger"]
    if not wrote_keep:
        pd.DataFrame(columns=all_cols).to_csv(keep_path, index=False, encoding="utf-8-sig")
    if not wrote_borderline:
        pd.DataFrame(columns=all_cols).to_csv(borderline_path, index=False, encoding="utf-8-sig")
    if not wrote_drop:
        pd.DataFrame(columns=all_cols).to_csv(drop_path, index=False, encoding="utf-8-sig")

    print("\n=== Summary ===")
    print(f"Input: {input_file}")
    print(f"Total rows: {total_input}")
    print(f"keep_rule: {keep_count} ({keep_count / total_input:.2%})")
    print(f"borderline_rule: {borderline_count} ({borderline_count / total_input:.2%})")
    print(f"drop_rule: {drop_count} ({drop_count / total_input:.2%})")

    print_group_counts(trigger_counter)


def process_xlsx(input_file: Path, keep_path: Path, borderline_path: Path, drop_path: Path) -> None:
    df = pd.read_excel(input_file, dtype=str)
    df = df.fillna("")

    title_col, body_col = find_text_columns(list(df.columns))

    out_df, trigger_counter = classify_dataframe(df, title_col, body_col)

    if set(out_df.columns) != set(list(df.columns) + ["rule_label", "rule_trigger"]):
        raise RuntimeError("输出列异常：除 rule_label、rule_trigger 外不应新增任何列")

    validate_result(out_df, len(df))

    keep_df = out_df.loc[out_df["rule_label"] == "keep_rule"]
    borderline_df = out_df.loc[out_df["rule_label"] == "borderline_rule"]
    drop_df = out_df.loc[out_df["rule_label"] == "drop_rule"]

    if len(df) != len(keep_df) + len(borderline_df) + len(drop_df):
        raise RuntimeError("总行数校验失败：三集合行数和不等于输入行数")

    keep_df.to_excel(keep_path, index=False)
    borderline_df.to_excel(borderline_path, index=False)
    drop_df.to_excel(drop_path, index=False)

    total_input = len(df)
    print("\n=== Summary ===")
    print(f"Input: {input_file}")
    print(f"Total rows: {total_input}")
    print(f"keep_rule: {len(keep_df)} ({len(keep_df) / total_input:.2%})")
    print(f"borderline_rule: {len(borderline_df)} ({len(borderline_df) / total_input:.2%})")
    print(f"drop_rule: {len(drop_df)} ({len(drop_df) / total_input:.2%})")

    print_group_counts(trigger_counter)


def main() -> None:
    parser = argparse.ArgumentParser(description="按规则筛选人用季节性流感疫苗相关文本。")
    parser.add_argument("--input", required=True, help="输入文件路径，支持 CSV 或 XLSX。")
    parser.add_argument("--output-dir", default="rule_screening_output", help="输出目录。")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    keep_path, borderline_path, drop_path, suffix = ensure_output_paths(input_file, output_dir)

    if suffix == ".csv":
        process_csv(input_file, keep_path, borderline_path, drop_path)
    else:
        process_xlsx(input_file, keep_path, borderline_path, drop_path)

    print("\nOutput files:")
    print(f"  {keep_path}")
    print(f"  {borderline_path}")
    print(f"  {drop_path}")


if __name__ == "__main__":
    main()
