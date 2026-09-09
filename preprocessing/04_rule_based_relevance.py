import re
import unicodedata

import pandas as pd


explicit_flu_vaccine = [
    "流感疫苗", "流感预防针", "流感针", "鼻喷流感疫苗", "三价流感疫苗",
    "四价流感疫苗", "流感裂解疫苗", "流感亚单位疫苗", "流感减毒活疫苗"
]
flu_terms = ["流感", "甲流", "乙流", "influenza", "flu"]
vaccine_terms = ["疫苗", "预防针"]
action_terms = [
    "打", "接种", "注射", "预约", "到苗", "到货", "缺货", "价格", "费用",
    "副作用", "不良反应", "保护效果", "保护率", "适用人群", "要不要打",
    "有必要打吗", "哪里打", "社区医院", "门诊"
]

animal_terms = [
    "禽流感", "h5n1", "h7n9", "鸡", "鸭", "鹅", "养殖", "养鸡", "养鸭",
    "养殖场", "畜牧", "兽医", "猪场", "宠物疫苗", "犬疫苗", "猫疫苗", "动物疫苗"
]
ecommerce_terms = [
    "秒杀", "优惠券", "领券", "下单", "拼团", "团购", "直播间", "橱窗",
    "小黄车", "店铺", "客服", "现货", "发货", "包邮", "淘宝", "京东",
    "拼多多", "抖音小店", "链接"
]
stock_terms = [
    "涨停", "跌停", "概念股", "板块", "资金流入", "主力", "建仓", "k线",
    "盘中", "市值", "股价", "财报"
]
hib_terms = ["流感嗜血杆菌", "hib", "b型流感嗜血杆菌", "嗜血杆菌疫苗"]
disease_terms = [
    "甲流", "乙流", "流感高发", "流感症状", "发烧", "咳嗽", "奥司他韦",
    "医院爆满", "检测", "感染", "退烧", "门诊"
]
other_vaccine_terms = [
    "新冠疫苗", "hpv", "宫颈癌疫苗", "乙肝疫苗", "肺炎疫苗", "狂犬疫苗",
    "麻腮风", "水痘疫苗", "带状疱疹疫苗"
]
general_health_terms = ["增强免疫力", "多喝水", "勤洗手", "戴口罩", "多锻炼", "少去人多地方"]
brand_terms = ["华兰", "科兴", "长生", "金迪克", "国药", "四价流感", "三价流感", "鼻喷流感"]
vaccination_context = ["打", "接种", "预约", "到苗", "门诊", "价格", "缺货", "副作用", "儿童", "老人", "适合", "效果"]

weak_product_terms = ["三价", "四价", "鼻喷", "预防针"]
notice_terms = ["通知", "提醒", "门诊接种安排", "接种安排", "科普", "新闻", "公告"]
context_dependent_terms = ["这个我肯定不打", "去年打了今年", "去年打了，今年", "到了但没去", "再看看"]

action_bound = re.compile(
    r"((打|接种|注射)流感疫苗|流感疫苗(预约|副作用|不良反应|价格|费用|保护效果|保护率|到苗|缺货|门诊))"
)
decision_patterns = [
    re.compile(r"今年要不要打流感疫苗"),
    re.compile(r"孩子要不要接种流感疫苗"),
    re.compile(r"去年打了今年还要打吗"),
    re.compile(r"流感疫苗有没有必要"),
    re.compile(r"流感疫苗效果怎么样"),
    re.compile(r"流感疫苗副作用大不大"),
]


def normalize(value):
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def contains(text, terms):
    return any(term in text for term in terms)


def strong_relevance(text):
    return (
        contains(text, explicit_flu_vaccine)
        or bool(action_bound.search(text))
        or (contains(text, flu_terms) and contains(text, vaccine_terms)
            and contains(text, action_terms))
    )


def classify_relevance(title, body):
    text = f"{normalize(title)} {normalize(body)}".strip()
    strong = strong_relevance(text)
    has_flu_vaccine = contains(text, explicit_flu_vaccine)

    if not strong:
        exclusion_rules = [
            (animal_terms, "animal_context"),
            (ecommerce_terms, "ecommerce_context"),
            (stock_terms, "stock_context"),
            (hib_terms, "Haemophilus_influenzae_context"),
        ]
        for terms, reason in exclusion_rules:
            if contains(text, terms):
                return "exclude", reason
        if contains(text, disease_terms) and not has_flu_vaccine:
            return "exclude", "influenza_disease_only"
        if contains(text, other_vaccine_terms) and not has_flu_vaccine:
            return "exclude", "other_vaccine_only"
        if contains(text, general_health_terms) and not has_flu_vaccine:
            return "exclude", "general_health_only"

    if contains(text, explicit_flu_vaccine):
        return "include", "explicit_influenza_vaccine"
    if action_bound.search(text):
        return "include", "influenza_vaccine_action"
    if any(pattern.search(text) for pattern in decision_patterns):
        return "include", "vaccination_decision"
    if contains(text, brand_terms) and contains(text, vaccination_context):
        return "include", "brand_with_vaccination_context"

    noise_terms = (
        animal_terms + ecommerce_terms + stock_terms + hib_terms + disease_terms
        + other_vaccine_terms + general_health_terms
    )
    if contains(text, flu_terms) and ("疫苗" in text or "打针" in text):
        return "llm_review", "weak_influenza_vaccine_link"
    if contains(text, weak_product_terms) and not contains(text, explicit_flu_vaccine):
        return "llm_review", "weak_product_form"
    if contains(text, brand_terms) and not contains(text, vaccination_context):
        return "llm_review", "brand_only"
    if contains(text, notice_terms) and not has_flu_vaccine:
        return "llm_review", "notice_or_news"
    if contains(text, context_dependent_terms):
        return "llm_review", "context_dependent"
    if (contains(text, flu_terms) or contains(text, vaccine_terms)
            or contains(text, brand_terms)) and contains(text, noise_terms):
        return "llm_review", "mixed_signal"
    return "llm_review", "other"


data = pd.read_csv("structure_clean_records.csv")
data[["relevance_status", "relevance_reason"]] = [
    classify_relevance(title, body)
    for title, body in zip(data["标题"], data["正文"])
]

rule_relevant = data[data["relevance_status"] == "include"].copy()
rule_irrelevant = data[data["relevance_status"] == "exclude"].copy()
relevance_llm_review = data[data["relevance_status"] == "llm_review"].copy()
