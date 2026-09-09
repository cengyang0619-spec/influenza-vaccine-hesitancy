import re

import pandas as pd


institution_terms = [
    "医院", "卫生院", "妇幼", "疾控", "cdc", "接种门诊", "社区卫生服务中心",
    "卫生服务站", "门诊部", "保健院", "卫生健康", "卫健", "医联体", "医共体"
]
school_terms = ["幼儿园", "小学", "中学", "学校", "教育局", "教育集团", "校园", "托育"]
media_terms = ["新闻", "日报", "晚报", "广播", "电视台", "融媒体", "传媒", "报社", "记者", "资讯"]
professional_terms = [
    "医生", "医师", "药师", "护士", "健康科普", "疫苗科普", "健康讲堂",
    "营养师", "健康号"
]
commercial_terms = ["大药房", "药店", "药房", "门店", "旗舰店", "生物", "医药", "药业", "商贸", "商城"]
nonpublic_terms = ["机构", "官方", "政府", "媒体", "学校", "医疗机构", "企业", "商家"]

notice_terms = [
    "通知", "公告", "接种安排", "门诊安排", "接种时间", "接种地点", "接种对象",
    "接种费用", "预约方式", "预约电话", "请携带", "请居民", "请家长",
    "具体安排如下", "免费接种"
]
news_terms = [
    "本报讯", "记者从", "通讯员", "据悉", "获悉", "发布会", "通报", "原标题",
    "本文转载", "来源：", "专家表示"
]
science_terms = [
    "健康科普", "健康提示", "科普课堂", "知识讲堂", "医生提醒", "专家建议",
    "健康宣教", "接种禁忌", "适用人群", "注意事项", "常见问题"
]
marketing_terms = [
    "优惠", "团购", "套餐", "促销", "限时", "私信咨询", "点击咨询", "扫码咨询",
    "咨询热线", "欢迎到店", "名额有限", "代预约"
]

personal_decision = [
    re.compile(r"我.{0,15}(打了|打完|去打|准备打|预约|约了|不打|犹豫|担心|后悔)"),
    re.compile(r"我家.{0,12}(孩子|宝宝|家人|老人).{0,12}(打|接种|预约|不打)"),
    re.compile(r"(带孩子去打|给孩子打|陪家人打|我们家.{0,12}(接种|不打|预约))"),
]


def normalize(value):
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text.replace("\ufeff", "").replace("\u200b", " ")).strip().lower()


def contains(text, terms):
    return any(term in text for term in terms)


def classify_public_post(row):
    content = " ".join(normalize(row[column]) for column in ["标题", "正文", "摘要"])
    author_context = " ".join(
        normalize(row[column])
        for column in ["作者名称", "来源网站", "媒体类型", "发布者性质"]
    )

    if any(pattern.search(content) for pattern in personal_decision):
        return "llm_review", ""

    institution = contains(author_context, institution_terms)
    school = contains(author_context, school_terms)
    media = contains(author_context, media_terms)
    professional = contains(author_context, professional_terms)
    commercial = contains(author_context, commercial_terms)
    nonpublic = contains(normalize(row["发布者性质"]), nonpublic_terms)

    notice = contains(content, notice_terms)
    news = contains(content, news_terms)
    science = contains(content, science_terms)
    marketing = contains(content, marketing_terms)

    if commercial and marketing:
        return "exclude", "commercial_promotion"
    if institution and notice:
        return "exclude", "institutional_notice"
    if school and notice:
        return "exclude", "school_notice"
    if media and news:
        return "exclude", "news_or_media_content"
    if professional and science:
        return "exclude", "professional_health_education"
    if nonpublic and notice:
        return "exclude", "nonpublic_service_notice"
    if nonpublic and news:
        return "exclude", "nonpublic_news_content"
    if nonpublic and marketing:
        return "exclude", "nonpublic_commercial_content"
    return "llm_review", ""


data = pd.read_csv("relevance_screened_records.csv")
data[["public_post_status", "public_post_reason"]] = [
    classify_public_post(row) for _, row in data.iterrows()
]

rule_nonpublic = data[data["public_post_status"] == "exclude"].copy()
public_post_llm_review = data[data["public_post_status"] == "llm_review"].copy()
