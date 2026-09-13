# Original Chinese Search Strategy

This file provides the original Chinese-language search terms used for data retrieval in the study **“LLM-Enabled Digital Analysis of Influenza Vaccine Hesitancy: Decoupling Delay and Refusal via Semantic Attribution.”**

The English translations of these search terms are reported in Supplementary Table S1. The Chinese terms below are reproduced verbatim to preserve the actual retrieval strategy and support reproducibility. The search terms were not retrospectively expanded after data collection.

## 1. Direct influenza vaccine terms

```text
流感疫苗 OR 流感预防针 OR 流感针 OR 流感病毒裂解疫苗 OR 流感病毒亚单位疫苗 OR 流感病毒减毒活疫苗 OR 鼻喷流感
```

## 2. Brand or product terms

```text
安尔来福 OR 凡尔灵 OR 凡尔佳 OR 迪福赛尔 OR 慧尔康欣 OR 感雾 OR 适普利尔 OR 孚洛克 OR 御感宁 OR 御菲宁 OR 英扶宁 OR 安定伏
```

## 3. Influenza context terms

```text
流感 OR 甲流 OR 乙流
```

## 4. Manufacturer and valency terms

```text
科兴四价 OR 科兴三价 OR 华兰三价 OR 华兰四价 OR 长春所三价 OR 长春所四价 OR 长春生物四价 OR 长春生物三价 OR 上海所三价 OR 上海所四价 OR 上海生物三价 OR 上海生物四价 OR 武汉所三价 OR 武汉所四价 OR 武汉生物三价 OR 武汉生物四价 OR 雅立峰三价 OR 雅立峰四价 OR 复星三价 OR 复星四价 OR 智飞四价 OR 赛诺菲三价 OR 赛诺菲四价
```

## 5. General vaccination terms

```text
疫苗 OR 预防针 OR 预防接种 OR 免疫预防 OR 主动免疫
```

## 6. Combined contextual search logic

```text
Influenza context terms AND (Manufacturer and valency terms OR General vaccination terms)
```

This component was used to capture influenza-vaccination-related records when direct expressions such as “流感疫苗” were not used.

## 7. Exclusion terms: animal or avian influenza

```text
禽流感 OR 鸡流感 OR 鸭流感 OR 鹅流感 OR 猪流感 OR 犬流感 OR 猫流感 OR 马流感 OR H5N1 OR H5N6 OR H5N8 OR H7N9 OR H9N2 OR 猫瘟 OR 鸡瘟 OR 犬瘟 OR 猪瘟 OR 宠物疫苗 OR 动物疫病 OR 养殖场 OR 养殖 OR 家禽 OR 鸭场 OR 猪场 OR 鸡场 OR 仔猪 OR 畜牧 OR 畜禽 OR 兽医 OR 兽用 OR 猫舍 OR 犬舍 OR 检疫 OR 屠宰
```

## 8. Exclusion terms: e-commerce and marketing

```text
秒杀 OR 抢购 OR 团购 OR 代购 OR 拼单 OR 降价 OR 特价 OR 促销 OR 优惠 OR 折扣 OR 下单 OR 购买 OR 售价 OR 到手价 OR 领券 OR 优惠券 OR 满减 OR 拼团 OR 预售 OR 现货 OR 包邮 OR 发货 OR 物流 OR 链接 OR 详情页 OR 客服 OR 售后 OR 正品 OR 保真 OR 假一赔十 OR 直播 OR 带货 OR 种草 OR 安利 OR 推广 OR 广告 OR 软文 OR 合作 OR 返利 OR 橱窗 OR 淘宝 OR 天猫 OR 京东 OR 拼多多 OR 抖店 OR 快手小店 OR 唯品会 OR 苏宁 OR 得物 OR 闲鱼 OR 旗舰店 OR 专卖店 OR 店铺 OR 招商 OR 加盟 OR 代理 OR 渠道 OR 供应商
```

## 9. Exclusion terms: cosmetic or beauty-related terms

```text
雾眉 OR 纹眉 OR 定妆 OR 持妆 OR 脱妆 OR 底妆 OR 哑光 OR 雾感 OR 妆感 OR 唇釉 OR 显白 OR 质感 OR 化妆 OR 护肤 OR 精华 OR 抗衰 OR 医美 OR 肉毒 OR 水光针 OR 热玛吉 OR 瘦脸针 OR 超声刀
```

## 10. Exclusion terms: insurance, finance, and investment

```text
保险 OR 理赔 OR 保额 OR 投资 OR 交易 OR 股票 OR 市值 OR 涨停 OR 跌停 OR 美股 OR 港股
```

## 11. Final Boolean logic

```text
(Direct influenza vaccine terms OR Brand or product terms OR Combined contextual search logic)
AND NOT Animal or avian influenza exclusions
AND NOT E-commerce and marketing exclusions
AND NOT Cosmetic or beauty-related exclusions
AND NOT Insurance, finance, and investment exclusions
```

## Notes

- Exclusion terms were applied at retrieval to reduce irrelevant records; final eligibility was determined through subsequent cleaning and screening procedures.
- Manufacturer and brand term coverage was cross-checked against official influenza-vaccine supply and batch-release records for 2016–2025.
- Of 15 manufacturers identified, 13 were represented by either manufacturer/abbreviation terms or included brand terms; Jiangsu Jindike and Changchun Changsheng were not represented.
- The search terms shown above reflect the strategy actually used and were not retrospectively expanded.
