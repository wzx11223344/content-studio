---
name: content-studio-zx
displayName: 内容创作工坊
version: 2.0.0
summary: 10个高级算法内容工具：TextRank摘要/TF-IDF关键词/词典情感分析/10维度SEO评分/中文Flesch可读性/N-gram查重/社媒文案/简化LDA话题聚类/标题评分/内容日历规划
tags: [content, seo, textrank, tfidf, lda, sentiment-analysis, social-media]
license: MIT
---

# 内容创作工坊 (content-studio)

## 简介

content-studio 是一套包含10个高级算法驱动的内容创作与SEO优化工具的 Python 技能包。所有功能仅使用Python标准库实现，无需安装任何外部依赖。涵盖TextRank自动摘要、TF-IDF关键词提取、情感分析、SEO评分、可读性评估、内容查重、社媒文案、LDA话题聚类、标题评分、内容日历规划等场景。

## 功能列表

| # | 函数名 | 算法原理 | 复杂度 |
|---|--------|----------|--------|
| 1 | `textrank_summary` | TextRank：句子分割→TF-IDF相似度矩阵→PageRank迭代→Top-K | O(n^2 + n^2*T) |
| 2 | `tfidf_keyword_extraction` | TF-IDF：双字组/三字组分词→IDF计算→关键词排序 | O(n*m) |
| 3 | `sentiment_analyzer` | 词典法+否定词处理+程度副词加权+转折词处理 | O(n) |
| 4 | `seo_score_analyzer` | 10维度SEO评分模型 | O(n) |
| 5 | `readability_calculator` | 中文版Flesch-Kincaid公式 | O(n) |
| 6 | `content_duplicate_checker` | N-gram指纹+Jaccard相似度 | O(n*k) |
| 7 | `social_media_caption_generator` | 模板引擎+话题标签+表情匹配+字数限制 | O(1) |
| 8 | `topic_cluster_analysis` | 简化LDA：文档-词频矩阵→Gibbs采样→话题分布 | O(n*V*K*T) |
| 9 | `title_scorer` | 多维度评分（数字/疑问/情感/长度/关键词） | O(n) |
| 10 | `content_calendar_planner` | 话题优先级排序+频率分配+时间槽调度 | O(n log n) |

## 安装

无需安装额外依赖，仅使用Python标准库（re, math, collections, random, datetime等）。

## 使用示例

```python
from main import textrank_summary, sentiment_analyzer, seo_score_analyzer

# TextRank自动摘要
text = "长文本内容..."
summary = textrank_summary(text, num_sentences=3)
print(summary["summary"])

# 情感分析
result = sentiment_analyzer("这个产品非常好，但价格有点贵")
print(f"正向: {result['positive']}, 负向: {result['negative']}")

# SEO评分
score = seo_score_analyzer(content, target_keyword="Python教程")
print(f"SEO得分: {score['total_score']}/100")
```

## 依赖

无外部依赖（仅使用Python标准库）

## License

MIT
