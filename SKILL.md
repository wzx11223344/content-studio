---
name: content-studio-zx
displayName: 内容创作工坊
version: 1.0.1
summary: 10个内容工具：博文大纲/SEO分析/社媒文案/标签生成/内容日历/标题评分/文章摘要/查重/Meta描述/邮件模板
tags: [content, seo, social-media, writing]
license: MIT
---

# 内容创作工坊 (content-studio)

## 简介

content-studio 是一套包含10个内容创作与SEO优化工具的 Python 技能包，覆盖博文写作、SEO分析、社交媒体运营、内容日历管理等场景。

## 功能列表

| # | 函数名 | 功能描述 |
|---|--------|----------|
| 1 | `generate_blog_outline` | 博文大纲生成 |
| 2 | `seo_keyword_analyzer` | SEO关键词分析（TF-IDF） |
| 3 | `social_media_caption` | 社媒文案生成 |
| 4 | `hashtag_generator` | 标签生成器 |
| 5 | `content_calendar_generator` | 内容日历生成 |
| 6 | `headline_analyzer` | 标题评分器 |
| 7 | `content_summarizer` | 文章摘要（TextRank） |
| 8 | `plagiarism_checker` | 文本相似度检测 |
| 9 | `meta_description_generator` | SEO Meta描述生成 |
| 10 | `email_template_renderer` | 邮件模板渲染 |

## 安装

```bash
pip install jieba scikit-learn numpy
```

## 使用示例

```python
from main import headline_analyzer, hashtag_generator

# 标题评分
result = headline_analyzer("震惊！2025年最全Python教程")
print(f"得分: {result['total_score']}/100 ({result['grade']})")

# 标签生成
tags = hashtag_generator("编程", count=10)
print(tags["hashtags"])
```

## 依赖

- `jieba`: 中文分词
- `scikit-learn`: TF-IDF计算
- `numpy`: 数值计算

## License

MIT
