# content-studio

内容创作工坊 - 10个内容创作与SEO优化工具集

## 功能概览

- **博文大纲生成** - 根据主题和关键词生成博文结构
- **SEO关键词分析** - TF-IDF算法提取关键词
- **社媒文案生成** - 多平台多语气文案生成
- **标签生成器** - 自动生成社交媒体标签
- **内容日历** - 按频率生成发布日历
- **标题评分器** - 多维度评估标题吸引力
- **文章摘要** - TextRank算法自动摘要
- **文本查重** - Jaccard+余弦双算法相似度检测
- **Meta描述生成** - SEO优化Meta描述
- **邮件模板渲染** - 变量替换邮件模板

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from main import content_summarizer

result = content_summarizer("长文本内容...", sentences_count=3)
print(result["summary"])
```

## License

MIT
