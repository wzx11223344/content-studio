# content-studio

内容创作工坊 - 10个高级算法驱动的内容创作与SEO优化工具集

## 功能概览

| # | 函数名 | 算法原理 | 复杂度 |
|---|--------|----------|--------|
| 1 | `textrank_summary` | TextRank：句子分割→TF-IDF相似度矩阵→PageRank迭代→Top-K | O(n^2 + n^2*T) |
| 2 | `tfidf_keyword_extraction` | TF-IDF：双字组/三字组分词→IDF计算→关键词排序 | O(n*m) |
| 3 | `sentiment_analyzer` | 词典法+否定词处理+程度副词加权+转折词处理 | O(n)，n=文本长度 |
| 4 | `seo_score_analyzer` | 10维度SEO评分模型（关键词密度/标题/可读性等） | O(n) |
| 5 | `readability_calculator` | 中文版Flesch-Kincaid公式（平均句长+词长+虚词比例） | O(n) |
| 6 | `content_duplicate_checker` | N-gram指纹+Jaccard相似度 | O(n*k)，k=n-gram大小 |
| 7 | `social_media_caption_generator` | 模板引擎+话题标签推荐+表情匹配+字数限制 | O(1)模板匹配 |
| 8 | `topic_cluster_analysis` | 简化版LDA：文档-词频矩阵→Gibbs采样→话题分布 | O(n*V*K*T) |
| 9 | `title_scorer` | 多维度标题评分（数字/疑问/情感/长度/关键词） | O(n) |
| 10 | `content_calendar_planner` | 话题优先级排序+频率分配+时间槽调度 | O(n log n) |

## 算法详解

### 1. TextRank自动摘要算法 (`textrank_summary`)
- **句子分割**: 基于中文标点（。！？；）和英文标点(.!?)进行句子边界检测
- **TF-IDF相似度矩阵**: 对每对句子计算TF-IDF加权的余弦相似度
- **PageRank迭代**: `WS(Vi) = (1-d)/N + d * Σ(S_ij * WS(Vj) / Σ_k S_kj)`，d=0.85
- **收敛条件**: 迭代直到变化量 < 1e-6 或达到最大迭代次数
- **Top-K提取**: 按最终PageRank得分排序，选前K句作为摘要
- **复杂度**: 相似度矩阵 O(n^2)，PageRank迭代 O(n^2*T)

### 2. TF-IDF关键词提取 (`tfidf_keyword_extraction`)
- **分词策略**: 基于统计的双字组(bigram)+三字组(trigram)提取
- **TF计算**: `tf(t) = count(t) / total_terms`
- **IDF计算**: `idf(t) = log(N / (1 + df(t)))`，平滑处理避免除零
- **排序**: 按 TF-IDF 得降序排列，取Top-K
- **复杂度**: O(n*m)，n=文档数，m=词数

### 3. 情感分析器 (`sentiment_analyzer`)
- **词典法**: 基于预定义的正向/负向情感词典
- **否定词处理**: 检测"不"、"没"、"无"等否定词，反转后续词的情感极性
- **程度副词加权**: "非常"(×2.0)、"比较"(×1.5)、"稍微"(×0.5)等程度修饰
- **转折词处理**: "但是"、"然而"等转折词后，权重重新分配
- **输出**: 正向得分、负向得分、中性得分、整体情感倾向

### 4. SEO评分模型 (`seo_score_analyzer`)
- **10个评分维度**:
  1. 关键词密度（2-5%为最佳）
  2. 标题标签（H1唯一性、关键词位置）
  3. 元描述长度（120-160字符）
  4. 内链密度
  5. 可读性评分
  6. H标签结构层次
  7. 图片alt属性覆盖
  8. 段落长度分布
  9. 关键词在首段出现
  10. 内容长度评估
- **评分**: 每维度0-10分，总分0-100，附带优化建议

### 5. 可读性评估 (`readability_calculator`)
- **中文版Flesch-Kincaid公式**: 
  - 平均句长(ASL) = 字符数 / 句子数
  - 平均词长(ASW) = 字符数 / 词数
  - 虚词比例 = 虚词数 / 总词数
  - `可读性 = 100 - ASL*1.5 - ASW*10 - 虚词比例*5`
- **难度等级**: 0-30困难，30-60中等，60-100简单

### 6. 内容查重器 (`content_duplicate_checker`)
- **N-gram指纹**: 将文本切分为N-gram（默认N=5），生成指纹集合
- **Jaccard相似度**: `J = |F1 ∩ F2| / |F1 ∪ F2|`
- **重复定位**: 滑动窗口检测重复段落位置
- **复杂度**: O(n*k)，k=N-gram大小

### 7. 社媒文案生成器 (`social_media_caption_generator`)
- **模板引擎**: 基于平台(微信/微博/小红书/抖音)和语气(专业/活泼/正式)的模板选择
- **话题标签推荐**: 基于关键词的热门标签匹配
- **表情符号匹配**: 根据语气自动匹配表情符号
- **字数限制**: 平台限制自动截断（微博140/小红书1000等）

### 8. LDA话题聚类分析 (`topic_cluster_analysis`)
- **文档-词频矩阵**: 构建D×V矩阵（D=文档数，V=词表大小）
- **Gibbs采样**:
  - 初始化: 随机分配每个词的话题
  - 迭代: `P(z_i=k | z_-i, w) ∝ (n_k^w + β) / (n_k + V*β) * (n_d^k + α) / (n_d + K*α)`
  - 收敛: 达到最大迭代次数或分布稳定
- **话题分布**: 每个文档的话题概率分布 θ，每个话题的词分布 φ
- **复杂度**: O(n*V*K*T)，T=迭代次数

### 9. 标题评分器 (`title_scorer`)
- **评分维度**:
  1. 数字开头加分（"5个技巧..."）
  2. 疑问句加分（"如何..."、"为什么..."）
  3. 情感词加分（"震惊"、"必看"等）
  4. 长度评估（15-30字为最佳）
  5. 关键词覆盖率（目标关键词出现次数）
- **输出**: 0-100分 + 等级(A/B/C/D) + 优化建议

### 10. 内容日历规划器 (`content_calendar_planner`)
- **话题优先级排序**: 基于热度、时效性、相关性的综合评分
- **频率分配**: 根据发布频率(每日/每周/每月)分配日期
- **时间槽调度**: 根据最佳发布时间（早9点/午12点/晚8点）分配时间槽
- **复杂度**: 排序O(n log n)，分配O(n)

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

## License

MIT
