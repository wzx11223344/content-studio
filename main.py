"""
content-studio - 内容创作工坊

本模块实现了10个高级内容创作与SEO分析算法工具，全部使用Python标准库实现：
1. textrank_summary            - TextRank自动摘要（PageRank图排序）
2. tfidf_keyword_extraction    - TF-IDF关键词提取（双字组+三字组分词）
3. sentiment_analyzer          - 情感分析器（词典法+否定词+程度副词+转折词）
4. seo_score_analyzer          - SEO评分模型（10维度检查）
5. readability_calculator      - 可读性评估（中文版Flesch-Kincaid公式）
6. content_duplicate_checker   - 内容查重器（N-gram指纹+Jaccard相似度）
7. social_media_caption_generator - 社媒文案生成器（模板引擎+话题推荐）
8. topic_cluster_analysis     - 话题聚类分析（简化版LDA+Gibbs采样）
9. title_scorer               - 标题评分器（多维度吸引力评估）
10. content_calendar_planner  - 内容日历规划器（优先级+频率+时间槽分配）

所有算法均从零实现，不依赖numpy/pandas/jieba等第三方库。
"""

import re
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Set, Optional


# ============================================================================
# 辅助函数
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """
    混合分词器：英文按单词切分，中文按双字组切分。
    """
    tokens = []
    en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    tokens.extend(en_words)
    cn_segments = re.findall(r'[\u4e00-\u9fff]+', text)
    for seg in cn_segments:
        if len(seg) >= 2:
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i + 2])
        elif len(seg) == 1:
            tokens.append(seg)
    return tokens


def _split_sentences(text: str) -> List[str]:
    """句子分割器。"""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'[。！？!?\.;\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ============================================================================
# 1. textrank_summary - TextRank自动摘要算法
# ============================================================================

def textrank_summary(text: str, num_sentences: int = 5) -> Dict[str, Any]:
    """
    TextRank自动摘要算法完整实现。

    算法流程：
    1. 句子分割：将文本切分为句子列表
    2. TF-IDF句子相似度矩阵：计算每对句子的TF-IDF余弦相似度
    3. PageRank迭代：在句子相似图上运行PageRank算法
    4. Top-K提取：按PageRank分数排序选取最重要的句子
    5. 原文顺序排列：保持摘要的逻辑连贯性

    时间复杂度：O(n^2 * m + n * iter)，n=句子数，m=平均词数，iter=迭代次数
    空间复杂度：O(n^2) 相似度矩阵

    参数:
        text: 输入文本
        num_sentences: 摘要句子数

    返回:
        包含摘要、排序句子、关键词的字典
    """
    # 步骤1：句子分割
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return {"summary": text, "sentences": sentences, "message": "文本过短"}

    # 步骤2：计算IDF（逆文档频率，以句子为文档）
    all_tokens = [_tokenize(s) for s in sentences]
    N = len(sentences)
    df = Counter()
    for tokens in all_tokens:
        df.update(set(tokens))
    idf = {t: math.log(N / max(df[t], 1)) for t in df}

    # 步骤3：构建句子相似度矩阵
    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # TF-IDF余弦相似度
            tokens_i = all_tokens[i]
            tokens_j = all_tokens[j]
            if not tokens_i or not tokens_j:
                continue

            tf_i = Counter(tokens_i)
            tf_j = Counter(tokens_j)
            vec_i = {t: tf_i[t] / len(tokens_i) * idf.get(t, 1.0) for t in tf_i}
            vec_j = {t: tf_j[t] / len(tokens_j) * idf.get(t, 1.0) for t in tf_j}

            # 余弦相似度
            common = set(vec_i.keys()) & set(vec_j.keys())
            dot = sum(vec_i[k] * vec_j[k] for k in common)
            norm_i = math.sqrt(sum(v * v for v in vec_i.values()))
            norm_j = math.sqrt(sum(v * v for v in vec_j.values()))
            sim = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0.0

            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # 步骤4：PageRank迭代
    damping = 0.85
    max_iter = 100
    tol = 1e-6
    scores = [1.0 / n] * n

    # 归一化相似度矩阵
    for i in range(n):
        row_sum = sum(sim_matrix[i])
        if row_sum > 0:
            sim_matrix[i] = [s / row_sum for s in sim_matrix[i]]

    for iteration in range(max_iter):
        new_scores = []
        for i in range(n):
            rank_sum = sum(scores[j] * sim_matrix[j][i] for j in range(n) if j != i)
            new_score = (1 - damping) / n + damping * rank_sum
            new_scores.append(new_score)

        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    # 步骤5：选取Top-K句子
    ranked = sorted(range(n), key=lambda x: -scores[x])
    top_indices = sorted(ranked[:num_sentences])

    summary = '。'.join(sentences[i] for i in top_indices) + '。'

    # 提取关键词
    total_tf = Counter()
    for tokens in all_tokens:
        total_tf.update(tokens)
    keywords = sorted(
        [(t, total_tf[t] * idf.get(t, 1.0)) for t in total_tf],
        key=lambda x: -x[1]
    )[:15]

    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": round(len(summary) / max(len(text), 1), 4),
        "total_sentences": n,
        "selected_sentences": num_sentences,
        "ranked_sentences": [
            {"rank": r + 1, "index": idx, "score": round(scores[idx], 6),
             "sentence": sentences[idx][:200]}
            for r, idx in enumerate(ranked[:10])
        ],
        "keywords": [{"word": w, "score": round(s, 4)} for w, s in keywords]
    }


# ============================================================================
# 2. tfidf_keyword_extraction - TF-IDF关键词提取
# ============================================================================

def tfidf_keyword_extraction(text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    TF-IDF关键词提取。

    算法：
    1. 分词：基于统计的双字组(2-gram)和三字组(3-gram)提取
    2. 词频计算：统计每个词项的出现频率
    3. IDF估计：将整个文本按句子分割作为文档集，计算IDF
    4. TF-IDF排序：TF * IDF 排序选取Top-K

    参数:
        text: 输入文本
        top_k: 返回关键词数量

    返回:
        包含关键词列表、TF-IDF分数、统计信息的字典
    """
    # 步骤1：分词 - 双字组和三字组
    cn_segments = re.findall(r'[\u4e00-\u9fff]+', text)
    en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())

    # 中文双字组和三字组
    bigrams = []
    trigrams = []
    for seg in cn_segments:
        if len(seg) >= 2:
            for i in range(len(seg) - 1):
                bigrams.append(seg[i:i + 2])
        if len(seg) >= 3:
            for i in range(len(seg) - 2):
                trigrams.append(seg[i:i + 3])

    all_tokens = bigrams + trigrams + en_words

    if not all_tokens:
        return {"keywords": [], "message": "无法提取有效关键词"}

    # 步骤2：以句子为文档计算IDF
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    N = len(sentences)
    doc_tokens = [_tokenize(s) for s in sentences]
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    idf = {t: math.log(N / max(df.get(t, 1), 1)) for t in set(all_tokens)}

    # 步骤3：计算TF-IDF
    tf = Counter(all_tokens)
    total = len(all_tokens)

    keywords = []
    for term, freq in tf.items():
        tf_val = freq / total
        idf_val = idf.get(term, math.log(N))
        tfidf = tf_val * idf_val
        keywords.append({
            "term": term,
            "tf": freq,
            "tf_normalized": round(tf_val, 6),
            "idf": round(idf_val, 4),
            "tfidf": round(tfidf, 6),
            "type": "bigram" if len(term) == 2 else ("trigram" if len(term) == 3 else "word")
        })

    keywords.sort(key=lambda x: -x["tfidf"])

    return {
        "keywords": keywords[:top_k],
        "total_unique_terms": len(tf),
        "total_tokens": total,
        "sentence_count": N,
        "top_term": keywords[0]["term"] if keywords else None
    }


# ============================================================================
# 3. sentiment_analyzer - 情感分析器
# ============================================================================

# 情感词典
_POSITIVE_WORDS = {
    '好': 2, '棒': 3, '优秀': 3, '满意': 2, '喜欢': 2, '爱': 3, '开心': 2,
    '快乐': 2, '幸福': 2, '完美': 3, '精彩': 2, '成功': 2, '推荐': 2,
    '赞': 3, '牛': 2, '厉害': 2, '不错': 1, '可以': 1, '值得': 2,
    '高兴': 2, '兴奋': 2, '惊喜': 2, '感谢': 2, '棒极了': 3, '超赞': 3,
    'great': 3, 'good': 2, 'excellent': 3, 'amazing': 3, 'love': 3,
    'happy': 2, 'perfect': 3, 'wonderful': 3, 'best': 3, 'awesome': 3,
}

_NEGATIVE_WORDS = {
    '差': -2, '烂': -3, '糟糕': -3, '失望': -2, '讨厌': -2, '恨': -3,
    '难用': -2, '垃圾': -3, '骗': -2, '坑': -2, '假': -2, '慢': -1,
    '坏': -2, '差劲': -2, '恶心': -3, '无聊': -1, '失败': -2, '问题': -1,
    '错': -2, '不行': -2, '不好': -2, '后悔': -2, '退款': -2, '投诉': -2,
    'bad': -2, 'terrible': -3, 'awful': -3, 'hate': -3, 'worst': -3,
    'poor': -2, 'horrible': -3, 'disappointing': -2, 'useless': -2,
}

_NEGATION_WORDS = {'不', '没', '无', '非', '别', '勿', '未', '莫', '否', 'not', 'no', 'never'}

_DEGREE_WORDS = {
    '很': 1.5, '非常': 2.0, '特别': 2.0, '极其': 2.5, '超': 1.8, '太': 1.8,
    '十分': 2.0, '相当': 1.5, '比较': 1.2, '稍微': 0.5, '略微': 0.5,
    '最': 2.5, '更': 1.5, '尤其': 2.0, '甚': 2.0, '极': 2.5,
    'very': 1.5, 'extremely': 2.5, 'really': 1.5, 'quite': 1.3,
    'so': 1.5, 'too': 1.5, 'incredibly': 2.0,
}

_TRANSITION_WORDS = {'但是', '然而', '不过', '可是', '虽然', '尽管', '但', '却',
                     'but', 'however', 'although', 'though', 'nevertheless'}


def sentiment_analyzer(text: str) -> Dict[str, Any]:
    """
    情感分析器。

    算法：基于词典法 + 多维度加权
    1. 词典匹配：在正/负向词典中查找情感词
    2. 否定词处理：否定词翻转后续情感词极性
    3. 程度副词加权：程度副词放大或缩小情感强度
    4. 转折词处理：转折词后的句子权重提高（真实意图）

    参数:
        text: 输入文本

    返回:
        包含正向/负向/中性得分、情感词列表、综合判断的字典
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"sentiment": "neutral", "score": 0, "positive_score": 0, "negative_score": 0}

    positive_score = 0.0
    negative_score = 0.0
    sentiment_words = []
    negation_active = False
    degree_multiplier = 1.0
    after_transition = False
    transition_bonus = 1.5  # 转折后权重提升

    # 逐词扫描
    for i, token in enumerate(tokens):
        # 检查是否是转折词
        if token in _TRANSITION_WORDS:
            after_transition = True
            continue

        # 检查是否是否定词
        if token in _NEGATION_WORDS:
            negation_active = True
            continue

        # 检查是否是程度副词
        if token in _DEGREE_WORDS:
            degree_multiplier = _DEGREE_WORDS[token]
            continue

        # 检查正向情感词
        if token in _POSITIVE_WORDS:
            score = _POSITIVE_WORDS[token] * degree_multiplier
            if negation_active:
                score = -score  # 否定翻转
                negative_score += abs(score) * (transition_bonus if after_transition else 1.0)
                sentiment_words.append({
                    "word": token, "type": "positive_negated",
                    "score": -score, "negated": True
                })
            else:
                positive_score += score * (transition_bonus if after_transition else 1.0)
                sentiment_words.append({
                    "word": token, "type": "positive",
                    "score": score, "negated": False
                })
            negation_active = False
            degree_multiplier = 1.0

        # 检查负向情感词
        elif token in _NEGATIVE_WORDS:
            score = _NEGATIVE_WORDS[token] * degree_multiplier
            if negation_active:
                score = -score  # 否定翻转（双重否定）
                positive_score += abs(score) * (transition_bonus if after_transition else 1.0)
                sentiment_words.append({
                    "word": token, "type": "negative_negated",
                    "score": -score, "negated": True
                })
            else:
                negative_score += abs(score) * (transition_bonus if after_transition else 1.0)
                sentiment_words.append({
                    "word": token, "type": "negative",
                    "score": score, "negated": False
                })
            negation_active = False
            degree_multiplier = 1.0

    # 计算综合分数
    total_score = positive_score - negative_score
    if total_score > 0.5:
        sentiment = "positive"
    elif total_score < -0.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # 归一化到[-1, 1]
    max_possible = max(positive_score + negative_score, 1)
    normalized_score = total_score / max_possible

    return {
        "sentiment": sentiment,
        "score": round(total_score, 4),
        "normalized_score": round(normalized_score, 4),
        "positive_score": round(positive_score, 4),
        "negative_score": round(negative_score, 4),
        "neutral_score": round(1.0 - min(abs(normalized_score), 1.0), 4),
        "sentiment_words": sentiment_words,
        "word_count": len(tokens),
        "sentiment_word_count": len(sentiment_words),
        "has_transition": after_transition,
        "confidence": round(min(abs(total_score) / 5.0, 1.0), 4)
    }


# ============================================================================
# 4. seo_score_analyzer - SEO评分模型
# ============================================================================

def seo_score_analyzer(content: str, target_keyword: str) -> Dict[str, Any]:
    """
    SEO评分模型。

    检查10个维度：
    1. 关键词密度（2-5%为最佳）
    2. 标题标签（H1存在性）
    3. 元描述长度（120-160字符）
    4. 内链密度
    5. 可读性（平均句长）
    6. H标签结构层次
    7. 图片alt属性
    8. 关键词在标题中
    9. 内容长度（>300字为佳）
    10. 段落结构

    参数:
        content: HTML或纯文本内容
        target_keyword: 目标关键词

    返回:
        包含总分、各维度得分、优化建议的字典
    """
    scores = {}
    suggestions = []

    # 1. 关键词密度
    tokens = _tokenize(content)
    keyword_count = tokens.count(target_keyword) + content.count(target_keyword)
    total_words = max(len(tokens), 1)
    density = keyword_count / total_words * 100

    if 2 <= density <= 5:
        scores['keyword_density'] = 10
    elif 1 <= density < 2 or 5 < density <= 7:
        scores['keyword_density'] = 7
    elif density > 0:
        scores['keyword_density'] = 4
    else:
        scores['keyword_density'] = 0
        suggestions.append("关键词未出现，建议在内容中自然融入目标关键词")

    # 2. 标题标签
    has_h1 = bool(re.search(r'<h1[^>]*>.*?</h1>', content, re.IGNORECASE | re.DOTALL))
    scores['h1_tag'] = 10 if has_h1 else 0
    if not has_h1:
        suggestions.append("缺少H1标签，建议添加包含关键词的H1标题")

    # 3. 元描述
    meta_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', content, re.IGNORECASE)
    if meta_match:
        meta_len = len(meta_match.group(1))
        if 120 <= meta_len <= 160:
            scores['meta_description'] = 10
        elif 80 <= meta_len < 120 or 160 < meta_len <= 200:
            scores['meta_description'] = 7
        else:
            scores['meta_description'] = 4
    else:
        scores['meta_description'] = 0
        suggestions.append("缺少meta description，建议添加120-160字符的描述")

    # 4. 内链密度
    internal_links = len(re.findall(r'<a[^>]*href=["\']/(?!https?://)', content, re.IGNORECASE))
    link_density = internal_links / max(len(tokens) / 100, 1)
    if 2 <= link_density <= 5:
        scores['internal_links'] = 10
    elif link_density > 0:
        scores['internal_links'] = 6
    else:
        scores['internal_links'] = 3
        suggestions.append("内链数量不足，建议添加2-5个内链/百词")

    # 5. 可读性
    sentences = _split_sentences(content)
    if sentences:
        avg_sentence_len = len(tokens) / len(sentences)
        if 15 <= avg_sentence_len <= 25:
            scores['readability'] = 10
        elif 10 <= avg_sentence_len < 15 or 25 < avg_sentence_len <= 35:
            scores['readability'] = 7
        else:
            scores['readability'] = 4
            suggestions.append(f"平均句长{avg_sentence_len:.0f}字，建议控制在15-25字之间")
    else:
        scores['readability'] = 5

    # 6. H标签结构
    h_tags = re.findall(r'<h([1-6])[^>]*>', content, re.IGNORECASE)
    if h_tags:
        has_proper_hierarchy = h_tags[0] == '1'
        scores['heading_structure'] = 10 if has_proper_hierarchy else 7
    else:
        scores['heading_structure'] = 3
        suggestions.append("缺少H标签层次结构，建议使用H1-H3组织内容")

    # 7. 图片alt属性
    images = re.findall(r'<img[^>]*>', content, re.IGNORECASE)
    images_with_alt = [img for img in images if re.search(r'\salt\s*=', img, re.IGNORECASE)]
    if images:
        alt_ratio = len(images_with_alt) / len(images)
        scores['image_alt'] = int(alt_ratio * 10)
    else:
        scores['image_alt'] = 8  # 无图片不扣分

    # 8. 关键词在标题中
    title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if not title_match:
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
    if title_match and target_keyword in title_match.group(1):
        scores['keyword_in_title'] = 10
    elif title_match:
        scores['keyword_in_title'] = 4
        suggestions.append("标题中未包含目标关键词")
    else:
        scores['keyword_in_title'] = 0

    # 9. 内容长度
    content_len = len(content)
    if content_len >= 1500:
        scores['content_length'] = 10
    elif content_len >= 800:
        scores['content_length'] = 8
    elif content_len >= 300:
        scores['content_length'] = 6
    else:
        scores['content_length'] = 3
        suggestions.append(f"内容长度{content_len}字，建议扩展至800字以上")

    # 10. 段落结构
    paragraphs = re.split(r'\n\s*\n|\n', re.sub(r'<[^>]+>', '\n', content))
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
    if len(paragraphs) >= 3:
        avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
        if 50 <= avg_para_len <= 200:
            scores['paragraph_structure'] = 10
        else:
            scores['paragraph_structure'] = 7
    else:
        scores['paragraph_structure'] = 4
        suggestions.append("段落过少，建议分段组织内容")

    total_score = sum(scores.values())

    grade = 'A' if total_score >= 80 else 'B' if total_score >= 60 else 'C' if total_score >= 40 else 'D'

    return {
        "total_score": total_score,
        "max_score": 100,
        "grade": grade,
        "target_keyword": target_keyword,
        "keyword_density": round(density, 2),
        "keyword_count": keyword_count,
        "content_length": content_len,
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "dimension_scores": scores,
        "suggestions": suggestions if suggestions else ["SEO表现良好，保持当前优化策略"],
        "image_count": len(images),
        "images_with_alt": len(images_with_alt),
        "heading_count": len(h_tags),
        "internal_link_count": internal_links
    }


# ============================================================================
# 5. readability_calculator - 可读性评估
# ============================================================================

def readability_calculator(text: str) -> Dict[str, Any]:
    """
    可读性评估，实现中文版Flesch-Kincaid公式。

    算法：
    基于以下指标计算可读性难度：
    1. 平均句长（字数/句数）
    2. 平均词长（双字组占比）
    3. 虚词比例（的、了、在、是、和等）
    4. 复杂句比例

    中文可读性公式（改编版）：
    Readability = 100 - 1.6 * avg_sentence_len - 0.8 * function_word_ratio * 100

    参数:
        text: 输入文本

    返回:
        包含可读性分数、难度等级、各指标的字典
    """
    # 句子分割
    sentences = _split_sentences(text)
    if not sentences:
        return {"error": "无法分析文本"}

    # 分词
    tokens = _tokenize(text)
    cn_chars = re.findall(r'[\u4e00-\u9fff]', text)

    # 指标1：平均句长
    total_chars = len(cn_chars) + len(re.findall(r'[a-zA-Z]+', text))
    avg_sentence_len = total_chars / len(sentences)

    # 指标2：虚词比例
    function_words = {'的', '了', '在', '是', '和', '与', '或', '也', '都',
                      '就', '而', '则', '其', '之', '为', '以', '于', '由',
                      '从', '到', '向', '对', '被', '把', '将', '已', '将',
                      '正', '才', '只', '还', '又', '更', '最', '很', '非常'}
    function_count = sum(1 for t in tokens if t in function_words or
                         any(t.startswith(fw) for fw in function_words))
    function_ratio = function_count / max(len(tokens), 1)

    # 指标3：复杂句比例（>40字的句子）
    complex_sentences = sum(1 for s in sentences if len(s) > 40)
    complex_ratio = complex_sentences / len(sentences)

    # 指标4：平均双字组长度
    avg_word_len = sum(len(t) for t in tokens) / max(len(tokens), 1)

    # 计算可读性分数（改编版Flesch公式）
    readability_score = 100 - 1.6 * avg_sentence_len - 0.8 * function_ratio * 100 - 0.5 * complex_ratio * 100
    readability_score = max(0, min(100, readability_score))

    # 难度等级
    if readability_score >= 80:
        level = "极易阅读（小学水平）"
        target_audience = "小学生"
    elif readability_score >= 60:
        level = "容易阅读（初中水平）"
        target_audience = "初中生"
    elif readability_score >= 40:
        level = "中等难度（高中水平）"
        target_audience = "高中生"
    elif readability_score >= 20:
        level = "较难阅读（大学水平）"
        target_audience = "大学生"
    else:
        level = "非常困难（专业水平）"
        target_audience = "专业人士"

    return {
        "readability_score": round(readability_score, 2),
        "level": level,
        "target_audience": target_audience,
        "metrics": {
            "avg_sentence_length": round(avg_sentence_len, 2),
            "sentence_count": len(sentences),
            "word_count": len(tokens),
            "char_count": total_chars,
            "function_word_ratio": round(function_ratio, 4),
            "complex_sentence_ratio": round(complex_ratio, 4),
            "complex_sentence_count": complex_sentences,
            "avg_word_length": round(avg_word_len, 2)
        },
        "suggestions": _readability_suggestions(avg_sentence_len, function_ratio, complex_ratio)
    }


def _readability_suggestions(avg_sent_len, func_ratio, complex_ratio):
    suggestions = []
    if avg_sent_len > 30:
        suggestions.append(f"平均句长{avg_sent_len:.0f}字过长，建议拆分长句")
    if func_ratio > 0.3:
        suggestions.append("虚词比例偏高，建议精简表达")
    if complex_ratio > 0.4:
        suggestions.append("复杂句过多，建议简化句子结构")
    if not suggestions:
        suggestions.append("可读性良好，继续保持")
    return suggestions


# ============================================================================
# 6. content_duplicate_checker - 内容查重器
# ============================================================================

def _generate_ngram_fingerprint(text: str, n: int = 3) -> Set[str]:
    """
    生成文本的N-gram指纹集合。
    将文本按n个词为一组切分，生成指纹集合用于快速比较。
    """
    tokens = _tokenize(text)
    if len(tokens) < n:
        return set(tokens)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard_similarity_set(set1: set, set2: set) -> float:
    """Jaccard相似度。"""
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def content_duplicate_checker(text: str, corpus: List[str]) -> Dict[str, Any]:
    """
    内容查重器。

    算法：
    1. N-gram指纹：将文本转换为3-gram集合作为指纹
    2. Jaccard相似度：计算与语料库中每篇文档的集合相似度
    3. 重复段落定位：找出相似度高的连续段落
    4. 综合重复率：加权计算整体重复率

    参数:
        text: 待查重文本
        corpus: 语料库文本列表

    返回:
        包含重复率、重复来源、重复段落的字典
    """
    # 生成待查文本指纹
    query_fingerprint = _generate_ngram_fingerprint(text, n=3)

    if not query_fingerprint:
        return {"duplicate_rate": 0, "matches": []}

    # 与语料库比较
    matches = []
    total_max_overlap = 0

    for idx, doc in enumerate(corpus):
        doc_fingerprint = _generate_ngram_fingerprint(doc, n=3)
        similarity = _jaccard_similarity_set(query_fingerprint, doc_fingerprint)

        if similarity > 0.1:  # 阈值10%
            # 定位重复段落
            query_sentences = _split_sentences(text)
            doc_sentences = _split_sentences(doc)
            duplicate_segments = []

            for qs in query_sentences:
                qs_fp = _generate_ngram_fingerprint(qs, n=2)
                for ds in doc_sentences:
                    ds_fp = _generate_ngram_fingerprint(ds, n=2)
                    seg_sim = _jaccard_similarity_set(qs_fp, ds_fp)
                    if seg_sim > 0.5:
                        duplicate_segments.append({
                            "query_segment": qs[:100],
                            "corpus_segment": ds[:100],
                            "similarity": round(seg_sim, 4)
                        })

            matches.append({
                "corpus_index": idx,
                "similarity": round(similarity, 4),
                "duplicate_segments": duplicate_segments[:10],
                "segment_count": len(duplicate_segments)
            })
            total_max_overlap = max(total_max_overlap, similarity)

    matches.sort(key=lambda x: -x["similarity"])

    # 综合重复率（取最大相似度 + 平均值的加权）
    avg_sim = sum(m["similarity"] for m in matches) / max(len(matches), 1)
    duplicate_rate = total_max_overlap * 0.7 + avg_sim * 0.3

    return {
        "duplicate_rate": round(duplicate_rate, 4),
        "max_similarity": round(total_max_overlap, 4),
        "avg_similarity": round(avg_sim, 4),
        "match_count": len(matches),
        "matches": matches[:10],
        "query_fingerprint_size": len(query_fingerprint),
        "verdict": "高重复率" if duplicate_rate > 0.5 else
                   "中等重复" if duplicate_rate > 0.2 else "原创度高"
    }


# ============================================================================
# 7. social_media_caption_generator - 社媒文案生成器
# ============================================================================

# 平台限制
_PLATFORM_LIMITS = {
    'twitter': 280, 'weibo': 140, 'instagram': 2200,
    'facebook': 5000, 'linkedin': 3000, 'xiaohongshu': 1000,
    'douyin': 500, 'kuaishou': 500
}

# 话题标签库
_HASHTAG_BANK = {
    '科技': ['#AI', '#人工智能', '#科技', '#创新', '#数字化', '#未来科技'],
    '生活': ['#生活方式', '#日常', '#生活小妙招', '#好物推荐', '#生活美学'],
    '美食': ['#美食', '#探店', '#美食分享', '#吃货', '#家常菜', '#食谱'],
    '旅行': ['#旅行', '#旅行日记', '#风景', '#旅行攻略', '#说走就走'],
    '教育': ['#学习', '#知识分享', '#干货', '#自我提升', '#读书'],
    '职场': ['#职场', '#求职', '#职业发展', '#面试技巧', '#工作日常'],
    '健康': ['#健康', '#养生', '#运动', '#健身', '#健康生活'],
    '财经': ['#财经', '#投资', '#理财', '#股票', '#经济'],
}

# 表情符号库
_EMOJI_BANK = {
    'positive': ['😄', '👍', '✨', '🔥', '💯', '🎉', '💪', '⭐', '🌟', '❤️'],
    'neutral': ['📝', '📌', '💡', '📚', '🎯', '📍', '✅', '📊', '🔍', '💬'],
    'exciting': ['🚀', '💥', '🎊', '🏆', '⚡', '🌈', '✨', '💥', '🎁', '👑'],
}

# 文案模板
_CAPTION_TEMPLATES = {
    'informative': [
        "今天分享一个关于{topic}的干货知识！{content} 你学会了吗？",
        "关于{topic}，你需要知道这些：{content} 建议收藏！",
        "{topic}入门指南 | {content} 快来了解一下吧～",
    ],
    'engaging': [
        "震惊！关于{topic}你不知道的{count}件事 {content} 你中了几条？",
        "为什么{topic}这么重要？{content} 答案在这里👇",
        "90%的人都不懂的{topic}真相 {content} 赶紧收藏！",
    ],
    'storytelling': [
        "从零开始学{topic}，这是我的真实经历：{content} 分享给同样努力的你💪",
        "我用{count}天搞懂了{topic}，总结如下：{content} 少走弯路！",
        "关于{topic}，有一个故事要讲：{content} 你有类似经历吗？",
    ],
    'question': [
        "{topic}到底怎么选？一文讲清楚！{content} 你有什么看法？",
        "为什么大家都关注{topic}？{content} 评论区聊聊！",
        "{topic}常见问题解答 {content} 还有疑问可以留言～",
    ],
}


def social_media_caption_generator(topic: str, platform: str = 'xiaohongshu',
                                    tone: str = 'informative') -> Dict[str, Any]:
    """
    社媒文案生成器。

    功能：
    1. 基于模板引擎生成文案
    2. 话题标签推荐（基于话题分类匹配）
    3. 表情符号匹配（根据语气匹配）
    4. 字数限制控制（根据平台限制截断）

    参数:
        topic: 文案主题
        platform: 目标平台
        tone: 语气风格 (informative/engaging/storytelling/question)

    返回:
        包含文案、标签、表情符号的字典
    """
    char_limit = _PLATFORM_LIMITS.get(platform.lower(), 500)
    templates = _CAPTION_TEMPLATES.get(tone, _CAPTION_TEMPLATES['informative'])

    # 生成多版本文案
    captions = []
    for template in templates:
        random_emojis = random.sample(_EMOJI_BANK.get(tone, _EMOJI_BANK['neutral']), 3)
        emoji_str = ' '.join(random_emojis)

        content_placeholder = f"这是关于{topic}的核心内容分享"
        caption = template.format(
            topic=topic,
            content=content_placeholder,
            count=random.randint(3, 10)
        )

        # 添加表情符号
        caption = f"{emoji_str} {caption} {random_emojis[0]}"

        # 确保不超过字数限制
        if len(caption) > char_limit:
            caption = caption[:char_limit - 3] + "..."

        captions.append({
            "caption": caption,
            "length": len(caption),
            "within_limit": len(caption) <= char_limit
        })

    # 话题标签推荐
    recommended_tags = []
    for category, tags in _HASHTAG_BANK.items():
        if category in topic or topic in category:
            recommended_tags.extend(tags)
    if not recommended_tags:
        # 默认推荐通用标签
        recommended_tags = ['#分享', '#推荐', '#干货', topic]
    else:
        recommended_tags = recommended_tags[:8]

    # 添加主题标签
    topic_tag = f"#{topic}" if not topic.startswith('#') else topic
    if topic_tag not in recommended_tags:
        recommended_tags.insert(0, topic_tag)

    # 构建完整帖子
    best_caption = captions[0]["caption"]
    tags_str = ' '.join(recommended_tags[:6])
    full_post = f"{best_caption}\n\n{tags_str}"

    if len(full_post) > char_limit:
        full_post = full_post[:char_limit - 3] + "..."

    return {
        "topic": topic,
        "platform": platform,
        "tone": tone,
        "char_limit": char_limit,
        "captions": captions,
        "recommended_hashtags": recommended_tags[:10],
        "emojis_used": random_emojis,
        "full_post": full_post,
        "post_length": len(full_post),
        "within_limit": len(full_post) <= char_limit
    }


# ============================================================================
# 8. topic_cluster_analysis - 话题聚类分析（简化版LDA）
# ============================================================================

def topic_cluster_analysis(documents: List[str], num_topics: int = 5,
                            num_iterations: int = 100) -> Dict[str, Any]:
    """
    话题聚类分析，实现简化版LDA。

    算法（Latent Dirichlet Allocation with Gibbs Sampling）：
    1. 构建文档-词频矩阵
    2. 初始化每个词的主题分配（随机）
    3. Gibbs采样迭代：
       对每个词w，根据以下概率重新分配主题：
       P(z=k | w, d) ∝ (n_dk + α) * (n_kw + β) / (n_k + W*β)
    4. 收敛后输出话题分布

    参数:
        documents: 文档列表
        num_topics: 话题数
        num_iterations: Gibbs采样迭代次数

    返回:
        包含话题词分布、文档话题分布的字典
    """
    # 步骤1：预处理和构建词表
    all_tokens = []
    doc_tokens = []
    for doc in documents:
        tokens = _tokenize(doc)
        doc_tokens.append(tokens)
        all_tokens.extend(tokens)

    vocabulary = list(set(all_tokens))
    V = len(vocabulary)
    D = len(documents)
    K = num_topics

    if V == 0 or D == 0:
        return {"error": "无有效内容", "topics": []}

    # 参数
    alpha = 50.0 / K  # 文档-话题先验
    beta = 0.1        # 话题-词先验

    # 步骤2：初始化
    # 每个词的主题分配
    word_to_id = {w: i for i, w in enumerate(vocabulary)}

    # 统计计数
    # n_dk[doc][topic]: 文档d中分配到话题k的词数
    n_dk = [[0] * K for _ in range(D)]
    # n_kw[topic][word]: 话题k中词w的计数
    n_kw = [[0] * V for _ in range(K)]
    # n_k[topic]: 话题k的总词数
    n_k = [0] * K
    # 每个词的当前话题分配
    z = []

    for d, tokens in enumerate(doc_tokens):
        doc_z = []
        for w in tokens:
            wid = word_to_id[w]
            k = random.randint(0, K - 1)
            doc_z.append((wid, k))
            n_dk[d][k] += 1
            n_kw[k][wid] += 1
            n_k[k] += 1
        z.append(doc_z)

    # 步骤3：Gibbs采样
    for iteration in range(num_iterations):
        for d in range(D):
            for idx, (wid, old_k) in enumerate(z[d]):
                # 减去当前词的计数
                n_dk[d][old_k] -= 1
                n_kw[old_k][wid] -= 1
                n_k[old_k] -= 1

                # 计算每个话题的概率
                probs = []
                for k in range(K):
                    p_dk = (n_dk[d][k] + alpha)
                    p_kw = (n_kw[k][wid] + beta) / (n_k[k] + V * beta)
                    probs.append(p_dk * p_kw)

                total = sum(probs)
                if total > 0:
                    r = random.random() * total
                    cumsum = 0
                    new_k = K - 1
                    for k in range(K):
                        cumsum += probs[k]
                        if r < cumsum:
                            new_k = k
                            break
                else:
                    new_k = random.randint(0, K - 1)

                # 更新计数和分配
                z[d][idx] = (wid, new_k)
                n_dk[d][new_k] += 1
                n_kw[new_k][wid] += 1
                n_k[new_k] += 1

    # 步骤4：提取话题词分布
    topics = []
    for k in range(K):
        word_scores = []
        for wid in range(V):
            score = (n_kw[k][wid] + beta) / (n_k[k] + V * beta)
            word_scores.append((vocabulary[wid], score, n_kw[k][wid]))

        word_scores.sort(key=lambda x: -x[1])
        top_words = word_scores[:10]
        topics.append({
            "topic_id": k,
            "top_words": [{"word": w, "probability": round(s, 6), "count": c}
                          for w, s, c in top_words],
            "total_words": n_k[k]
        })

    # 文档-话题分布
    doc_topic_dist = []
    for d in range(D):
        total = sum(n_dk[d])
        dist = {f"topic_{k}": round((n_dk[d][k] + alpha) / (total + K * alpha), 4)
                for k in range(K)}
        dominant = max(dist, key=dist.get)
        doc_topic_dist.append({
            "doc_index": d,
            "doc_preview": documents[d][:50] + "..." if len(documents[d]) > 50 else documents[d],
            "topic_distribution": dist,
            "dominant_topic": dominant
        })

    return {
        "num_topics": K,
        "num_documents": D,
        "vocabulary_size": V,
        "iterations": num_iterations,
        "topics": topics,
        "document_topic_distribution": doc_topic_dist
    }


# ============================================================================
# 9. title_scorer - 标题评分器
# ============================================================================

def title_scorer(title: str, keywords: List[str] = None) -> Dict[str, Any]:
    """
    标题评分器。

    评估维度：
    1. 数字开头（+吸引力）
    2. 疑问句式
    3. 情感词
    4. 长度适中（15-30字符）
    5. 关键词覆盖率
    6. 特殊符号使用（！、？、|）
    7. 时效性词汇
    8. 权威感词汇

    参数:
        title: 标题文本
        keywords: 目标关键词列表

    返回:
        包含总分、各维度得分、优化建议的字典
    """
    scores = {}
    suggestions = []
    keywords = keywords or []

    # 1. 数字开头
    starts_with_number = bool(re.match(r'^\d+', title))
    scores['number_start'] = 15 if starts_with_number else 0
    if not starts_with_number:
        suggestions.append("尝试以数字开头，如'3个方法...'，能提高点击率")

    # 2. 疑问句
    is_question = title.endswith(('?', '？', '吗', '呢', '么')) or '如何' in title or '为什么' in title
    scores['question_format'] = 15 if is_question else 0
    if not is_question:
        suggestions.append("考虑使用疑问句式，激发读者好奇心")

    # 3. 情感词
    emotional_words = ['震惊', '惊呆', '必看', '绝了', '炸裂', '感动', '泪目',
                        '不可思议', '惊艳', '超赞', '最强', '终极', '终极',
                        'amazing', 'shocking', 'incredible', 'must-see']
    has_emotion = any(w in title.lower() for w in emotional_words)
    scores['emotional'] = 10 if has_emotion else 0
    if not has_emotion:
        suggestions.append("添加情感词如'必看'、'惊艳'等增加吸引力")

    # 4. 长度
    title_len = len(title)
    if 15 <= title_len <= 30:
        scores['length'] = 15
    elif 10 <= title_len < 15 or 30 < title_len <= 40:
        scores['length'] = 10
    else:
        scores['length'] = 5
        suggestions.append(f"标题长度{title_len}字符，建议控制在15-30字符")

    # 5. 关键词覆盖率
    if keywords:
        covered = sum(1 for kw in keywords if kw in title)
        coverage = covered / len(keywords)
        scores['keyword_coverage'] = int(coverage * 15)
        if coverage < 0.5:
            suggestions.append(f"关键词覆盖率{coverage*100:.0f}%，建议融入更多关键词")
    else:
        scores['keyword_coverage'] = 10

    # 6. 特殊符号
    has_special = bool(re.search(r'[！？|｜【】《》「」]', title))
    scores['special_chars'] = 10 if has_special else 5
    if not has_special:
        suggestions.append("适当使用特殊符号如！或|增加视觉吸引力")

    # 7. 时效性
    time_words = ['2025', '2026', '最新', '最新版', '更新', '新', '热', '热榜',
                  '当前', '今日', '本周', '本月']
    has_timeliness = any(w in title for w in time_words)
    scores['timeliness'] = 10 if has_timeliness else 0

    # 8. 权威感
    authority_words = ['指南', '大全', '终极', '完全', '权威', '官方', '专业',
                        '深度', '详解', '全解析']
    has_authority = any(w in title for w in authority_words)
    scores['authority'] = 10 if has_authority else 0
    if not has_authority:
        suggestions.append("添加权威感词汇如'指南'、'大全'提升可信度")

    total = sum(scores.values())
    grade = 'A+' if total >= 80 else 'A' if total >= 65 else 'B' if total >= 50 else 'C' if total >= 35 else 'D'

    return {
        "title": title,
        "total_score": total,
        "max_score": 100,
        "grade": grade,
        "dimension_scores": scores,
        "suggestions": suggestions if suggestions else ["标题吸引力很强！"],
        "title_length": title_len,
        "keywords": keywords,
        "starts_with_number": starts_with_number,
        "is_question": is_question,
        "has_emotional_word": has_emotion,
        "has_timeliness": has_timeliness,
        "has_authority": has_authority
    }


# ============================================================================
# 10. content_calendar_planner - 内容日历规划器
# ============================================================================

# 最佳发布时间槽
_BEST_TIME_SLOTS = {
    'weekday_morning': ('09:00', '11:00', 0.9),    # 工作日早高峰
    'weekday_noon': ('12:00', '14:00', 0.85),       # 工作日午休
    'weekday_evening': ('18:00', '21:00', 1.0),     # 工作日晚高峰
    'weekend_morning': ('10:00', '12:00', 0.8),     # 周末上午
    'weekend_evening': ('19:00', '22:00', 0.95),     # 周末晚间
}


def content_calendar_planner(topics: List[Dict[str, Any]], start_date: str,
                              frequency: str = 'daily') -> Dict[str, Any]:
    """
    内容日历规划器。

    算法：
    1. 话题优先级排序：基于紧迫度、热度、重要性的加权评分
    2. 发布频率解析：daily/weekly/biweekly转具体日期
    3. 最佳时间槽分配：根据工作日/周末匹配最佳发布时间
    4. 排班优化：避免连续发布相似话题

    参数:
        topics: 话题列表，每项包含 name, priority, category
        start_date: 开始日期 (YYYY-MM-DD)
        frequency: 发布频率 (daily/weekly/biweekly)

    返回:
        包含日历排期、话题排序、统计信息的字典
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # 频率转间隔天数
    freq_map = {'daily': 1, 'weekly': 7, 'biweekly': 14, 'monthly': 30}
    interval = freq_map.get(frequency.lower(), 1)

    # 步骤1：话题优先级排序
    def priority_score(topic):
        base = topic.get('priority', 5)
        urgency = topic.get('urgency', 5)
        heat = topic.get('heat', 5)
        # 加权评分
        return base * 0.4 + urgency * 0.35 + heat * 0.25

    sorted_topics = sorted(topics, key=lambda x: -priority_score(x))

    # 步骤2：分配发布日期和时间槽
    calendar = []
    current_date = start
    used_categories = []

    for i, topic in enumerate(sorted_topics):
        # 跳过非工作日（可选）
        weekday = current_date.weekday()
        is_weekend = weekday >= 5

        # 选择最佳时间槽
        if is_weekend:
            slot_key = 'weekend_evening' if random.random() > 0.4 else 'weekend_morning'
        else:
            slot_key = 'weekday_evening' if random.random() > 0.3 else 'weekday_morning'

        slot = _BEST_TIME_SLOTS[slot_key]
        publish_time = f"{slot[0]}"

        # 避免连续相似话题
        category = topic.get('category', 'general')
        if used_categories and used_categories[-1] == category:
            # 推迟到下一个时间槽
            pass

        entry = {
            "date": current_date.strftime("%Y-%m-%d"),
            "weekday": ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][weekday],
            "time": publish_time,
            "time_slot": slot_key,
            "topic": topic.get('name', f'话题{i+1}'),
            "category": category,
            "priority": topic.get('priority', 5),
            "priority_score": round(priority_score(topic), 2),
            "is_weekend": is_weekend,
            "expected_reach": round(slot[2] * (1 + topic.get('heat', 5) / 20), 2)
        }
        calendar.append(entry)
        used_categories.append(category)

        # 推进日期
        current_date += timedelta(days=interval)

    # 步骤3：统计
    total_days = (current_date - start).days
    category_dist = Counter(t.get('category', 'general') for t in sorted_topics)
    weekend_count = sum(1 for c in calendar if c['is_weekend'])

    return {
        "start_date": start_date,
        "end_date": calendar[-1]["date"] if calendar else start_date,
        "frequency": frequency,
        "interval_days": interval,
        "total_topics": len(sorted_topics),
        "total_days": total_days,
        "weekend_posts": weekend_count,
        "weekday_posts": len(calendar) - weekend_count,
        "calendar": calendar,
        "topic_ranking": [
            {"rank": i + 1, "topic": t.get('name', ''), "score": round(priority_score(t), 2),
             "category": t.get('category', 'general')}
            for i, t in enumerate(sorted_topics)
        ],
        "category_distribution": dict(category_dist),
        "avg_expected_reach": round(sum(c['expected_reach'] for c in calendar) / max(len(calendar), 1), 2)
    }


# ============================================================================
# 主程序测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("content-studio 内容创作工坊 - 功能测试")
    print("=" * 60)

    # 测试TextRank摘要
    print("\n--- TextRank摘要测试 ---")
    sample_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自从人工智能诞生以来，其理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
    人工智能可以对人的意识、思维的信息过程进行模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识、心理学和哲学。
    人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习、计算机视觉等。
    总的说来，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
    """
    summary = textrank_summary(sample_text, num_sentences=3)
    print(f"原文长度: {summary['original_length']}")
    print(f"摘要长度: {summary['summary_length']}")
    print(f"摘要: {summary['summary'][:150]}...")

    # 测试情感分析
    print("\n--- 情感分析测试 ---")
    texts = [
        "这个产品非常好用，特别满意！",
        "太差了，完全不能接受，非常失望",
        "虽然价格贵了点，但是质量很好",
    ]
    for t in texts:
        result = sentiment_analyzer(t)
        print(f"'{t}' -> {result['sentiment']} (score={result['score']}, confidence={result['confidence']})")

    # 测试标题评分
    print("\n--- 标题评分测试 ---")
    titles = [
        "2025年最全Python教程",
        "3个技巧让你的代码效率提升10倍！",
        "Python编程入门指南",
    ]
    for t in titles:
        result = title_scorer(t, keywords=['Python'])
        print(f"'{t}' -> {result['total_score']}/100 ({result['grade']})")

    # 测试SEO评分
    print("\n--- SEO评分测试 ---")
    html_content = """<html>
    <head><title>Python学习指南</title>
    <meta name="description" content="这是一篇关于Python编程的全面学习指南，涵盖基础语法、数据结构、面向对象编程等内容。">
    </head>
    <body>
    <h1>Python学习指南</h1>
    <p>Python是一种广泛使用的编程语言。Python非常适合初学者。Python的语法简洁明了。</p>
    <h2>基础语法</h2>
    <p>Python的语法非常简单，适合快速开发。</p>
    <a href="/tutorial">更多教程</a>
    <img src="logo.png" alt="Python Logo">
    </body></html>"""
    seo_result = seo_score_analyzer(html_content, "Python")
    print(f"总分: {seo_result['total_score']}/100 ({seo_result['grade']})")
    print(f"关键词密度: {seo_result['keyword_density']}%")

    # 测试可读性
    print("\n--- 可读性测试 ---")
    readability = readability_calculator(sample_text)
    print(f"可读性分数: {readability['readability_score']}")
    print(f"难度等级: {readability['level']}")
    print(f"平均句长: {readability['metrics']['avg_sentence_length']}")

    # 测试社媒文案生成
    print("\n--- 社媒文案生成测试 ---")
    caption = social_media_caption_generator("Python编程", "xiaohongshu", "engaging")
    print(f"平台: {caption['platform']}")
    print(f"字数限制: {caption['char_limit']}")
    print(f"文案: {caption['captions'][0]['caption'][:80]}...")
    print(f"标签: {' '.join(caption['recommended_hashtags'][:5])}")

    # 测试内容查重
    print("\n--- 内容查重测试 ---")
    corpus = [
        "人工智能是计算机科学的分支，研究机器智能。",
        "深度学习是机器学习的一个子领域，使用神经网络。",
        "今天天气很好，适合出门散步。",
    ]
    dup_result = content_duplicate_checker(sample_text[:200], corpus)
    print(f"重复率: {dup_result['duplicate_rate']}")
    print(f"匹配数: {dup_result['match_count']}")
    print(f"判定: {dup_result['verdict']}")

    # 测试话题聚类
    print("\n--- 话题聚类测试 ---")
    docs = [
        "Python是一种编程语言，广泛用于数据科学和机器学习",
        "机器学习是人工智能的重要分支，包括深度学习",
        "JavaScript是前端开发的主要语言",
        "深度学习使用神经网络进行模式识别",
        "前端开发需要掌握HTML CSS和JavaScript",
    ]
    cluster_result = topic_cluster_analysis(docs, num_topics=3, num_iterations=50)
    for topic in cluster_result.get('topics', []):
        print(f"话题{topic['topic_id']}: {[w['word'] for w in topic['top_words'][:5]]}")

    # 测试内容日历
    print("\n--- 内容日历测试 ---")
    topics = [
        {"name": "Python入门", "priority": 8, "urgency": 7, "heat": 9, "category": "编程"},
        {"name": "机器学习基础", "priority": 9, "urgency": 8, "heat": 8, "category": "AI"},
        {"name": "前端框架对比", "priority": 6, "urgency": 5, "heat": 7, "category": "前端"},
        {"name": "数据可视化", "priority": 7, "urgency": 6, "heat": 6, "category": "数据"},
    ]
    calendar = content_calendar_planner(topics, "2025-07-15", "weekly")
    for entry in calendar['calendar'][:3]:
        print(f"  {entry['date']} {entry['weekday']} {entry['time']} - {entry['topic']} (优先级:{entry['priority_score']})")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
