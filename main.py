"""
内容创作工坊 (content-studio)

提供10个内容创作与SEO优化工具，包括博文大纲生成、SEO关键词分析（TF-IDF）、
社媒文案生成、标签生成、内容日历、标题评分、文章摘要（TextRank）、文本相似度检测、
Meta描述生成和邮件模板渲染。

主要功能:
    - generate_blog_outline: 博文大纲生成
    - seo_keyword_analyzer: SEO关键词分析（TF-IDF）
    - social_media_caption: 社媒文案生成
    - hashtag_generator: 标签生成器
    - content_calendar_generator: 内容日历生成
    - headline_analyzer: 标题评分器
    - content_summarizer: 文章摘要（TextRank算法）
    - plagiarism_checker: 文本相似度检测
    - meta_description_generator: SEO Meta描述生成
    - email_template_renderer: 邮件模板渲染

依赖:
    - jieba: 中文分词
    - scikit-learn: TF-IDF计算
    - numpy: 数值计算
"""

import re
import math
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict


# =============================================================================
# 1. 博文大纲生成
# =============================================================================
def generate_blog_outline(topic, keywords, target_audience):
    """
    根据主题、关键词和目标受众生成博文大纲。

    Args:
        topic (str): 博文主题。
        keywords (list): 关键词列表，用于指导内容方向。
        target_audience (str): 目标受众描述。

    Returns:
        dict: 包含以下键的字典:
            - "title": 建议标题
            - "audience": 目标受众
            - "outline": 大纲列表，每项为 {"section": 章节标题, "points": 要点列表}
            - "keywords": 关键词列表
            - "estimated_word_count": 预估字数

    Example:
        >>> outline = generate_blog_outline(
        ...     "Python自动化",
        ...     ["自动化", "效率", "脚本"],
        ...     "Python初学者"
        ... )
    """
    sections = [
        {
            "section": "引言",
            "points": [
                f"引入主题：{topic}的重要性和现实意义",
                f"目标读者：{target_audience}能从中获得什么",
                "文章结构概览",
            ],
        },
        {
            "section": f"什么是{topic}",
            "points": [
                f"{topic}的基本概念和定义",
                "发展背景与现状",
            ],
        },
        {
            "section": "核心要点",
            "points": [],
        },
        {
            "section": "实践指南",
            "points": [
                "具体操作步骤",
                "常见问题与解决方案",
            ],
        },
        {
            "section": "案例分析",
            "points": [
                "实际应用场景示例",
                "效果与成果展示",
            ],
        },
        {
            "section": "总结与展望",
            "points": [
                "核心观点回顾",
                f"对{target_audience}的建议",
                "未来趋势展望",
            ],
        },
    ]

    # 将关键词分配到核心要点
    for kw in keywords:
        sections[2]["points"].append(f"深入解析：{kw}")

    title = f"{topic}完全指南：面向{target_audience}的实用教程"

    # 预估字数：每个section约500字
    estimated = len(sections) * 500

    return {
        "title": title,
        "audience": target_audience,
        "outline": sections,
        "keywords": keywords,
        "estimated_word_count": estimated,
    }


# =============================================================================
# 2. SEO关键词分析（TF-IDF）
# =============================================================================
def seo_keyword_analyzer(content, top_n=10):
    """
    使用TF-IDF算法分析内容中的关键词。

    Args:
        content (str): 待分析的文本内容。
        top_n (int): 返回的关键词数量，默认10。

    Returns:
        dict: 包含以下键的字典:
            - "keywords": 关键词列表，每项为 {"word": 词语, "tfidf": TF-IDF值, "tf": 词频}
            - "total_words": 总词数
            - "unique_words": 去重后词数

    Example:
        >>> result = seo_keyword_analyzer("Python是一种编程语言...", top_n=5)
        {'keywords': [{'word': 'Python', 'tfidf': 0.15, 'tf': 3}, ...], ...}
    """
    import jieba

    # 分词
    words = list(jieba.cut(content))
    # 过滤停用词和标点
    stop_chars = set("，。！？、；：""''（）【】《》\n\r\t ...")
    filtered_words = [
        w.strip() for w in words
        if w.strip() and w.strip() not in stop_chars and len(w.strip()) > 1
    ]

    if not filtered_words:
        return {"keywords": [], "total_words": 0, "unique_words": 0}

    total_words = len(filtered_words)

    # 计算词频 (TF)
    word_counts = Counter(filtered_words)
    tf_dict = {word: count / total_words for word, count in word_counts.items()}

    # 计算IDF（基于文档内容模拟，使用对数平滑）
    # 将内容按句号分割为多个"文档"来模拟语料库
    docs = [d.strip() for d in re.split(r"[。！？\n]", content) if d.strip()]
    if len(docs) < 2:
        docs = [content]

    N = len(docs)
    df_dict = defaultdict(int)
    for word in word_counts:
        for doc in docs:
            if word in doc:
                df_dict[word] += 1

    # IDF = log(N / (df + 1)) + 1
    idf_dict = {
        word: math.log(N / (df + 1)) + 1
        for word, df in df_dict.items()
    }

    # TF-IDF
    tfidf_dict = {
        word: tf_dict[word] * idf_dict.get(word, 1.0)
        for word in word_counts
    }

    # 排序取top_n
    sorted_words = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    keywords = [
        {"word": word, "tfidf": round(score, 4), "tf": word_counts[word]}
        for word, score in sorted_words
    ]

    return {
        "keywords": keywords,
        "total_words": total_words,
        "unique_words": len(word_counts),
    }


# =============================================================================
# 3. 社媒文案生成
# =============================================================================
def social_media_caption(topic, platform, tone):
    """
    根据主题、平台和语气生成社交媒体文案。

    Args:
        topic (str): 文案主题。
        platform (str): 平台名称，支持 "微博"/"小红书"/"抖音"/"微信公众号"/"twitter"。
        tone (str): 语气风格，支持 "活泼"/"专业"/"幽默"/"感性"/"励志"。

    Returns:
        dict: 包含以下键的字典:
            - "caption": 生成的文案文本
            - "platform": 平台名称
            - "tone": 语气风格
            - "char_count": 字符数
            - "suggested_hashtags": 推荐标签列表

    Example:
        >>> result = social_media_caption("新店开业", "小红书", "活泼")
    """
    tone_templates = {
        "活泼": {
            "intro": ["哇！{topic}也太棒了吧！"],
            "body": ["今天必须和大家分享这个好消息～", "真的超级推荐！"],
            "ending": ["快冲！姐妹们！🏃‍♀️✨"],
        },
        "专业": {
            "intro": ["关于{topic}，你需要知道这些。"],
            "body": ["本文将从专业角度为您深度解析。", "数据说话，拒绝套路。"],
            "ending": ["关注我们，获取更多专业内容。"],
        },
        "幽默": {
            "intro": ["{topic}？别慌，我有故事说给你听😂"],
            "body": ["事情是这样的...", "我差点信了（不是）"],
            "ending": ["点个赞再走吧，就当给我买根冰棍🍦"],
        },
        "感性": {
            "intro": ["想起{topic}，心里突然柔软了一下。"],
            "body": ["有些故事值得被记录。", "愿我们都被温柔以待。"],
            "ending": ["你呢？有什么想分享的吗？🌙"],
        },
        "励志": {
            "intro": ["{topic}，是你改变的开始。"],
            "body": ["没有人的成功是一蹴而就的。", "但每一次坚持都在靠近梦想。"],
            "ending": ["加油！你值得更好的自己！💪"],
        },
    }

    platform_config = {
        "微博": {"max_chars": 140, "emoji_density": "medium"},
        "小红书": {"max_chars": 1000, "emoji_density": "high"},
        "抖音": {"max_chars": 100, "emoji_density": "high"},
        "微信公众号": {"max_chars": 200, "emoji_density": "low"},
        "twitter": {"max_chars": 280, "emoji_density": "medium"},
    }

    template = tone_templates.get(tone, tone_templates["活泼"])
    platform_info = platform_config.get(platform, {"max_chars": 200, "emoji_density": "medium"})

    # 组装文案
    parts = []
    for intro in template["intro"]:
        parts.append(intro.format(topic=topic))
    for body in template["body"]:
        parts.append(body)
    for ending in template["ending"]:
        parts.append(ending)

    caption = "\n".join(parts)

    # 生成推荐标签
    hashtags = [f"#{topic}", f"#{platform}分享", f"#{tone}日常"]

    return {
        "caption": caption,
        "platform": platform,
        "tone": tone,
        "char_count": len(caption),
        "suggested_hashtags": hashtags,
    }


# =============================================================================
# 4. 标签生成器
# =============================================================================
def hashtag_generator(topic, count=10):
    """
    根据主题生成社交媒体标签。

    Args:
        topic (str): 主题关键词。
        count (int): 生成标签数量，默认10。

    Returns:
        dict: 包含以下键的字典:
            - "hashtags": 标签列表
            - "topic": 原始主题
            - "count": 生成数量

    Example:
        >>> result = hashtag_generator("健身", count=5)
        {'hashtags': ['#健身', '#健身打卡', ...], 'topic': '健身', 'count': 5}
    """
    # 标签后缀模板
    suffix_templates = [
        "", "打卡", "日常", "分享", "推荐", "教程", "日记",
        "攻略", "指南", "心得", "记录", "vlog", "知识", "小技巧",
        "好物", "安利", "必看", "入门", "进阶", "合集",
    ]

    # 标签前缀模板
    prefix_templates = [
        "我的", "新手", "零基础", "高效", "超全", "必看",
    ]

    hashtags = [f"#{topic}"]

    # 生成带后缀的标签
    for suffix in suffix_templates:
        if len(hashtags) >= count:
            break
        tag = f"#{topic}{suffix}"
        if tag not in hashtags:
            hashtags.append(tag)

    # 生成带前缀的标签
    for prefix in prefix_templates:
        if len(hashtags) >= count:
            break
        tag = f"#{prefix}{topic}"
        if tag not in hashtags:
            hashtags.append(tag)

    # 如果还不够，用英文变体补充
    if len(hashtags) < count:
        en_variants = [
            f"#{topic}life", f"#{topic}tips", f"#{topic}lover",
            f"#{topic}gram", f"#{topic}daily",
        ]
        for tag in en_variants:
            if len(hashtags) >= count:
                break
            if tag not in hashtags:
                hashtags.append(tag)

    return {
        "hashtags": hashtags[:count],
        "topic": topic,
        "count": len(hashtags[:count]),
    }


# =============================================================================
# 5. 内容日历生成
# =============================================================================
def content_calendar_generator(topics, start_date, frequency="weekly"):
    """
    根据主题列表生成内容发布日历。

    Args:
        topics (list): 内容主题列表。
        start_date (str): 开始日期，格式 "YYYY-MM-DD"。
        frequency (str): 发布频率，支持 "daily"/"weekly"/"biweekly"。

    Returns:
        dict: 包含以下键的字典:
            - "calendar": 日历列表，每项为 {"date": 日期, "topic": 主题, "weekday": 星期几}
            - "total_posts": 总发布数
            - "frequency": 发布频率
            - "date_range": 日期范围 {"start": 开始, "end": 结束}

    Example:
        >>> cal = content_calendar_generator(["AI", "Python"], "2025-01-01", "weekly")
    """
    freq_days = {
        "daily": 1,
        "weekly": 7,
        "biweekly": 14,
    }
    interval = freq_days.get(frequency, 7)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    weekdays_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    calendar = []
    current_date = start
    topic_index = 0

    for topic in topics:
        weekday = weekdays_cn[current_date.weekday()]
        calendar.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "topic": topic,
            "weekday": weekday,
            "post_number": topic_index + 1,
        })
        current_date += timedelta(days=interval)
        topic_index += 1

    return {
        "calendar": calendar,
        "total_posts": len(calendar),
        "frequency": frequency,
        "date_range": {
            "start": calendar[0]["date"] if calendar else "",
            "end": calendar[-1]["date"] if calendar else "",
        },
    }


# =============================================================================
# 6. 标题评分器
# =============================================================================
def headline_analyzer(headline):
    """
    分析标题并给出评分（情感/长度/关键词/吸引力）。

    Args:
        headline (str): 待分析的标题文本。

    Returns:
        dict: 包含以下键的字典:
            - "headline": 原标题
            - "total_score": 总分（0-100）
            - "grade": 等级（A/B/C/D）
            - "details": 评分明细字典:
                - "length_score": 长度评分
                - "emotion_score": 情感评分
                - "keyword_score": 关键词评分
                - "power_score": 吸引力评分
            - "suggestions": 改进建议列表

    Example:
        >>> result = headline_analyzer("震惊！这个方法让你月入过万")
    """
    details = {}
    suggestions = []

    # 1. 长度评分（理想长度: 20-60字符）
    length = len(headline)
    if 20 <= length <= 60:
        details["length_score"] = 25
    elif 15 <= length < 20 or 60 < length <= 80:
        details["length_score"] = 18
    elif 10 <= length < 15 or 80 < length <= 100:
        details["length_score"] = 10
    else:
        details["length_score"] = 5
        suggestions.append("标题长度不理想，建议控制在20-60个字符")

    # 2. 情感评分（包含情感词汇加分）
    emotion_words = [
        "震惊", "惊喜", "震撼", "感动", "激动", "兴奋", "开心",
        "可怕", "危险", "警告", "注意", "必看", "惊人",
        "最", "第一", "独家", "首发", "重磅", "突破",
    ]
    emotion_count = sum(1 for word in emotion_words if word in headline)
    details["emotion_score"] = min(emotion_count * 8, 25)
    if emotion_count == 0:
        suggestions.append("可考虑添加情感词汇增强吸引力")

    # 3. 关键词评分（包含数字、问号等）
    keyword_score = 0
    if re.search(r"\d+", headline):
        keyword_score += 10  # 含数字
    if "?" in headline or "？" in headline:
        keyword_score += 8  # 含问号
    if ":" in headline or "：" in headline:
        keyword_score += 7  # 含冒号
    if re.search(r"[【\[].*[\]\]]", headline):
        keyword_score += 5  # 含方括号
    details["keyword_score"] = min(keyword_score, 25)

    # 4. 吸引力评分（点击诱饵词）
    power_words = [
        "如何", "为什么", "揭秘", "真相", "秘密", "终极",
        "完全", "彻底", "简单", "快速", "免费", "必备",
        "你应该", "千万别", "一定要", "建议收藏",
    ]
    power_count = sum(1 for word in power_words if word in headline)
    details["power_score"] = min(power_count * 9, 25)
    if power_count == 0:
        suggestions.append('可添加「如何」「揭秘」等吸引力词汇')

    total_score = sum(details.values())
    if total_score >= 80:
        grade = "A"
    elif total_score >= 60:
        grade = "B"
    elif total_score >= 40:
        grade = "C"
    else:
        grade = "D"
        suggestions.append("标题吸引力不足，建议重写")

    if not suggestions:
        suggestions.append("标题表现良好，可考虑A/B测试")

    return {
        "headline": headline,
        "total_score": total_score,
        "grade": grade,
        "details": details,
        "suggestions": suggestions,
    }


# =============================================================================
# 7. 文章摘要（TextRank）
# =============================================================================
def content_summarizer(text, sentences_count=3):
    """
    使用TextRank算法提取文章摘要。

    Args:
        text (str): 待摘要的文章文本。
        sentences_count (int): 摘要句子数量，默认3。

    Returns:
        dict: 包含以下键的字典:
            - "summary": 摘要文本
            - "sentences": 摘要句子列表
            - "original_length": 原文字符数
            - "summary_length": 摘要字符数
            - "compression_ratio": 压缩比

    Example:
        >>> result = content_summarizer("很长的文章...", sentences_count=2)
    """
    import jieba

    # 分句
    sentences = re.split(r"[。！？\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) <= sentences_count:
        summary_text = "。".join(sentences) + "。"
        return {
            "summary": summary_text,
            "sentences": sentences,
            "original_length": len(text),
            "summary_length": len(summary_text),
            "compression_ratio": round(len(summary_text) / max(len(text), 1), 2),
        }

    # 分词并构建词频表
    words_per_sentence = []
    all_words = []
    for sent in sentences:
        words = [w for w in jieba.cut(sent) if w.strip() and len(w.strip()) > 1]
        words_per_sentence.append(words)
        all_words.extend(words)

    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1
    for w in word_freq:
        word_freq[w] = word_freq[w] / max_freq

    # 计算句子得分（基于词频）
    sentence_scores = []
    for i, words in enumerate(words_per_sentence):
        score = sum(word_freq.get(w, 0) for w in words)
        # 归一化
        score = score / max(len(words), 1)
        # 位置加权：开头和结尾的句子权重略高
        position_weight = 1.0
        if i == 0:
            position_weight = 1.2
        elif i == len(sentences) - 1:
            position_weight = 1.1
        score *= position_weight
        sentence_scores.append((i, score))

    # 按得分排序取top_n
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = sorted([idx for idx, _ in sentence_scores[:sentences_count]])

    # 按原文顺序输出摘要
    selected_sentences = [sentences[i] for i in selected_indices]
    summary_text = "。".join(selected_sentences) + "。"

    return {
        "summary": summary_text,
        "sentences": selected_sentences,
        "original_length": len(text),
        "summary_length": len(summary_text),
        "compression_ratio": round(len(summary_text) / max(len(text), 1), 2),
    }


# =============================================================================
# 8. 文本相似度检测
# =============================================================================
def plagiarism_checker(text1, text2):
    """
    检测两段文本的相似度。

    使用Jaccard相似度和余弦相似度两种方法进行检测。

    Args:
        text1 (str): 第一段文本。
        text2 (str): 第二段文本。

    Returns:
        dict: 包含以下键的字典:
            - "similarity_score": 综合相似度（0-1）
            - "jaccard_similarity": Jaccard相似度
            - "cosine_similarity": 余弦相似度
            - "common_words": 共同词汇列表
            - "verdict": 判定结论（"高度相似"/"中等相似"/"低相似度"/"几乎不同"）

    Example:
        >>> result = plagiarism_checker("今天天气真好", "今天天气不错")
    """
    import jieba

    # 分词
    words1 = set(w.strip() for w in jieba.cut(text1) if w.strip() and len(w.strip()) > 1)
    words2 = set(w.strip() for w in jieba.cut(text2) if w.strip() and len(w.strip()) > 1)

    if not words1 or not words2:
        return {
            "similarity_score": 0.0,
            "jaccard_similarity": 0.0,
            "cosine_similarity": 0.0,
            "common_words": [],
            "verdict": "几乎不同",
        }

    # Jaccard相似度
    intersection = words1 & words2
    union = words1 | words2
    jaccard = len(intersection) / len(union) if union else 0

    # 余弦相似度
    all_words = words1 | words2
    vec1 = [1 if w in words1 else 0 for w in all_words]
    vec2 = [1 if w in words2 else 0 for w in all_words]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    cosine = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    # 综合相似度
    similarity = (jaccard + cosine) / 2

    # 判定
    if similarity >= 0.7:
        verdict = "高度相似"
    elif similarity >= 0.4:
        verdict = "中等相似"
    elif similarity >= 0.2:
        verdict = "低相似度"
    else:
        verdict = "几乎不同"

    return {
        "similarity_score": round(similarity, 4),
        "jaccard_similarity": round(jaccard, 4),
        "cosine_similarity": round(cosine, 4),
        "common_words": list(intersection),
        "verdict": verdict,
    }


# =============================================================================
# 9. SEO Meta描述生成
# =============================================================================
def meta_description_generator(title, content):
    """
    根据标题和内容生成SEO Meta描述。

    Args:
        title (str): 页面标题。
        content (str): 页面内容。

    Returns:
        dict: 包含以下键的字典:
            - "meta_description": 生成的Meta描述
            - "char_count": 字符数
            - "contains_keyword": 是否包含关键词
            - "suggestions": 优化建议列表

    Example:
        >>> result = meta_description_generator("Python教程", "Python是一种...")
    """
    suggestions = []

    # 提取内容前几句作为摘要基础
    sentences = re.split(r"[。！？\n]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 从标题提取关键词
    title_keywords = [w for w in re.split(r"[\s\-_|/]+", title) if len(w) > 1]

    # 构建描述
    base_text = ""
    for sent in sentences:
        if len(base_text) + len(sent) <= 150:
            base_text += sent + "。"
        else:
            remaining = 150 - len(base_text)
            if remaining > 10:
                base_text += sent[:remaining] + "..."
            break

    if not base_text:
        base_text = title + " - 了解更多相关内容。"

    # 确保以标题相关内容开头
    if title_keywords:
        first_keyword = title_keywords[0]
        if first_keyword not in base_text[:30]:
            base_text = f"关于{first_keyword}： " + base_text

    # 控制长度（SEO最佳: 120-160字符）
    char_count = len(base_text)
    if char_count > 160:
        base_text = base_text[:157] + "..."
        char_count = len(base_text)
    elif char_count < 120:
        suggestions.append(f"描述偏短（{char_count}字符），建议补充至120-160字符")

    # 检查关键词
    contains_keyword = any(kw in base_text for kw in title_keywords)
    if not contains_keyword:
        suggestions.append("描述中未包含标题关键词，建议添加")
    else:
        suggestions.append("描述已包含标题关键词，表现良好")

    return {
        "meta_description": base_text,
        "char_count": len(base_text),
        "contains_keyword": contains_keyword,
        "suggestions": suggestions,
    }


# =============================================================================
# 10. 邮件模板渲染
# =============================================================================
def email_template_renderer(template, variables):
    """
    渲染邮件模板，将变量替换到模板中。

    支持两种占位符语法:
        - {{variable_name}} - 双花括号
        - {variable_name} - 单花括号（与Python format兼容）

    Args:
        template (str): 邮件模板字符串，包含占位符。
        variables (dict): 变量字典，键为变量名，值为替换内容。

    Returns:
        dict: 包含以下键的字典:
            - "rendered": 渲染后的邮件文本
            - "replaced_count": 替换的变量数量
            - "unreplaced": 未替换的占位符列表
            - "success": 是否全部替换成功

    Example:
        >>> result = email_template_renderer(
        ...     "Hello {{name}}, your order {{order_id}} is ready.",
        ...     {"name": "Alice", "order_id": "12345"}
        ... )
    """
    rendered = template
    replaced_count = 0
    unreplaced = []

    # 替换 {{variable}} 双花括号语法
    double_brace_pattern = re.compile(r"\{\{(\w+)\}\}")
    for match in double_brace_pattern.finditer(template):
        var_name = match.group(1)
        if var_name in variables:
            rendered = rendered.replace(
                f"{{{{{var_name}}}}}", str(variables[var_name])
            )
            replaced_count += 1
        else:
            unreplaced.append(f"{{{{{var_name}}}}}")

    # 替换 {variable} 单花括号语法
    single_brace_pattern = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
    for match in single_brace_pattern.finditer(rendered):
        var_name = match.group(1)
        if var_name in variables:
            rendered = rendered.replace(
                f"{{{var_name}}}", str(variables[var_name]), 1
            )
            replaced_count += 1
        else:
            if f"{{{var_name}}}" not in unreplaced:
                unreplaced.append(f"{{{var_name}}}")

    return {
        "rendered": rendered,
        "replaced_count": replaced_count,
        "unreplaced": unreplaced,
        "success": len(unreplaced) == 0,
    }


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("内容创作工坊 (content-studio)")
    print("=" * 60)
    print("可用工具:")
    tools = [
        "1. generate_blog_outline       - 博文大纲生成",
        "2. seo_keyword_analyzer        - SEO关键词分析",
        "3. social_media_caption        - 社媒文案生成",
        "4. hashtag_generator           - 标签生成器",
        "5. content_calendar_generator  - 内容日历",
        "6. headline_analyzer           - 标题评分器",
        "7. content_summarizer          - 文章摘要",
        "8. plagiarism_checker          - 文本相似度检测",
        "9. meta_description_generator  - SEO Meta描述",
        "10. email_template_renderer    - 邮件模板渲染",
    ]
    for tool in tools:
        print(f"  {tool}")
    print("=" * 60)

    # 演示：标题评分
    print("\n演示 - 标题评分:")
    headline_result = headline_analyzer("震惊！2025年最全Python教程，建议收藏！")
    print(f"  标题: {headline_result['headline']}")
    print(f"  得分: {headline_result['total_score']}/100 (等级: {headline_result['grade']})")
    print(f"  建议: {headline_result['suggestions'][0]}")
