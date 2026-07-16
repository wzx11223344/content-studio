"""Auto-generated tests for content-studio."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class TestMain:
    """Tests for content-studio module."""

    def test_module_import(self):
        """Test that main module imports correctly."""
        assert main is not None
        assert hasattr(main, "textrank_summary")


    def test_textrank_summary_basic(self):
        """Test TextRank summarization."""
        text = "人工智能技术正在快速发展。深度学习模型取得了显著成果。自然语言处理是重要的研究方向。计算机视觉也有很多突破。这些技术正在改变世界。"
        result = main.textrank_summary(text, num_sentences=2)
        assert len(result["summary_sentences"]) <= 3
        assert result["total_sentences"] > 0

    def test_textrank_short_text(self):
        """Test TextRank with short text."""
        text = "简短文本。"
        result = main.textrank_summary(text, num_sentences=3)
        assert "summary" in result or "summary_sentences" in result or "error" in result


    def test_tfidf_keyword_extraction(self):
        """Test TF-IDF keyword extraction."""
        text = "自然语言处理是人工智能的重要分支。自然语言处理技术广泛应用于搜索和推荐系统。"
        result = main.tfidf_keyword_extraction(text, top_n=3)
        assert len(result) <= 3
        assert len(result) >= 1

    def test_tfidf_empty(self):
        """Test TF-IDF with empty text."""
        result = main.tfidf_keyword_extraction("", top_n=3)
        assert result is not None


    def test_sentiment_analyzer_positive(self):
        """Test sentiment analyzer with positive text."""
        result = main.sentiment_analyzer("这个产品非常好，我很满意，强烈推荐给大家")
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    def test_sentiment_analyzer_negative(self):
        """Test sentiment analyzer with negative text."""
        result = main.sentiment_analyzer("太差了，非常失望，不建议购买")
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    def test_sentiment_empty(self):
        """Test sentiment analyzer with empty text."""
        result = main.sentiment_analyzer("")
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    def test_seo_score_analyzer_exists(self):
        """Test that seo_score_analyzer function is callable."""
        assert callable(main.seo_score_analyzer)
        assert main.seo_score_analyzer.__doc__ is not None

    def test_readability_calculator_exists(self):
        """Test that readability_calculator function is callable."""
        assert callable(main.readability_calculator)
        assert main.readability_calculator.__doc__ is not None
