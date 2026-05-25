"""
fact_checker.py
---------------
Kiểm tra thông tin (fact-checking) trong văn bản:
  1. Tìm kiếm tin tức liên quan qua Google News RSS
  2. Chạy model phân loại fake/real news (HuggingFace)
  3. Tổng hợp kết quả
"""

import feedparser
from urllib.parse import quote_plus
from transformers import pipeline

_FACT_PIPELINE = None


def _get_fact_pipeline():
    global _FACT_PIPELINE
    if _FACT_PIPELINE is None:
        print("Loading fact-check model (hamzab/roberta-fake-news-classification)...")
        _FACT_PIPELINE = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
        )
        print("✅ Fact-check model loaded")
    return _FACT_PIPELINE


def search_news(query: str, max_results: int = 5, lang: str = "vi") -> list:
    """
    Tìm kiếm tin tức liên quan qua Google News RSS.

    Args:
        query      : Cụm từ tìm kiếm
        max_results: Số lượng bài báo tối đa
        lang       : Mã ngôn ngữ ("vi" hoặc "en")

    Returns:
        Danh sách articles: [{title, link, published, summary}]
    """
    try:
        encoded = quote_plus(query)
        if lang == "vi":
            url = f"https://news.google.com/rss/search?q={encoded}&hl=vi&gl=VN&ceid=VN:vi"
        else:
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"

        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            articles.append({
                "title"    : entry.get("title", ""),
                "link"     : entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary"  : entry.get("summary", ""),
            })
        return articles
    except Exception as e:
        print(f"[WARN] Google News search failed: {e}")
        return []


def extract_key_claim(text: str, max_chars: int = 150) -> str:
    """Trích xuất câu chủ đề chính từ transcript để làm query tìm kiếm."""
    sentences = text.replace("\n", " ").split(". ")
    return sentences[0][:max_chars].strip() if sentences else text[:max_chars]


def fact_check_text(text: str, lang: str = "vi") -> dict:
    """
    Kiểm tra nội dung văn bản.

    Args:
        text: Văn bản cần kiểm tra (transcript từ audio)
        lang: Ngôn ngữ để tìm kiếm tin tức ("vi" hoặc "en")

    Returns:
        {
            "verdict"      : "REAL" | "FAKE" | "UNVERIFIED",
            "confidence"   : float (0–100),
            "model_verdict": str,
            "claim"        : str,
            "sources"      : list,
            "analysis"     : str,
        }
    """
    claim   = extract_key_claim(text)
    sources = search_news(claim, lang=lang)

    # Chạy fake-news classifier
    # Model yêu cầu format: <title> TITLE <content> CONTENT <end>
    clf       = _get_fact_pipeline()
    input_str = f"<title> {claim} <content> {text} <end>"
    result    = clf(input_str[:512])[0]
    label  = result["label"]
    score  = result["score"]

    # Chuẩn hóa nhãn
    is_fake = "fake" in label.lower()
    verdict = "FAKE" if is_fake else "REAL"
    if score < 0.65:
        verdict = "UNVERIFIED"

    # Tạo phần phân tích
    if sources:
        src_titles = "\n".join(f"  • {s['title']}" for s in sources[:3])
        analysis   = (
            f"Tìm thấy {len(sources)} bài báo liên quan:\n{src_titles}\n\n"
            f"Model phân loại: {label} (confidence: {score*100:.1f}%)"
        )
    else:
        analysis = (
            "Không tìm thấy bài báo liên quan trên Google News.\n"
            f"Model phân loại: {label} (confidence: {score*100:.1f}%)\n"
            "Khuyến nghị: Kiểm tra thông tin tại các nguồn báo chính thống."
        )

    return {
        "verdict"      : verdict,
        "confidence"   : round(score * 100, 2),
        "model_verdict": label,
        "claim"        : claim,
        "sources"      : sources,
        "analysis"     : analysis,
    }
