"""
conversation_analyzer.py
------------------------
Phân tích hội thoại thông thường: sentiment analysis, tóm tắt nội dung và trích xuất chủ đề (keywords).
"""

from transformers import pipeline

_SENTIMENT = None
_SUMMARIZER = None


def _get_sentiment():
    global _SENTIMENT
    if _SENTIMENT is None:
        print("Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
        _SENTIMENT = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        print("✅ Sentiment model loaded")
    return _SENTIMENT


def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        print("Loading summarizer model (facebook/bart-large-cnn)...")
        _SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
        )
        print("✅ Summarizer model loaded")
    return _SUMMARIZER


def analyze_conversation(text: str) -> dict:
    """
    Phân tích hội thoại thông thường:
        - Cảm xúc (Sentiment)
        - Tóm tắt nội dung (Summarization)
        - Chủ đề chính (Keyword extraction)

    Args:
        text: Nội dung hội thoại (transcript)

    Returns:
        dict:
        {
            "sentiment"      : str,
            "sentiment_score": float (0-100),
            "summary"        : str,
            "topics"         : list[str],
        }
    """
    if not text or not text.strip():
        return {
            "sentiment"      : "Neutral",
            "sentiment_score": 50.0,
            "summary"        : "",
            "topics"         : [],
        }

    # 1. Sentiment analysis
    sent_model   = _get_sentiment()
    sent_result  = sent_model(text[:512])[0]
    sentiment    = sent_result["label"]
    sent_score   = round(sent_result["score"] * 100, 2)

    # 2. Tóm tắt nội dung
    summary = ""
    words = text.split()
    if len(words) > 30:
        summarizer = _get_summarizer()
        max_len = min(130, len(words) // 2)
        min_len = min(30, max_len - 10)
        try:
            summary_out = summarizer(
                text[:1024],
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )
            summary = summary_out[0]["summary_text"]
        except Exception as e:
            print(f"[WARN] Summarization failed: {e}")
            summary = text[:200] + "..."
    else:
        summary = text

    # 3. Trích xuất chủ đề (đơn giản qua đếm tần suất từ khóa)
    stopwords = {"và", "là", "của", "có", "trong", "the", "a", "is", "to", "of", "mà", "cho", "với", "những", "một", "các"}
    filtered_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 3]
    
    freq = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1
    
    topics = sorted(freq, key=freq.get, reverse=True)[:5]

    return {
        "sentiment"      : sentiment,
        "sentiment_score": sent_score,
        "summary"        : summary,
        "topics"         : topics,
    }
