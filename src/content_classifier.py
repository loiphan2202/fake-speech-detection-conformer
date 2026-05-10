"""
content_classifier.py
---------------------
Phân loại nội dung văn bản: Tin tức (news) hay Hội thoại (conversation).
Dùng Zero-Shot Classification với facebook/bart-large-mnli.
"""

from transformers import pipeline

_CLASSIFIER = None


def _get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        print("Loading content classifier (facebook/bart-large-mnli)...")
        _CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        print("✅ Content classifier loaded")
    return _CLASSIFIER


def classify_content(text: str) -> dict:
    """
    Phân loại văn bản là tin tức hay hội thoại thông thường.

    Args:
        text: Văn bản cần phân loại

    Returns:
        {
            "type"      : "news" | "conversation",
            "confidence": float (0–100),
            "label"     : str mô tả đầy đủ
        }
    """
    if not text or not text.strip():
        return {"type": "conversation", "confidence": 50.0, "label": "unknown (empty text)"}

    clf = _get_classifier()
    candidate_labels = [
        "news report or news broadcast",
        "casual conversation or personal dialogue",
    ]

    result     = clf(text[:512], candidate_labels)
    top_label  = result["labels"][0]
    top_score  = result["scores"][0] * 100
    content_type = "news" if "news" in top_label else "conversation"

    return {
        "type"      : content_type,
        "confidence": round(top_score, 2),
        "label"     : top_label,
    }
