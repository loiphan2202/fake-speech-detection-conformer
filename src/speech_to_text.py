"""
speech_to_text.py
-----------------
Chuyển đổi audio sang văn bản dùng Whisper (Hugging Face).
Hỗ trợ tiếng Việt, tiếng Anh và hơn 90 ngôn ngữ.
"""

from transformers import pipeline
import torch

_ASR_PIPELINE = {}   # cache theo model_name


def _get_asr(model_name: str):
    """Cache ASR pipeline để tránh load lại nhiều lần."""
    if model_name not in _ASR_PIPELINE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ASR model: {model_name} on {device}...")
        _ASR_PIPELINE[model_name] = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        print("✅ ASR model loaded")
    return _ASR_PIPELINE[model_name]


def transcribe_audio(
    audio_path: str,
    language: str = None,
    model_name: str = "openai/whisper-large-v3",
    chunk_length_s: int = 30,
) -> str:
    """
    Chuyển đổi audio sang text dùng Whisper.

    Args:
        audio_path    : Đường dẫn file audio (.wav/.flac/.mp3/.m4a)
        language      : Mã ngôn ngữ ISO 639-1 (vd: "vi", "en"), None = tự phát hiện
        model_name    : Whisper model từ Hugging Face Hub
        chunk_length_s: Độ dài mỗi chunk (giây) cho audio dài

    Returns:
        Chuỗi văn bản transcript
    """
    asr = _get_asr(model_name)

    kwargs = {
        "chunk_length_s": chunk_length_s,
        "batch_size": 8,
        "return_timestamps": False,
    }
    if language:
        kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}

    result = asr(audio_path, **kwargs)

    if isinstance(result, list):
        return " ".join([r["text"] for r in result]).strip()
    return result["text"].strip()


def transcribe_audio_fast(audio_path: str, language: str = None) -> str:
    """
    Phiên bản nhanh hơn dùng whisper-base (độ chính xác thấp hơn).
    Dùng cho preview hoặc testing.
    """
    return transcribe_audio(
        audio_path,
        language=language,
        model_name="openai/whisper-base",
    )
