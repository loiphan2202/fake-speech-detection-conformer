import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from src.audio_processing import preprocess_audio, load_audio, audio_to_spectrogram, crop_or_pad, trim_audio, normalize_audio, AUDIO_LEN
from src.deepfake_detector import DeepfakeDetector
from src.speech_to_text import transcribe_audio
from src.content_classifier import classify_content
from src.fact_checker import fact_check_text
from src.conversation_analyzer import analyze_conversation

print("Khởi tạo hệ thống...")
# Khởi tạo detector ngay khi app start (sẽ tốn thời gian load architecture)
detector = DeepfakeDetector("models/ckpt.weights.h5")


def analyze_audio(audio_path: str, language: str = "auto"):
    if audio_path is None:
        return "Vui lòng upload file audio.", "", "", "", None

    lang = None if language == "auto" else language

    # ── 1. Deepfake detection ──────────────────────────────────────
    try:
        spec_img  = preprocess_audio(audio_path)
        df_result = detector.predict(spec_img)
    except Exception as e:
         return f"**Lỗi xử lý audio:** {e}", "", "", "", None

    # ── Spectrogram visualization ──────────────────────────────────
    try:
        audio = normalize_audio(crop_or_pad(trim_audio(load_audio(audio_path))))
        spec  = audio_to_spectrogram(audio)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma")
        ax.set_title("Mel-Spectrogram")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel frequency bins")
        plt.tight_layout()
    except Exception as e:
        print(f"Error plotting spectrogram: {e}")
        fig = None

    # ── 2. Transcript ──────────────────────────────────────────────
    try:
        transcript = transcribe_audio(audio_path, language=lang)
    except Exception as e:
        transcript = f"[LỖI TRANSCRIPT] {e}"

    if not transcript or transcript.startswith("[LỖI"):
        return format_voice_result(df_result), transcript, "", "", fig

    # ── 3. Content classification ──────────────────────────────────
    try:
        content = classify_content(transcript)
    except Exception as e:
        content = {"type": "error", "confidence": 0, "label": str(e)}

    # ── 4. Fact-check or conversation analysis ─────────────────────
    analysis_text = ""
    if content["type"] == "news":
        try:
            analysis_result = fact_check_text(transcript)
            analysis_text = (
                f"**📰 Kiểm tra thông tin (Fact-check)**\n\n"
                f"**Claim:** {analysis_result['claim']}\n\n"
                f"**Kết quả:** {analysis_result['verdict']} "
                f"(confidence: {analysis_result['confidence']}%)\n\n"
                f"**Phân tích:**\n{analysis_result['analysis']}"
            )
        except Exception as e:
             analysis_text = f"**Lỗi Fact-check:** {e}"
    elif content["type"] == "conversation":
        try:
            conv = analyze_conversation(transcript)
            analysis_text = (
                f"**💬 Phân tích hội thoại**\n\n"
                f"**Cảm xúc:** {conv['sentiment']} ({conv['sentiment_score']}%)\n\n"
                f"**Tóm tắt:** {conv['summary']}\n\n"
                f"**Chủ đề chính:** {', '.join(conv['topics'])}"
            )
        except Exception as e:
             analysis_text = f"**Lỗi phân tích hội thoại:** {e}"

    # ── Tổng hợp output ────────────────────────────────────────────
    voice_result   = format_voice_result(df_result)
    transcript_out = f"**📝 Transcript**\n\n{transcript}"
    
    if content["type"] != "error":
        content_out = (
            f"**🗂️ Loại nội dung:** {content['type'].upper()} "
            f"(confidence: {content['confidence']}%)"
        )
    else:
        content_out = "**🗂️ Lỗi phân loại nội dung**"

    return voice_result, transcript_out, content_out, analysis_text, fig


def format_voice_result(df_result: dict) -> str:
    return (
        f"**🎙️ Kết quả phát hiện deepfake**\n\n"
        f"**Verdict:** {df_result['verdict']}\n"
        f"**Score:** {df_result['score']} (0.0 = Real, 1.0 = Fake)\n"
        f"**Confidence:** {df_result['confidence']}%"
    )


# ── Gradio UI ──────────────────────────────────────────────────────
with gr.Blocks(title="Deepfake Speech Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Deepfake Speech Detection & Fact-Checking System")
    gr.Markdown(
        "Upload file audio để phát hiện giọng nói deepfake, "
        "transcript nội dung, và kiểm tra thông tin."
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        language    = gr.Dropdown(
            ["auto", "vi", "en", "zh", "ja", "ko", "fr", "de"],
            value="auto",
            label="Ngôn ngữ transcript",
        )

    btn = gr.Button("🔍 Phân tích", variant="primary")

    with gr.Row():
        voice_out   = gr.Markdown(label="Deepfake Detection")
        content_out = gr.Markdown(label="Loại nội dung")

    transcript_out = gr.Markdown(label="Transcript")
    analysis_out   = gr.Markdown(label="Fact-check / Conversation Analysis")
    spec_plot      = gr.Plot(label="Mel-Spectrogram")

    btn.click(
        fn=analyze_audio,
        inputs=[audio_input, language],
        outputs=[voice_out, transcript_out, content_out, analysis_out, spec_plot],
    )


if __name__ == "__main__":
    print("Khởi động Gradio server...")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
