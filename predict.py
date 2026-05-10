import argparse
import sys
from src.audio_processing import preprocess_audio
from src.deepfake_detector import DeepfakeDetector
from src.speech_to_text import transcribe_audio
from src.content_classifier import classify_content
from src.fact_checker import fact_check_text
from src.conversation_analyzer import analyze_conversation


def run_analysis(audio_path: str, language: str = None, deepfake_only: bool = False):
    print("\n" + "=" * 50)
    print("  DEEPFAKE SPEECH DETECTION RESULTS")
    print("=" * 50)

    # 1. Deepfake detection
    try:
        detector = DeepfakeDetector("models/ckpt.h5")
        spec     = preprocess_audio(audio_path)
        df       = detector.predict(spec)
        print(f"\n[🎙️ VOICE AUTHENTICITY]")
        print(f"  Verdict    : {df['verdict']}")
        print(f"  Score      : {df['score']} (0.0=Real, 1.0=Fake)")
        print(f"  Confidence : {df['confidence']}%")
    except Exception as e:
        print(f"\n[🎙️ VOICE AUTHENTICITY ERROR] {e}")
        return

    if deepfake_only:
        print("\n" + "=" * 50)
        return

    # 2. Transcript
    print("\n[📝 TRANSCRIPT] (đang xử lý...)")
    try:
        transcript = transcribe_audio(audio_path, language=language)
        print(f"  {transcript}")
    except Exception as e:
        print(f"  Error transcribing audio: {e}")
        return

    if not transcript:
        print("\n  [INFO] Không có nội dung để phân tích thêm.")
        return

    # 3. Content classification
    print("\n[📰 CONTENT TYPE]")
    try:
        content = classify_content(transcript)
        print(f"  Type       : {content['type'].upper()}")
        print(f"  Confidence : {content['confidence']}%")
    except Exception as e:
        print(f"  Error classifying content: {e}")
        return

    # 4. Fact-check or conversation
    if content["type"] == "news":
        print("\n[🔍 FACT-CHECK RESULT]")
        try:
            fc = fact_check_text(transcript)
            print(f"  Claim   : {fc['claim']}")
            print(f"  Verdict : {fc['verdict']} (confidence: {fc['confidence']}%)")
            print(f"  Analysis:\n    {fc['analysis'].replace(chr(10), chr(10) + '    ')}")
        except Exception as e:
            print(f"  Error fact checking: {e}")
    else:
        print("\n[💬 CONVERSATION ANALYSIS]")
        try:
            conv = analyze_conversation(transcript)
            print(f"  Sentiment : {conv['sentiment']} ({conv['sentiment_score']}%)")
            print(f"  Summary   : {conv['summary']}")
            print(f"  Topics    : {', '.join(conv['topics'])}")
        except Exception as e:
            print(f"  Error analyzing conversation: {e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Speech Detection CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default=None, help="Language code (vi, en, ...) or None for auto")
    parser.add_argument("--deepfake-only", action="store_true", help="Only run deepfake detection, skip ASR")
    args = parser.parse_args()
    
    run_analysis(args.audio, language=args.language, deepfake_only=args.deepfake_only)
