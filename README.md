# 🎙️ Deepfake Speech Detection System

> Hệ thống AI phát hiện giọng nói deepfake, phân tích nội dung âm thanh và kiểm chứng thông tin sử dụng mô hình **Conformer** (Convolution-augmented Transformer).

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dataset](#dataset)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Pipeline xử lý âm thanh](#pipeline-xử-lý-âm-thanh)
- [API Reference](#api-reference)
- [Kết quả mô hình](#kết-quả-mô-hình)

---

## 🔍 Tổng quan

Dự án này xây dựng một hệ thống toàn diện để:

1. **Phát hiện giọng nói deepfake** — Phân loại audio là `Real` (thật) hay `Fake` (giả mạo) bằng mô hình Conformer được huấn luyện trên dataset ASVspoof 2019.
2. **Chuyển đổi âm thanh sang văn bản (ASR)** — Sử dụng Automatic Speech Recognition để transcript nội dung.
3. **Kiểm tra thông tin (Fact-checking)** — Dùng Hugging Face models và tìm kiếm tin tức để xác minh thông tin trong audio là tin thật hay tin giả.
4. **Phân tích nội dung** — Dù là tin tức hay hội thoại thông thường, hệ thống đều phân tích và đưa ra nhận xét.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 🔊 **Deepfake Detection** | Phát hiện giọng nói giả mạo bằng Mel-spectrogram + Conformer |
| 📝 **Speech-to-Text** | Transcript âm thanh sang văn bản (hỗ trợ tiếng Việt & tiếng Anh) |
| 📰 **News Fact-checking** | Phát hiện tin giả trong nội dung âm thanh qua Hugging Face |
| 💬 **Conversation Analysis** | Phân tích nội dung hội thoại thông thường |
| 📊 **Spectrogram Visualization** | Hiển thị Mel-spectrogram của audio đầu vào |
| ⚡ **Real-time Processing** | Xử lý nhanh với TensorFlow + GPU/CPU |

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER UPLOADS AUDIO                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AUDIO PREPROCESSING                            │
│  1. Load & Decode (librosa / soundfile)                         │
│  2. Resample → 16,000 Hz                                        │
│  3. Trim silence (epsilon=0.15)                                 │
│  4. Crop or Pad → 56,000 samples (3.5 giây)                     │
│  5. Normalize (zero-mean, unit-variance)                        │
└───────────┬─────────────────────────────────┬───────────────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────────────┐
│  SPECTROGRAM PIPELINE │         │      RAW AUDIO (for ASR)      │
│  Audio2Spec():        │         └───────────────┬───────────────┘
│  • STFT (nfft=2048)   │                         │
│  • Mel-scale (128 mel)│                         ▼
│  • dB-scale (top=80)  │         ┌───────────────────────────────┐
│  • Shape: [256, 128]  │         │   SPEECH-TO-TEXT (ASR)        │
│  • Reshape [256,128,1]│         │   openai/whisper-large-v3 or  │
└──────────┬────────────┘         │   facebook/wav2vec2-large     │
           │                      └───────────────┬───────────────┘
           ▼                                      │
┌───────────────────────┐                         ▼
│  CONFORMER MODEL      │         ┌───────────────────────────────┐
│  (Pretrained .h5)     │         │      TRANSCRIPT TEXT          │
│  → Sigmoid output     │         └───────────────┬───────────────┘
│  → Score [0.0, 1.0]   │                         │
└──────────┬────────────┘                         ▼
           │                      ┌───────────────────────────────┐
           ▼                      │  CONTENT CLASSIFICATION       │
┌───────────────────────┐         │  (News vs Conversation)       │
│  DEEPFAKE RESULT      │         │  facebook/bart-large-mnli     │
│  • Real / Fake        │         └───────────────┬───────────────┘
│  • Confidence score   │                         │
└───────────────────────┘          ┌──────────────┴──────────────┐
                                   ▼                              ▼
                        ┌──────────────────┐         ┌──────────────────────┐
                        │   NEWS PATH      │         │  CONVERSATION PATH   │
                        │ • Google News    │         │ • Sentiment Analysis │
                        │   search query  │         │ • Topic Extraction   │
                        │ • Fact-check    │         │ • Context Summary    │
                        │   (HuggingFace) │         └──────────────────────┘
                        │ • Real/Fake News│
                        │   verdict       │
                        └──────────────────┘
```

---

## 💻 Yêu cầu hệ thống

- Python >= 3.8
- TensorFlow >= 2.6
- CUDA 11.x (nếu dùng GPU) — khuyến nghị
- RAM >= 8 GB
- Disk >= 5 GB (cho models)

---

## ⚙️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/deepfake-speech-detection.git
cd deepfake-speech-detection
```

> **Lưu ý:** Đặt file `ckpt.h5` vào thư mục `models/` trước khi chạy.

### 2. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
# Core ML
tensorflow>=2.6.0
tensorflow-io==0.21.0
tensorflow-addons==0.15.0
tensorflow-probability==0.14.1

# Audio processing
librosa>=0.9.0
soundfile>=0.10.3
numpy>=1.21.0

# ASR & NLP (Hugging Face)
transformers>=4.30.0
datasets>=2.0.0
torch>=1.13.0
torchaudio>=0.13.0
sentencepiece>=0.1.97

# Fact-checking / Search
requests>=2.28.0
beautifulsoup4>=4.11.0
newspaper3k>=0.2.8
feedparser>=6.0.0

# Web app (nếu dùng Gradio)
gradio>=3.40.0

# Utilities
tqdm>=4.64.0
Pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.4.0
```

### 4. Tải pretrained model

Đặt file `ckpt.h5` vào thư mục `models/`:

```bash
mkdir -p models
cp /path/to/ckpt.h5 models/ckpt.h5
```

> File `ckpt.h5` là checkpoint đã được huấn luyện trên ASVspoof 2019, kiến trúc **Conformer** với input shape `[256, 128, 1]` (Mel-spectrogram).

---

## 📁 Cấu trúc dự án

```
deepfake-speech-detection/
│
├── models/
│   └── ckpt.h5                  # Pretrained Conformer weights
│
├── src/
│   ├── audio_processing.py      # Tiền xử lý âm thanh
│   ├── deepfake_detector.py     # Load model & dự đoán
│   ├── speech_to_text.py        # ASR transcript
│   ├── content_classifier.py    # Phân loại: News vs Conversation
│   ├── fact_checker.py          # Kiểm tra tin giả (Hugging Face)
│   └── conversation_analyzer.py # Phân tích hội thoại
│
├── app.py                       # Gradio web interface
├── predict.py                   # CLI prediction script
├── requirements.txt
└── README.md
```

---

## 🗄️ Dataset

Dự án sử dụng **ASVspoof 2019 Logical Access (LA)**:

| Dataset | Link |
|---------|------|
| Raw audio (FLAC) | [asvpoof-2019-dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) |
| TFRecord format | [asvspoof-2019-tfrecord-dataset](https://www.kaggle.com/datasets/awsaf49/asvspoof-2019-tfrecord-dataset) |

**Cấu trúc nhãn:**

| `class_name` | `target` | Ý nghĩa |
|-------------|---------|---------|
| `bonafide` | `0` | Giọng nói thật |
| `spoof` | `1` | Giọng nói giả mạo |

**Hệ thống giả mạo:** A01 – A19 (19 thuật toán TTS/VC khác nhau)

---

## 🚀 Hướng dẫn sử dụng

### Cách 1: Web Interface (Gradio)

```bash
python app.py
```

Truy cập `http://localhost:7860`, upload file audio và nhận kết quả phân tích.

### Cách 2: CLI

```bash
python predict.py --audio path/to/audio.wav
```

**Ví dụ output:**

```
=== DEEPFAKE SPEECH DETECTION RESULTS ===

[🎙️ VOICE AUTHENTICITY]
Verdict  : FAKE (Spoof)
Score    : 0.923 (1.0 = Fake, 0.0 = Real)
Confidence: 92.3%

[📝 TRANSCRIPT]
"Hôm nay chính phủ thông báo tăng lương tối thiểu lên 20% từ tháng 1 năm sau..."

[📰 CONTENT TYPE]
Type: News Content (Tin tức)
Confidence: 87.4%

[🔍 FACT-CHECK RESULT]
Claim    : "chính phủ thông báo tăng lương tối thiểu lên 20%"
Verdict  : UNVERIFIED / POTENTIALLY MISLEADING
Sources  : [No matching credible news found]
Analysis : Thông tin này chưa được xác minh từ các nguồn đáng tin cậy.
           Khuyến nghị kiểm tra tại các trang báo chính thống.
```

### Cách 3: Python API

```python
from src.audio_processing import preprocess_audio
from src.deepfake_detector import DeepfakeDetector
from src.speech_to_text import transcribe_audio
from src.fact_checker import FactChecker

# 1. Deepfake detection
detector = DeepfakeDetector(model_path="models/ckpt.h5")
spectrogram = preprocess_audio("audio.wav")
result = detector.predict(spectrogram)
print(f"Verdict: {'FAKE' if result > 0.5 else 'REAL'}, Score: {result:.3f}")

# 2. Transcription
transcript = transcribe_audio("audio.wav", language="vi")
print(f"Transcript: {transcript}")

# 3. Fact checking
checker = FactChecker()
analysis = checker.analyze(transcript)
print(analysis)
```

---

## 🔧 Pipeline xử lý âm thanh

Đây là phần quan trọng: phần tiền xử lý audio khi user upload phải **khớp chính xác** với cách model được huấn luyện.

### `src/audio_processing.py`

```python
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio


# ── Cấu hình (phải khớp với CFG trong notebook training) ──────────
SAMPLE_RATE  = 16000
DURATION     = 3.5               # giây
AUDIO_LEN    = int(SAMPLE_RATE * DURATION)   # 56,000 samples
SPEC_TIME    = 256               # trục thời gian của spectrogram
SPEC_FREQ    = 128               # trục tần số (mel bins)
N_FFT        = 2048
FMIN         = 20
FMAX         = SAMPLE_RATE // 2  # 8000 Hz
SPEC_SHAPE   = [SPEC_TIME, SPEC_FREQ]


def load_audio(file_path: str) -> np.ndarray:
    """
    Load audio từ file, resample về 16kHz, chuyển về mono.
    Hỗ trợ: .wav, .flac, .mp3, .ogg, .m4a
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def trim_audio(audio: np.ndarray, epsilon: float = 0.15) -> np.ndarray:
    """
    Cắt bỏ phần im lặng ở đầu và cuối.
    epsilon: ngưỡng biên độ để xác định im lặng.
    """
    audio_tf = tf.constant(audio)
    pos = tfio.audio.trim(audio_tf, axis=0, epsilon=epsilon)
    trimmed = audio[pos[0].numpy():pos[1].numpy()]
    return trimmed if len(trimmed) > 0 else audio


def crop_or_pad(audio: np.ndarray, target_len: int = AUDIO_LEN) -> np.ndarray:
    """
    Đảm bảo audio có đúng target_len samples.
    - Ngắn hơn → padding với zeros (ngẫu nhiên ở đầu/cuối).
    - Dài hơn  → crop ngẫu nhiên một đoạn.
    """
    audio_len = len(audio)
    if audio_len < target_len:
        diff = target_len - audio_len
        pad_left  = np.random.randint(0, diff + 1)
        pad_right = diff - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif audio_len > target_len:
        diff = audio_len - target_len
        start = np.random.randint(0, diff + 1)
        audio = audio[start:start + target_len]
    return audio.reshape(target_len)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa audio: zero-mean, unit-variance.
    """
    mean = np.mean(audio)
    std  = np.std(audio)
    if std > 0:
        audio = (audio - mean) / std
    return audio


def audio_to_spectrogram(
    audio: np.ndarray,
    spec_shape: list = SPEC_SHAPE,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Chuyển đổi audio thành Mel-spectrogram (dB scale).
    Pipeline khớp hoàn toàn với hàm Audio2Spec() trong notebook training:
        1. STFT  →  spectrogram
        2. Mel-scale (128 bins, 20–8000 Hz)
        3. dB-scale (top_db=80)
        4. Crop/reshape về spec_shape [256, 128]
    """
    audio_tf = tf.constant(audio, dtype=tf.float32)
    spec_time = spec_shape[0]
    spec_freq  = spec_shape[1]
    audio_len  = len(audio)
    hop_length = audio_len // (spec_time - 1)

    # STFT spectrogram
    spec = tfio.audio.spectrogram(audio_tf, nfft=n_fft,
                                  window=n_fft, stride=hop_length)
    # Mel-scale
    mel_spec = tfio.audio.melscale(spec, rate=sr,
                                   mels=spec_freq, fmin=fmin, fmax=fmax)
    # dB-scale
    db_mel_spec = tfio.audio.dbscale(mel_spec, top_db=80)

    # Đảm bảo đúng kích thước
    if db_mel_spec.shape[0] > spec_time:
        db_mel_spec = db_mel_spec[:spec_time, :]
    db_mel_spec = tf.reshape(db_mel_spec, spec_shape)

    return db_mel_spec.numpy()


def spectrogram_to_image(spec: np.ndarray) -> np.ndarray:
    """
    Chuyển spectrogram (H, W) → (H, W, 1) để feed vào model.
    """
    return spec[..., np.newaxis]


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Pipeline đầy đủ: load → trim → crop/pad → normalize → spectrogram → image.
    Trả về tensor shape [256, 128, 1] sẵn sàng cho model dự đoán.
    """
    audio = load_audio(file_path)
    audio = trim_audio(audio)
    audio = crop_or_pad(audio, target_len=AUDIO_LEN)
    audio = normalize_audio(audio)
    spec  = audio_to_spectrogram(audio)
    img   = spectrogram_to_image(spec)
    return img  # shape: [256, 128, 1]
```

---

### `src/deepfake_detector.py`

> **Quan trọng:** `ckpt.h5` là file **weights-only** (không phải full model). Phải rebuild architecture trước rồi mới gọi `load_weights()`.

```python
import numpy as np
import tensorflow as tf

# Shape cố định của model
SPEC_SHAPE  = [256, 128]   # [time, freq]
INPUT_SHAPE = (256, 128, 1)


def build_conformer_model(input_shape=INPUT_SHAPE, dropout_rate=0.0):
    """
    Rebuild lại đúng kiến trúc Conformer đã dùng khi training.
    Phải khớp 100% với notebook training (audio_classification_models).
    """
    import audio_classification_models as acm

    backbone = acm.ConformerModel(
        input_shape=input_shape,
        num_classes=1,
        include_top=False,
    )
    inp = tf.keras.Input(shape=input_shape)
    x   = backbone(inp)
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=out)


class DeepfakeDetector:
    """
    Load Conformer weights từ file .h5 và thực hiện dự đoán.
    Output: score trong [0, 1]  —  0.0 = Real, 1.0 = Fake
    """

    def __init__(self, model_path: str = "models/ckpt.h5"):
        # Rebuild architecture
        self.model = build_conformer_model()
        # Load weights (weights-only checkpoint)
        self.model.load_weights(model_path)
        print(f"✅ Model weights loaded from {model_path}")
        print(f"   Input shape  : {self.model.input_shape}")
        print(f"   Output shape : {self.model.output_shape}")

    def predict(self, spectrogram: np.ndarray) -> dict:
        """
        spectrogram: np.ndarray shape [256, 128, 1]
        Trả về dict gồm score, verdict, confidence.
        """
        inp   = np.expand_dims(spectrogram, axis=0).astype(np.float32)
        score = float(self.model.predict(inp, verbose=0)[0][0])

        verdict    = "FAKE (Spoof)"   if score >= 0.5 else "REAL (Bonafide)"
        confidence = score if score >= 0.5 else (1.0 - score)

        return {
            "score"     : round(score, 4),
            "verdict"   : verdict,
            "confidence": round(confidence * 100, 2),
            "is_fake"   : score >= 0.5,
        }

    def predict_from_file(self, audio_path: str) -> dict:
        """Tiện ích: nhận đường dẫn file, trả về kết quả."""
        from src.audio_processing import preprocess_audio
        spec = preprocess_audio(audio_path)
        return self.predict(spec)
```

---

### `src/speech_to_text.py`

```python
from transformers import pipeline
import torch


def transcribe_audio(
    audio_path: str,
    language: str = None,   # None = auto-detect, "vi" = tiếng Việt
    model_name: str = "openai/whisper-large-v3",
) -> str:
    """
    Chuyển đổi audio sang text dùng Whisper (Hugging Face).
    Hỗ trợ tiếng Việt, tiếng Anh và hơn 90 ngôn ngữ khác.

    Tham số:
        audio_path : đường dẫn file audio (.wav/.flac/.mp3)
        language   : mã ngôn ngữ ISO 639-1 (vd: "vi", "en"), None = tự phát hiện
        model_name : model Whisper từ Hugging Face Hub

    Trả về:
        Chuỗi văn bản transcript
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )

    kwargs = {}
    if language:
        kwargs["generate_kwargs"] = {"language": language}

    result = asr(audio_path, **kwargs)
    return result["text"].strip()
```

---

### `src/content_classifier.py`

```python
from transformers import pipeline

_CLASSIFIER = None

def _get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _CLASSIFIER


def classify_content(text: str) -> dict:
    """
    Phân loại văn bản là tin tức (News) hay hội thoại thông thường (Conversation).
    Dùng Zero-Shot Classification.

    Trả về:
        {
            "type"       : "news" | "conversation",
            "confidence" : float (0–100),
            "label"      : str mô tả
        }
    """
    clf = _get_classifier()
    candidate_labels = [
        "news report or news broadcast",
        "casual conversation or dialogue",
    ]
    result = clf(text, candidate_labels)

    top_label      = result["labels"][0]
    top_confidence = result["scores"][0] * 100
    content_type   = "news" if "news" in top_label else "conversation"

    return {
        "type"      : content_type,
        "confidence": round(top_confidence, 2),
        "label"     : top_label,
    }
```

---

### `src/fact_checker.py`

```python
import feedparser
import requests
from transformers import pipeline
from urllib.parse import quote_plus

_FACT_PIPELINE = None

def _get_fact_pipeline():
    global _FACT_PIPELINE
    if _FACT_PIPELINE is None:
        # Model phân loại tin thật/giả
        _FACT_PIPELINE = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
        )
    return _FACT_PIPELINE


def search_news(query: str, max_results: int = 5) -> list[dict]:
    """
    Tìm kiếm tin tức liên quan qua Google News RSS.

    Trả về danh sách các article: {title, link, published, summary}
    """
    encoded = quote_plus(query)
    url     = f"https://news.google.com/rss/search?q={encoded}&hl=vi&gl=VN&ceid=VN:vi"
    feed    = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:max_results]:
        articles.append({
            "title"    : entry.get("title", ""),
            "link"     : entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary"  : entry.get("summary", ""),
        })
    return articles


def extract_key_claim(text: str) -> str:
    """
    Trích xuất câu chủ đề chính từ transcript để làm query tìm kiếm.
    (Lấy câu đầu tiên, tối đa 100 ký tự)
    """
    sentences = text.replace("\n", " ").split(". ")
    return sentences[0][:150].strip() if sentences else text[:150]


def fact_check_text(text: str) -> dict:
    """
    Kiểm tra nội dung văn bản:
        1. Tìm tin tức liên quan (Google News RSS)
        2. Chạy model phân loại fake/real news
        3. Trả về kết quả tổng hợp

    Trả về:
        {
            "verdict"      : "REAL" | "FAKE" | "UNVERIFIED",
            "confidence"   : float,
            "model_verdict": str,
            "sources"      : list[dict],
            "analysis"     : str,
        }
    """
    claim   = extract_key_claim(text)
    sources = search_news(claim)

    # Chạy fake-news classifier trên transcript
    clf     = _get_fact_pipeline()
    result  = clf(text[:512])[0]   # giới hạn 512 tokens
    label   = result["label"]      # FAKE hoặc REAL
    score   = result["score"]

    # Chuẩn hoá nhãn
    verdict = "FAKE" if "fake" in label.lower() else "REAL"

    # Nếu confidence thấp → UNVERIFIED
    if score < 0.65:
        verdict = "UNVERIFIED"

    # Tạo phần phân tích
    if sources:
        src_titles = "\n".join(f"  - {s['title']}" for s in sources[:3])
        analysis   = (
            f"Hệ thống tìm thấy {len(sources)} bài báo liên quan:\n{src_titles}\n\n"
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
        "sources"      : sources,
        "analysis"     : analysis,
        "claim"        : claim,
    }
```

---

### `src/conversation_analyzer.py`

```python
from transformers import pipeline

_SENTIMENT = None
_SUMMARIZER = None

def _get_sentiment():
    global _SENTIMENT
    if _SENTIMENT is None:
        _SENTIMENT = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    return _SENTIMENT

def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
        )
    return _SUMMARIZER


def analyze_conversation(text: str) -> dict:
    """
    Phân tích hội thoại thông thường:
        - Cảm xúc (Sentiment)
        - Tóm tắt nội dung
        - Chủ đề chính (keyword extraction đơn giản)

    Trả về:
        {
            "sentiment" : str,
            "sentiment_score": float,
            "summary"   : str,
            "topics"    : list[str],
        }
    """
    # Sentiment
    sent_result  = _get_sentiment()(text[:512])[0]
    sentiment    = sent_result["label"]
    sent_score   = round(sent_result["score"] * 100, 2)

    # Tóm tắt (chỉ khi text đủ dài)
    summary = ""
    if len(text.split()) > 30:
        max_len = min(130, len(text.split()) // 2)
        min_len = min(30, max_len - 10)
        try:
            summary = _get_summarizer()(
                text[:1024],
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )[0]["summary_text"]
        except Exception:
            summary = text[:200] + "..."
    else:
        summary = text

    # Chủ đề đơn giản: lấy các từ xuất hiện nhiều (stopwords cơ bản)
    stopwords = {"và", "là", "của", "có", "trong", "the", "a", "is", "to", "of"}
    words     = [w.lower() for w in text.split() if w.lower() not in stopwords and len(w) > 3]
    freq      = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    topics = sorted(freq, key=freq.get, reverse=True)[:5]

    return {
        "sentiment"      : sentiment,
        "sentiment_score": sent_score,
        "summary"        : summary,
        "topics"         : topics,
    }
```

---

### `app.py` — Gradio Web Interface

```python
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from src.audio_processing import preprocess_audio, load_audio, audio_to_spectrogram, crop_or_pad, trim_audio, normalize_audio, AUDIO_LEN
from src.deepfake_detector import DeepfakeDetector
from src.speech_to_text import transcribe_audio
from src.content_classifier import classify_content
from src.fact_checker import fact_check_text
from src.conversation_analyzer import analyze_conversation

detector = DeepfakeDetector("models/ckpt.h5")


def analyze_audio(audio_path: str, language: str = "auto"):
    if audio_path is None:
        return "Vui lòng upload file audio.", "", "", "", None

    lang = None if language == "auto" else language

    # ── 1. Deepfake detection ──────────────────────────────────────
    spec_img  = preprocess_audio(audio_path)
    df_result = detector.predict(spec_img)

    # ── Spectrogram visualization ──────────────────────────────────
    audio = normalize_audio(crop_or_pad(trim_audio(load_audio(audio_path))))
    spec  = audio_to_spectrogram(audio)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Mel-Spectrogram")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bins")
    plt.tight_layout()

    # ── 2. Transcript ──────────────────────────────────────────────
    transcript = transcribe_audio(audio_path, language=lang)

    # ── 3. Content classification ──────────────────────────────────
    content = classify_content(transcript)

    # ── 4. Fact-check or conversation analysis ─────────────────────
    if content["type"] == "news":
        analysis_result = fact_check_text(transcript)
        analysis_text = (
            f"**📰 Kiểm tra thông tin (Fact-check)**\n\n"
            f"**Claim:** {analysis_result['claim']}\n\n"
            f"**Kết quả:** {analysis_result['verdict']} "
            f"(confidence: {analysis_result['confidence']}%)\n\n"
            f"**Phân tích:**\n{analysis_result['analysis']}"
        )
    else:
        conv = analyze_conversation(transcript)
        analysis_text = (
            f"**💬 Phân tích hội thoại**\n\n"
            f"**Cảm xúc:** {conv['sentiment']} ({conv['sentiment_score']}%)\n\n"
            f"**Tóm tắt:** {conv['summary']}\n\n"
            f"**Chủ đề chính:** {', '.join(conv['topics'])}"
        )

    # ── Tổng hợp output ────────────────────────────────────────────
    voice_result = (
        f"**🎙️ Kết quả phát hiện deepfake**\n\n"
        f"**Verdict:** {df_result['verdict']}\n"
        f"**Score:** {df_result['score']} (0.0 = Real, 1.0 = Fake)\n"
        f"**Confidence:** {df_result['confidence']}%"
    )

    transcript_out = f"**📝 Transcript**\n\n{transcript}"
    content_out    = (
        f"**🗂️ Loại nội dung:** {content['type'].upper()} "
        f"(confidence: {content['confidence']}%)"
    )

    return voice_result, transcript_out, content_out, analysis_text, fig


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
    demo.launch(share=False, server_port=7860)
```

---

### `predict.py` — CLI Script

```python
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
    detector = DeepfakeDetector("models/ckpt.h5")
    spec     = preprocess_audio(audio_path)
    df       = detector.predict(spec)
    print(f"\n[🎙️ VOICE AUTHENTICITY]")
    print(f"  Verdict    : {df['verdict']}")
    print(f"  Score      : {df['score']} (0.0=Real, 1.0=Fake)")
    print(f"  Confidence : {df['confidence']}%")

    if deepfake_only:
        return

    # 2. Transcript
    print("\n[📝 TRANSCRIPT] (đang xử lý...)")
    transcript = transcribe_audio(audio_path, language=language)
    print(f"  {transcript}")

    # 3. Content classification
    content = classify_content(transcript)
    print(f"\n[📰 CONTENT TYPE]")
    print(f"  Type       : {content['type'].upper()}")
    print(f"  Confidence : {content['confidence']}%")

    # 4. Fact-check or conversation
    if content["type"] == "news":
        print("\n[🔍 FACT-CHECK RESULT]")
        fc = fact_check_text(transcript)
        print(f"  Claim   : {fc['claim']}")
        print(f"  Verdict : {fc['verdict']} (confidence: {fc['confidence']}%)")
        print(f"  Analysis:\n    {fc['analysis']}")
    else:
        print("\n[💬 CONVERSATION ANALYSIS]")
        conv = analyze_conversation(transcript)
        print(f"  Sentiment : {conv['sentiment']} ({conv['sentiment_score']}%)")
        print(f"  Summary   : {conv['summary']}")
        print(f"  Topics    : {', '.join(conv['topics'])}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Speech Detection CLI")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default=None, help="Language code (vi, en, ...) or None for auto")
    parser.add_argument("--deepfake-only", action="store_true", help="Only run deepfake detection, skip ASR")
    args = parser.parse_args()
    run_analysis(args.audio, language=args.language, deepfake_only=args.deepfake_only)
```

---

## 📡 API Reference

### `preprocess_audio(file_path) → np.ndarray`
Toàn bộ pipeline tiền xử lý: load → trim → crop/pad → normalize → spectrogram.
Trả về shape `[256, 128, 1]`.

### `DeepfakeDetector.predict(spectrogram) → dict`
```json
{
  "score": 0.923,
  "verdict": "FAKE (Spoof)",
  "confidence": 92.3,
  "is_fake": true
}
```

### `transcribe_audio(audio_path, language) → str`
Trả về chuỗi transcript.

### `classify_content(text) → dict`
```json
{
  "type": "news",
  "confidence": 87.4,
  "label": "news report or news broadcast"
}
```

### `fact_check_text(text) → dict`
```json
{
  "verdict": "FAKE",
  "confidence": 81.2,
  "claim": "chính phủ tăng lương tối thiểu 20%",
  "sources": [...],
  "analysis": "..."
}
```

### `analyze_conversation(text) → dict`
```json
{
  "sentiment": "Neutral",
  "sentiment_score": 74.3,
  "summary": "...",
  "topics": ["chính trị", "kinh tế", ...]
}
```

---

## 📊 Kết quả mô hình

Model **Conformer** được huấn luyện trên **ASVspoof 2019 LA** (5,000 mẫu, 12 epochs):

### Tập Validation (A01–A06 — cùng hệ thống với train)

| Metric | Score |
|--------|-------|
| Loss | 0.1772 |
| Accuracy | 99.87% |
| Precision | 99.95% |
| Recall | 99.80% |
| **F1-Score** | **99.87%** |

### Tập Test (A07–A19 — hệ thống **chưa thấy** khi train)

| Metric | Score |
|--------|-------|
| Loss | 2.6068 |
| Accuracy | 73.95% |
| Precision | 99.69% |
| Recall | 48.05% |
| **F1-Score** | **64.84%** |

> ⚠️ **Lưu ý:** Khoảng cách lớn giữa Valid và Test phản ánh vấn đề **domain shift** — tập Test chứa các hệ thống giả mạo mới (A07–A19) chưa xuất hiện trong tập Train (A01–A06). Đây là thử thách chính của bài toán phát hiện giọng nói deepfake trong thực tế.

**Các hệ thống giả mạo trong dataset:** A01–A19 bao gồm TTS (Text-to-Speech) và VC (Voice Conversion).

**Hướng cải thiện:** Bật data augmentation, tăng dataset (full 25k mẫu), dùng Focal Loss thay Binary CE.

---

## ⚠️ Lưu ý quan trọng

1. **Preprocessing phải khớp với training** — Mọi thông số (`sample_rate=16000`, `duration=3.5s`, `n_fft=2048`, `spec_shape=[256,128]`) phải giống hệt với cấu hình `CFG` trong notebook huấn luyện.
2. **Ngưỡng phân loại** — Score `>= 0.5` → FAKE, `< 0.5` → REAL.
3. **Fact-checking** — Kết quả chỉ mang tính tham khảo. Luôn kiểm tra thêm tại các nguồn báo chính thống.
4. **Transcript tiếng Việt** — Whisper large-v3 cho độ chính xác cao nhất với tiếng Việt.

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Credits

- **Model:** Conformer (Google) — [paper](https://arxiv.org/pdf/2005.08100.pdf)
- **Dataset:** ASVspoof 2019 — [awsaf49](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)
- **ASR:** OpenAI Whisper via Hugging Face
- **Fact-check model:** `hamzab/roberta-fake-news-classification`
- **NLP:** `facebook/bart-large-mnli`, `facebook/bart-large-cnn`
