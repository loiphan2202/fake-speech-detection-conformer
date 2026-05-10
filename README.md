# Deepfake Speech Detection with Conformer

> Phát hiện giọng nói deepfake, chuyển đổi âm thanh thành văn bản, và kiểm chứng thông tin bằng mô hình Conformer.

## Features

- **Deepfake Detection** — Phân loại audio thật/giả bằng Conformer + Mel-spectrogram
- **Speech-to-Text** — ASR với Whisper (hỗ trợ tiếng Việt, Anh, và 90+ ngôn ngữ)
- **Content Classification** — Phân loại nội dung: tin tức hay hội thoại
- **Fact-checking** — Tra cứu tin tức + phân loại fake/real news
- **Conversation Analysis** — Phân tích cảm xúc, tóm tắt, trích xuất chủ đề
- **Web UI** — Giao diện Gradio trực quan

## Architecture

```
Audio Upload
  |
  +---> Preprocessing (librosa)
  |       16kHz → trim silence → crop/pad 3.5s → normalize → Mel-spec [256,128,1]
  |
  +---> Conformer Model ───────────> Deepfake Verdict (Real/Fake + confidence)
  |
  +---> Whisper ASR ───────────────> Transcript
  |
  +---> Content Classifier ────────> News ───> Fact-check (Google News + RoBERTa)
  |                                 └───> Conversation ───> Sentiment + Summary + Topics
  |
  +---> Gradio Web UI ──────────────> Combined Result
```

## Installation

**Windows (recommended):**
```bash
setup_windows.bat
```

**Manual:**
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**Requirements:** Python 3.9~3.12, TensorFlow, PyTorch, CUDA (optional)

## Usage

**Web interface:**
```bash
python app.py
# Open http://localhost:7860
```

**CLI:**
```bash
python predict.py --audio path/to/audio.wav
```

## Pretrained Model

Đặt file `ckpt.h5` (Conformer weights) vào thư mục `models/`. Input shape: `[256, 128, 1]`, output: sigmoid score (0=Real, 1=Fake).

## Dataset

[ASVspoof 2019 Logical Access (LA)](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)

- Train: A01–A06 (known spoofing systems)
- Test:  A07–A19 (novel spoofing attacks)

## Results

| Split | Accuracy | F1-Score |
|-------|----------|----------|
| Validation (A01–A06) | 99.87% | 99.87% |
| Test (A07–A19) | 73.95% | 64.84% |

Domain shift giữa train và test là thách thức chính của bài toán deepfake detection.

## Project Structure

```
├── app.py                     # Gradio web interface
├── predict.py                 # CLI tool
├── setup_windows.bat          # One-click setup (Windows)
├── requirements.txt
├── models/ckpt.h5             # Pretrained weights (not included)
└── src/
    ├── audio_processing.py    # Preprocessing pipeline
    ├── deepfake_detector.py   # Conformer model
    ├── speech_to_text.py      # Whisper ASR
    ├── content_classifier.py  # News vs Conversation
    ├── fact_checker.py        # News search + fake news classifier
    └── conversation_analyzer.py # Sentiment + summary
```

## License

MIT
