# README2 — Deepfake Speech Detection System (Giải thích chi tiết từ công thức đến triển khai)

> Tài liệu này giải thích **từng công thức toán học**, **từng dòng code**, và **từng quyết định thiết kế** trong toàn bộ hệ thống.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Dataset ASVspoof 2019](#2-dataset-asvspoof-2019)
3. [Audio Preprocessing — Công thức chi tiết](#3-audio-preprocessing--công-thức-chi-tiết)
4. [Huấn luyện (Training Pipeline)](#4-huấn-luyện-training-pipeline)
5. [Kiến trúc Conformer — Giải thích từng module](#5-kiến-trúc-conformer--giải-thích-từng-module)
6. [Source Code — Giải thích chi tiết](#6-source-code--giải-thích-chi-tiết)
7. [Speech-to-Text với Whisper](#7-speech-to-text-với-whisper)
8. [NLP Downstream Tasks](#8-nlp-downstream-tasks)
9. [Luồng thực thi (Execution Flow)](#9-luồng-thực-thi-execution-flow)
10. [Kết quả & Metrics](#10-kết-quả--metrics)

---

## 1. Tổng quan hệ thống

Hệ thống gồm **2 phase** chính:

### Phase 1: Training (xảy ra trên Kaggle)
```
Raw Audio (FLAC) → Mel-Spectrogram → Conformer Encoder → Binary Classifier (Real/Fake)
                               ↑
                    ASVspoof 2019 Dataset
```

### Phase 2: Inference (xảy ra trên máy local / server)
```
User Audio → Spectrogram → Conformer → Deepfake Score
                          ↘ Whisper ASR → Transcript
                                        → Content Classifier (News/Conversation)
                                          → Fact-checker OR Conversation Analyzer
```

**Kiến trúc tổng thể:**

```
                    ┌──────────────────────┐
                    │   Mel-Spectrogram    │  shape [256, 128, 1]
                    │   [time, freq, ch]   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Conformer Encoder    │  8× ConformerBlock
                    │  (ConvSubsampling +   │  6.7M params
                    │   8× ConformerBlock)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  GlobalAvgPooling1D   │  [64,144] → [144]
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Dense(1, sigmoid)    │  score ∈ [0, 1]
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Threshold at 0.5     │
                    │  ≥0.5 → FAKE         │
                    │  <0.5 → REAL         │
                    └──────────────────────┘
```

---

## 2. Dataset ASVspoof 2019

### 2.1 Bài toán

Cho một đoạn audio x, hãy xác định:
- **Bonafide (Real — 0)**: Giọng nói thật của con người
- **Spoof (Fake — 1)**: Giọng nói được tạo ra bằng máy

### 2.2 19 hệ thống spoofing (A01–A19)

Đây là các thuật toán TTS/VC được dùng để tạo fake audio:

| Loại | Hệ thống | Kỹ thuật |
|------|---------|----------|
| **TTS** (Text-to-Speech) | A01, A02, A07, A08, A14, A15, A18, A19 | WaveNet, Tacotron, Vocoder |
| **VC** (Voice Conversion) | A03–A06, A09–A13, A16, A17 | Spectral filtering, neural VC |

**Tại sao test accuracy thấp hơn valid nhiều?** Vì test dùng **A07–A19** — những hệ thống spoofing **không xuất hiện trong train**. Đây là **domain shift** có chủ đích, mô phỏng tình huống gặp phải deepfake từ công nghệ mới.

### 2.3 Cân bằng dữ liệu

```python
train_df = train_df.groupby(['target']).sample(2500)
```

Mỗi class lấy đúng 2500 mẫu, tổng 5000. Mục đích: tránh bias — nếu 80% là real, model có thể đạt 80% accuracy bằng cách luôn đoán "real".

---

## 3. Audio Preprocessing — Công thức chi tiết

### 3.1 Load audio với librosa

```python
audio, sr = librosa.load(file_path, sr=16000, mono=True)
```

**Công thức nội suy resample:**
Librosa dùng **band-limited sinc interpolation** (lọc thông thấp + nội suy) để chuyển sample rate gốc → 16kHz.

**Tại sao 16kHz?**
- Giọng nói con người: ~300Hz–8kHz
- Theo Nyquist: `fs ≥ 2 × f_max` → 8kHz × 2 = 16kHz
- Dư địa cho harmonic đến bậc 4 (4 × 300Hz × 2 = 2.4kHz → thoải mái trong 8kHz)

### 3.2 Trim silence

```python
def trim_audio(audio, epsilon=0.15):
    top_db = -20 * np.log10(epsilon)  # ≈ 16.48 dB
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
```

**Công thức năng lượng:**
```
E(t) = 10 * log10( Σ x[i]² / N )   (dB)
```

Với `epsilon = 0.15`:
```
threshold = -20 * log10(0.15) = -20 * (-0.8239) = 16.48 dB
```

Ý nghĩa: Cắt bỏ phần có năng lượng < 16.48 dB so với peak energy. Loại bỏ khoảng lặng đầu/cuối.

**Tại sao cần trim?**
- Giọng nói deepfake thường có khoảng lặng khác thường (quá dài hoặc quá ngắn)
- Trim giúp model tập trung vào phần speech, không bị nhiễu bởi silence pattern

### 3.3 Crop or Pad — căn chỉnh độ dài

```python
AUDIO_LEN = int(16000 * 3.5) = 56000 samples
```

**Nếu audio ngắn hơn 56000 samples → Pad:**
```
x_padded[n] = { x[n - pad_left]  nếu pad_left ≤ n < pad_left + len(x)
              { 0                 nếu ngược lại
```
Pad vị trí **ngẫu nhiên** (random left padding), không pad về 0 cả 2 phía.

**Nếu audio dài hơn 56000 samples → Crop:**
```
x_cropped[n] = x[start + n]  với n = 0..55999
start ~ Uniform(0, len(x) - 56000)
```

Crop vị trí **ngẫu nhiên**. Mục đích:
- Tạo data augmentation tự nhiên (mỗi epoch lấy 1 đoạn khác nhau)
- Giúp model robust với alignment

**Tại sao 3.5 giây?**
- Câu nói tiếng Anh trung bình: 2–4 giây
- Đủ dài để chứa đặc trưng giọng nói
- Đủ ngắn để batch size 32 vừa GPU memory

### 3.4 Normalize

```python
def normalize_audio(audio):
    mean = np.mean(audio)
    std  = np.std(audio)
    if std > 0:
        audio = (audio - mean) / std
    return audio
```

**Công thức chuẩn hóa Z-score:**
```
x_norm[n] = (x[n] - μ) / σ

với μ = (1/N) * Σ x[n]
     σ = sqrt( (1/N) * Σ (x[n] - μ)² )
```

Kết quả: `mean=0, std=1`. Mục đích:
- Đồng nhất dynamic range giữa các file audio (có file to, file nhỏ)
- Giúp gradient descent hội tụ nhanh hơn (tránh vanishing/exploding gradient)

### 3.5 STFT (Short-Time Fourier Transform)

```python
hop_length = audio_len // (spec_time - 1)
            = 56000 // 255 ≈ 219
```

**Công thức STFT:**
```
X(k, t) = Σ x[n] · w[n - t·h] · e^(-j·2π·k·n/N)

với:
- x[n]     : audio signal tại sample n
- w[n]     : window function (Hann window mặc định)
- h        : hop_length = 219 samples
- N        : n_fft = 2048
- k        : frequency bin (0..N/2 = 1024)
- t        : time frame index
```

Số time frames:
```
T = 1 + floor((len(x) - N) / h)
  = 1 + floor((56000 - 2048) / 219)
  = 1 + floor(246.35)
  = 247 (sau đó pad/crop → 256)
```

**Tại sao hop_length = 219?**
```
hop_length = 56000 / (256 - 1) = 56000 / 255 ≈ 219.6 → 219
```
Để sau khi STFT, số time frame ≈ spec_time = 256.

**Overlap giữa các frame:**
```
overlap = N - hop_length = 2048 - 219 = 1829 samples (~89%)
```
Overlap lớn → time resolution cao (mượt mà). Ở 16kHz, mỗi frame cách nhau `219/16000 ≈ 13.7ms`.

### 3.6 Mel Filterbank

```python
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
    hop_length=hop_length, n_mels=spec_freq, fmin=fmin, fmax=fmax)
```

**Công thức Mel scale (chuyển Hz → Mel):**
```
m = 2595 * log10(1 + f / 700)
```

Với `fmin=20Hz, fmax=8000Hz`:
```
m_min = 2595 * log10(1 + 20/700)   ≈ 31.8 Mel
m_max = 2595 * log10(1 + 8000/700) ≈ 2840.0 Mel
```

Chia đều 128 bins trên thang Mel:
```
m_i = m_min + (i + 0.5) × (m_max - m_min) / 128   với i = 0..127
```

Chuyển ngược Mel → Hz cho từng bin:
```
f_i = 700 × (10^(m_i / 2595) - 1)
```

**Triangular filter H_i[k]:**
Mỗi filter: 1 tam giác, đỉnh tại `f_i`, bằng 0 tại `f_{i-1}` và `f_{i+1}`:
```
H_i[k] = { 0                              k < f_{i-1}
         { (k - f_{i-1}) / (f_i - f_{i-1})  f_{i-1} ≤ k ≤ f_i
         { (f_{i+1} - k) / (f_{i+1} - f_i)  f_i ≤ k ≤ f_{i+1}
         { 0                              k > f_{i+1}
```

**Công thức Mel-spectrogram:**
```
MelSpec(t, i) = Σ |X(k, t)|² · H_i[k]
                k
```

128 Mel filters × 256 time frames → shape [256, 128] sau khi transpose.

**Tại sao dùng Mel thay vì linear frequency?**
- Tai người cảm nhận tần số theo thang log (Mel scale)
- Mel tập trung nhiều bins ở tần số thấp (nơi chứa formants — đặc trưng giọng nói)
- Giảm chiều dữ liệu: 1024 FFT bins → 128 Mel bins (giảm 8 lần)

### 3.7 Power to dB

```python
db_mel_spec = librosa.power_to_db(mel_spec, top_db=80)
```

**Công thức:**
```
dB_mel = 10 * log10(MelSpec / ref)

Clamp: dB_mel = max(dB_mel, max_value - 80)
```

Trong code, `ref` mặc định là `max(mel_spec)`:
```
ref = max(MelSpec)  → normalize to 0 dB at peak
dB_mel = 10 * log10(MelSpec / max(MelSpec))
threshold = max(dB_mel) - 80 = -80 dB

→ output range: [-80, 0] dB
```

**Tại sao cần dB scale?**
- Dynamic range của spectrogram rất lớn (gấp 10^6 lần)
- dB nén range về [-80, 0], dễ học hơn cho neural network

### 3.8 Reshape về [256, 128, 1]

```python
img = spec[..., np.newaxis]  # [256, 128] → [256, 128, 1]
```

Thêm channel dimension vì Conv2D yêu cầu input shape `(H, W, C)`.

---

## 4. Huấn luyện (Training Pipeline)

### 4.1 Cấu hình (CFG)

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `duration = 3.5` | 3.5s | Độ dài mỗi mẫu sau crop/pad |
| `audio_len = 56000` | samples | `16000 × 3.5` |
| `spec_shape = [256, 128]` | time × freq bins | 256 frames × 128 Mel bins |
| `time_mask = 40` | frames | SpecAugment: mask tối đa 40 time steps |
| `freq_mask = 20` | bins | SpecAugment: mask tối đa 20 freq bins |
| `batch_size = 32` | — | 32 mẫu / batch |
| `epochs = 12` | — | 12 epoch (5000/32 ≈ 156 steps/epoch, total ~1872 steps) |
| `lr = 1e-4` | — | Learning rate khởi tạo |
| `fake_weight = 3.0` | — | Trọng số cho lớp fake trong loss |

### 4.2 Cosine Learning Rate Schedule

```python
lrfn: epoch → lr
```

**3 phase:**

**Phase 1: Warmup (epoch 0 → 3):**
```
lr(e) = lr_min + (lr_max - lr_min) × (e / warmup_epochs)

với lr_min = 5e-5, lr_max = 5e-4
```

Tuyến tính tăng từ `5e-5` lên `5e-4` trong 3 epoch đầu.

**Phase 2: Cosine decay (epoch 3 → 12):**
```
lr(e) = lr_min + (lr_max - lr_min) × 0.5 × (1 + cos((e - warmup) × π / (epochs - warmup)))

với:
- total_steps = epochs - warmup = 9
- progress = (e - warmup) / total_steps
- cosine_factor = 0.5 × (1 + cos(progress × π))
```

Giá trị theo epoch:
```
epoch 0: 5.00e-5   (start warmup)
epoch 1: 2.00e-4   (warmup)
epoch 2: 3.50e-4   (warmup)
epoch 3: 5.00e-4   → cos(0) = 1.0 → 5.00e-4 (peak)
epoch 4: 4.82e-4   cos(π/9)
epoch 5: 4.33e-4   cos(2π/9)
epoch 6: 3.60e-4   cos(3π/9)
epoch 7: 2.75e-4   cos(4π/9)
epoch 8: 1.91e-4   cos(5π/9)
epoch 9: 1.19e-4   cos(6π/9)
epoch 10: 6.70e-5  cos(7π/9)
epoch 11: 2.83e-5  cos(8π/9)
epoch 12: 1.00e-5  cos(π) = -1.0 → 1.00e-5 (trough)
```

**Tại sao cosine schedule?**
- Warmup tránh "early divergence" (lr đột ngột cao ngay từ đầu)
- Cosine decay giảm lr từ từ, cho phép hội tụ đến local minimum tốt hơn so với step decay

### 4.3 Weighted Binary Cross-Entropy Loss

**Công thức BCE thông thường:**
```
BCE(y, ŷ) = -[y × log(ŷ) + (1 - y) × log(1 - ŷ)]

với:
- y  ∈ {0, 1}: ground truth (0=Real, 1=Fake)
- ŷ ∈ [0, 1]: prediction (score)
```

**Weighted BCE:**
```
WeightedBCE(y, ŷ) = BCE(y, ŷ) × (1 + (fake_weight - 1) × y)

Với y=0 (Real):    weight = 1.0 + (3.0 - 1.0) × 0 = 1.0
Với y=1 (Fake):    weight = 1.0 + (3.0 - 1.0) × 1 = 3.0
```

**Ví dụ cụ thể:**
```
Giả sử model dự đoán sai một mẫu:
- Mẫu Fake (y=1), model đoán ŷ=0.3 (nghĩ là Real)
  BCE = -[1×log(0.3) + 0×log(0.7)] = 1.204
  Weighted = 1.204 × 3.0 = 3.612 ← phạt nặng
  
- Mẫu Real (y=0), model đoán ŷ=0.7 (nghĩ là Fake)
  BCE = -[0×log(0.3) + 1×log(0.3)] = 1.204
  Weighted = 1.204 × 1.0 = 1.204 ← phạt nhẹ hơn
```

**Tại sao fake_weight = 3.0?**
Trong thực tế, **false negative** (bỏ sót deepfake) nguy hiểm hơn **false positive** (gán nhãn sai cho giọng thật). Một deepfake có thể được dùng để:
- Giả mạo CEO ra lệnh chuyển tiền
- Phát tán tin giả dưới danh nghĩa người nổi tiếng

Nên ta muốn model **thiên về phát hiện deepfake**, chấp nhận false positive còn hơn false negative.

### 4.4 TFRecord Data Pipeline

**Cấu trúc TFRecord Example:**
```protobuf
message Example {
  Features features = 1;
}

message Features {
  map<string, Feature> feature = 1;
}

// Mỗi feature là:
message Feature {
  oneof kind {
    BytesList   bytes_list   = 1;
    FloatList   float_list   = 2;
    Int64List   int64_list   = 3;
  }
}
```

**Cách parse TFRecord:**
```python
def read_tfrecord(example):
    tfrec_format = {
        "audio"     : tf.io.FixedLenFeature([], tf.string),
        "target"    : tf.io.FixedLenFeature([], tf.int64),
        ...
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    audio = tf.io.decode_wav(example["audio"])  # WAV → tensor
    ...
```

**Pipeline execution order (tối ưu hóa):**

```
File          → TFRecordDataset    → đọc từ disk (I/O bound)
TFRecord      → cache()            → lưu vào RAM sau lần đầu
Cache         → shuffle(1024)      → trộn ngẫu nhiên
Shuffle       → repeat()           → lặp vô hạn
Repeat        → map(read_tfrecord) → parse + augment (CPU bound)
Parse         → batch(32)          → gom batch (CPU bound)
Batch         → map(MixUp)         → augmentation (CPU bound)
MixUp         → prefetch(AUTO)     → pipeline song song
```

**Tại sao `repeat()` trước `shuffle()`?**
`repeat()` tạo infinite stream. Nếu shuffle trước repeat, chỉ shuffle trong 1 epoch. Nếu repeat trước shuffle, shuffle **xuyên epoch** → data mixing tốt hơn.

### 4.5 SpecAugment (Time/Freq Masking)

Lấy cảm hứng từ [SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779).

**Time masking:**
```
Với mỗi mẫu, chọn ngẫu nhiên t ∈ [0, time_mask_param)
và vị trí bắt đầu t0 ∈ [0, T - t)
Đặt MelSpec[:, t0:t0+t] = 0
Với time_mask_param = 40, T = 256 → mask tối đa 15.6% time axis
```

**Frequency masking:**
```
Chọn ngẫu nhiên f ∈ [0, freq_mask_param)
và f0 ∈ [0, F - f)
Đặt MelSpec[f0:f0+f, :] = 0
Với freq_mask_param = 20, F = 128 → mask tối đa 15.6% freq axis
```

**Tại sao dùng SpecAugment?**
- Chống overfitting: model không thể dựa vào 1 vùng spectrogram duy nhất
- Robust: giống như "che tai" khi nghe — vẫn phải hiểu được nội dung
- Time masking mô phỏng mất tín hiệu tạm thời
- Freq masking mô phỏng nhiễu băng tần

### 4.6 MixUp Augmentation

**Công thức (Zhang et al., 2018):**
```
λ ~ Beta(α, α) với α = 0.2

x_mix = λ × x_1 + (1-λ) × x_2
y_mix = λ × y_1 + (1-λ) × y_2

với x_1, x_2: 2 spectrogram khác nhau
     y_1, y_2: one-hot labels (0 hoặc 1)
```

**Ví dụ:**
```
x_1 = spectrogram của "Real", label y_1 = 0
x_2 = spectrogram của "Fake", label y_2 = 1
λ = 0.7

x_mix = 0.7 × x_1 + 0.3 × x_2
y_mix = 0.7 × 0 + 0.3 × 1 = 0.3
→ Model học để predict 0.3 (hơi nghiêng về Real)
```

**Tại sao dùng MixUp?**
- Tạo vô hạn mẫu training mới từ tổ hợp tuyến tính
- Model học decision boundary mượt hơn, generalize tốt hơn
- Giảm overfitting

### 4.7 CutMix Augmentation

**Công thức (Yun et al., 2019):**
```
λ ~ Beta(α, α)

Tạo mask M ∈ {0, 1}^H×W, với vùng hình chữ nhật (rx, ry, rw, rh) được cắt

x_cutmix = (1 - M) ⊙ x_1 + M ⊙ x_2
y_cutmix = (1 - λ_area) × y_1 + λ_area × y_2

với λ_area = rw × rh / (H × W): tỷ lệ diện tích vùng cắt
```

**Khác biệt với MixUp:**
- MixUp: trộn toàn bộ ảnh theo tỷ lệ λ
- CutMix: cắt 1 vùng từ ảnh 2 dán vào ảnh 1

---

## 5. Kiến trúc Conformer — Giải thích từng module

### 5.1 Tổng quan

**Conformer = Convolution + Transformer**

Conformer (Gulati et al., 2020) là kiến trúc được thiết kế cho speech recognition, kết hợp:
- **CNN:** Capture local patterns (formants, spectral transitions)
- **Self-Attention:** Capture global dependencies (ngữ cảnh dài)

**Tại sao Conformer phù hợp cho deepfake detection?**
1. Deepfake để lại artifact trong spectrogram ở local scale (convolution bắt được)
2. Cấu trúc câu nói (global) cũng khác nhau giữa người thật và máy (attention bắt được)
3. Pretrain trên LibriSpeech 1000h speech → model đã có hiểu biết âm thanh

### 5.2 ConvSubsampling — Giảm chiều không gian

```python
class ConformerConvSubsampling:
    conv1: Conv2D(144, kernel=3, stride=2)  # 256×128 → 128×64
    conv2: Conv2D(144, kernel=3, stride=2)  # 128×64 → 64×32
    reshape: [B, 64, 4608]
    linear: Dense(144)                       # 4608 → 144
```

**Tại sao cần subsampling?**
- Input spectrogram [256, 128, 1] quá lớn để đưa vào self-attention (O(T²) complexity)
- Giảm time dimension 256 → 64 (giảm 4×)
- 144 channels là `d_model` — kích thước feature vector cho mỗi time step

**Công thức convolution:**
```
Output(t, f, c) = ReLU( Σ Σ Σ Input(t·2 + i, f·2 + j, k) × W(i, j, k, c) + bias )
                    i j k

với kernel W ∈ ℝ³ˣ³ˣ¹ˣ¹⁴⁴, stride = 2
```

### 5.3 ConformerBlock — Module chính

Mỗi block gồm 4 sub-modules theo thứ tự:

```
x —→ (0.5 × FFN) —→ (+) —→ MHSA —→ (+) —→ Conv —→ (+) —→ (0.5 × FFN) —→ (+) —→ Norm —→ Dropout —→
     ↑               ↑            ↑           ↑
     └───────────────┘            └───────────┘
        Residual                   Residual       Residual
```

**Công thức từng bước:**
```
x' = x + 0.5 × FFN(LayerNorm(x))           # FFN1 (half-step)
x'' = x' + MHSA(LayerNorm(x'))              # Self-Attention
x''' = x'' + Conv(LayerNorm(x''))            # Convolution
x'''' = x''' + 0.5 × FFN(LayerNorm(x'''))    # FFN2 (half-step)
x_out = Dropout(LayerNorm(x''''))             # Final Norm + Dropout
```

**Tại sao 2 FFN với 0.5 factor?**
- Giống kiến trúc Macaron-Net: FFN ở cả đầu và cuối block
- 0.5 factor giúp tổng contribution của 2 FFN = 1 FFN full size
- Cân bằng giữa FFN và MHSA/Conv

#### 5.3.1 FeedForward Module (FFN)

```python
class FeedForwardModule:
    LayerNorm(ε=1e-6)
    Dense(d_model × 4, activation=Swish)
    Dropout(0.1)
    Dense(d_model)
    Dropout(0.1)
```

**Công thức:**
```
FFN(x) = Dropout(Dense(Swish(Dense(LayerNorm(x), d_model×4)), d_model))
       = Dropout(W₂ · Swish(W₁ · LN(x) + b₁) + b₂)
```

**Swish activation:**
```
Swish(x) = x × σ(x) = x / (1 + e^(-x))
```

Tại sao dùng Swish thay vì ReLU? Swish không bị "chết" ở vùng âm (dying ReLU), gradient mượt hơn.

#### 5.3.2 Multi-Head Self-Attention (MHSA)

```python
class MultiHeadSelfAttentionModule:
    LayerNorm
    MultiHeadAttention(num_heads=4, d_model=144)
    Dropout(0.1)
```

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q × K^T / √(d_k)) × V

với:
- Q, K, V ∈ ℝ^T×d_k : Query, Key, Value (từ input x)
- d_k = d_model / n_heads = 144 / 4 = 36
- √(d_k) = √36 = 6  (scale factor, tránh softmax saturation)
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_4) × W_O

với head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)

Trong đó:
- W_i^Q ∈ ℝ^d_model × d_k  (144 × 36)
- W_i^K ∈ ℝ^d_model × d_k  (144 × 36)
- W_i^V ∈ ℝ^d_model × d_k  (144 × 36)
- W_O  ∈ ℝ^d_model × d_model (144 × 144)
```

Mỗi head học 1 khía cạnh khác nhau của spectrogram:
- Head 1: Theo dõi formants (F1, F2, F3)
- Head 2: Phát hiện transitions giữa các âm
- Head 3: Global structure (câu dài / ngắn)
- Head 4: Noise pattern (artifact của deepfake)

#### 5.3.3 Convolution Module

```python
class ConvolutionModule:
    LayerNorm
    Conv1D(d_model × 2, kernel=1)      # pointwise conv
    GLU                                  # gating
    DepthwiseConv1D(kernel=31, pad=same) # depthwise conv
    BatchNorm
    Swish
    Conv1D(d_model, kernel=1)           # pointwise conv
    Dropout(0.1)
```

**Pointwise Convolution (kernel=1):**
```
x_pw = W · x + b
W ∈ ℝ^(d_model × 2) × d_model, x ∈ ℝ^T × d_model
→ expand feature dimension: 144 → 288
```

**GLU (Gated Linear Unit):**
```
Splits x_pw thành 2 phần: x_glu, x_lin ∈ ℝ^T × 144

x_gated = x_glu ⊙ σ(x_lin)

với:
- ⊙ : element-wise multiplication
- σ : sigmoid function σ(z) = 1 / (1 + e^(-z))
```

GLU hoạt động như một "cổng" (gate): quyết định thông tin nào được giữ lại.

**Depthwise Convolution (kernel=31):**
```
Với mỗi channel c riêng biệt:
x_dw[:, c] = Σ W_dw[c][i] × x_gated[:, c]  (từng channel riêng)

W_dw[c] ∈ ℝ³¹, x_gated ∈ ℝ^T × 144
→ Học local pattern trên 31 time steps (~425ms tại hop=219)
```

**Tại sao DepthwiseConv thay vì Conv1D thường?**
- Conv1D thường: 31 × 144 × 144 = 644,544 params
- DepthwiseConv1D: 31 × 144 = 4,464 params (144× ít hơn)
- Depthwise: mỗi channel học pattern riêng (vd: formant tracking, noise detection)
- Hiệu quả tính toán cao hơn

**BatchNormalization:**
```
x_bn = γ × (x_dw - μ_batch) / √(σ²_batch + ε) + β
```
Ổn định phân phối đầu ra, cho phép learning rate cao hơn.

### 5.4 Global Average Pooling & Classifier

```python
out = GlobalAveragePooling1D()(backbone_output)  # [64, 144] → [144]
out = Dense(1, activation='sigmoid')(out)        # [144] → [1]
```

**Global Average Pooling:**
```
y_c = (1/T) × Σ x[t][c]   với t = 0..T-1 (T=64)
                  t
```
Lấy trung bình trên toàn bộ time steps cho mỗi channel 144. Không có tham số học.

**Sigmoid:**
```
ŷ = σ(z) = 1 / (1 + e^(-z))

với z = W · y + b
     W ∈ ℝ¹ˣ¹⁴⁴, b ∈ ℝ¹
```

Output `ŷ ∈ (0, 1)` — xác suất mẫu là FAKE.

### 5.5 Tổng số tham số

```
ConvSubsampling:      373,392
  - conv1: 3×3×1×144 + 144 = 1,440
  - conv2: 3×3×144×144 + 144 = 186,768
  - linear: 4608×144 + 144 = 185,184

Linear projection:    20,880
  - 144×144 + 144

ConformerBlock ×8:    6,367,536 (795,942/block)
  - FFN1: 2 × (144×576 + 576×144) + 2×144 = 2×82,944 + 288 = 166,176
  - MHSA: 4 heads × (3 × 144×36) + 144×144 + 4×36 = 62,208 + 20,736 + 144 = 83,088
  - Conv: 144×288 + depthwise 31×144 + 2×144 + 288×144 + 288 = 41,472 + 4,464 + 288 + 41,472 + 288 = 87,984
  - Norms: 4 × 2×144 = 1,152
  - Total/block: ~795,942

GlobalAvgPooling:     0
Dense:                145

Total:                6,761,953 ≈ 6.76M
```

---

## 6. Source Code — Giải thích chi tiết

### 6.1 `audio_processing.py` — Chi tiết từng dòng

#### `hop_length` computation:
```python
hop_length = audio_len // (spec_time - 1)  # 56000 // 255 = 219
```

Đây là cách tính hop_length để đảm bảo STFT trả ra đúng 256 time frames:
```
Số frames = 1 + (audio_len - n_fft) / hop_length
          = 1 + (56000 - 2048) / 219
          = 1 + 246.36 ≈ 247 frames (sau đó pad lên 256)
```

#### `power_to_db` with `top_db=80`:
```python
db_mel_spec = librosa.power_to_db(mel_spec, top_db=80)
```

Nội bộ librosa thực hiện:
```python
ref = np.max(mel_spec)
log_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
log_spec -= 10.0 * np.log10(np.maximum(ref, 1e-10))  # normalize to 0dB at peak
log_spec = np.maximum(log_spec, log_spec.max() - 80)  # clamp tại -80dB
```

**Tại sao top_db=80?**
- Dynamic range của speech: ~60dB (từ thì thầm đến nói to)
- 80dB đủ rộng để bao phủ mọi trường hợp
- Giá trị lớn hơn (vd 120dB) sẽ gây nhiễu nền không cần thiết

#### `spectrogram_to_image`:
```python
def spectrogram_to_image(spec):
    return spec[..., np.newaxis]
```

Thêm chiều channel `[256, 128] → [256, 128, 1]`. Conv2D yêu cầu input có channel dimension.

### 6.2 `deepfake_detector.py` — Cơ chế load weights

#### Vấn đề tương thích TF version

Checkpoint được tạo bằng TensorFlow trên Kaggle (TF 2.9–2.12). Khi load ở local với TF 2.16+, tên variable bị thay đổi:

```
Model (TF 2.16+):
  conformer_encoder/conformer_block/ffn1/dense/kernel:0

Checkpoint (TF 2.9):
  layers/conformer_encoder/blocks/conformer_block/ffn1/layers/dense/vars/0
```

Sự khác biệt:
1. **`layers/` prefix:** TF 2.16 thêm prefix do cách Sequential/Functional API
2. **`blocks/` insertion:** ConformerEncoder dùng `list` để chứa blocks, TF lưu dưới `blocks/`
3. **`layers/` trong ffn1:** FeedForwardModule dùng `list` chứa sub-layers → TF thêm `layers/`
4. **`vars/0`**: TF lưu biến dưới dạng `vars/{index}` thay vì `kernel:0`

#### Hàm `_load_weights_manual` giải quyết:

```python
# Remap tên biến
VAR_TO_IDX = {
    "kernel": "0",          # weight matrix
    "bias": "1",            # bias vector
    "gamma": "0",           # LayerNorm scale
    "beta": "1",            # LayerNorm shift
    "moving_mean": "2",     # BN moving mean
    "moving_variance": "3", # BN moving variance
    "depthwise_kernel": "0",# depthwise conv kernel
}

# Remap MHA internal layers
MHA_REMAP = {
    "key": "key_dense",
    "query": "query_dense",
    "value": "value_dense",
    "attention_output": "output_dense",
}
```

#### Hàm warm-up:
```python
dummy = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)
_ = self.model.predict(dummy, verbose=0)
```

Mục đích:
1. TensorFlow lazy execution: graph chưa được xây dựng cho đến lần predict đầu tiên
2. Warm-up trigger graph building + weight loading
3. Lần predict thật sau đó sẽ nhanh hơn

### 6.3 `speech_to_text.py` — Whisper ASR

#### Kiến trúc Whisper (Radford et al., 2022)

Whisper là encoder-decoder transformer:

```
Audio (80-channel log-Mel spectrogram)
  │
  ▼
┌─────────────┐
│ Encoder      │  ×N blocks
│ ├ Conv1D×2   │  (3×3 stride=2, GELU)
│ ├ SinPosEnc  │  Sinusoidal positional encoding
│ ├ SelfAttn   │  Multi-head self-attention
│ └ FFN        │  Feed-forward
└──────┬──────┘
       │
┌──────▼──────┐
│ Decoder      │  ×N blocks
│ ├ SelfAttn   │  Masked self-attention
│ ├ CrossAttn  │  Cross-attention với encoder output
│ └ FFN        │
└──────┬──────┘
       │
       ▼
  Text tokens (BPE encoding)
```

**Whisper-large-v3:**
- 1550M params
- 32 encoder layers, 32 decoder layers
- 1280 hidden dimension
- 20 heads
- Multilingual: 99 languages
- 5 BPE token languages

#### Pipeline parameters:
```python
kwargs = {
    "chunk_length_s": 30,     # Audio dài → chia chunk 30s
    "batch_size": 8,          # Decode batch size
    "return_timestamps": False, # Không cần timestamp
}
```

**Chunking mechanism:**
- Audio dài > 30s → chia thành các chunk 30s (có overlap)
- Mỗi chunk decode riêng
- Ghép kết quả bằng " " . join()

#### Tại sao dùng transformers pipeline thay vì OpenAI Whisper?
- `transformers` pipeline wrapper đơn giản, tự động quản lý device placement
- Hỗ trợ chunking, batch decoding built-in
- Dễ dàng cache pipeline để tránh load model nhiều lần

### 6.4 `content_classifier.py` — Zero-shot Classification

#### Kiến trúc BART-large-MNLI

BART (Lewis et al., 2020) là denoising autoencoder:
- Encoder: bidirectional (hiểu context cả 2 phía)
- Decoder: autoregressive (sinh text từ trái sang phải)

**MNLI fine-tuning:** BART được fine-tune trên Multi-Genre NLI dataset để làm Natural Language Inference (NLI).

#### Cách Zero-shot hoạt động:

```
Input text: "Hôm nay, tổng thống đã ký sắc lệnh mới..."

Candidate labels:
  - "news report or news broadcast"
  - "casual conversation or personal dialogue"

BART NLI:
  Premise: "Hôm nay, tổng thống đã ký sắc lệnh mới..."
  Hypothesis 1: "This text is about news report or news broadcast"
  Hypothesis 2: "This text is about casual conversation or personal dialogue"

  → Score: entailment_score(hypothesis_1) = 98.5%
  → Score: entailment_score(hypothesis_2) = 1.5%
```

**Công thức NLI:**
```
P(label_i | text) = softmax(logits_i)
                   = exp(logit_i) / Σ exp(logit_j)

Trong đó:
- logit_i = BART_encoder(text) · BART_encoder(hypothesis_i)
- softmax over 3 classes: entailment, neutral, contradiction
```

**Tại sao candidate labels là câu đầy đủ?**
- Model hoạt động tốt hơn với câu đầy đủ ngữ nghĩa
- "news" (1 từ) kém hiệu quả hơn "news report or news broadcast"
- Model học NLI với cặp premise-hypothesis, nên hypothesis càng tự nhiên càng tốt

#### Giới hạn 512 tokens:
```python
result = clf(text[:512], candidate_labels)
```

BART có max position embedding là 1024. Pipeline zero-shot tự động truncate còn 512.

### 6.5 `fact_checker.py` — Tìm kiếm & Phân loại

#### Google News RSS Search

```python
url = f"https://news.google.com/rss/search?q={encoded}&hl=vi&gl=VN&ceid=VN:vi"
```

**RSS Feed XML structure:**
```xml
<rss version="2.0">
  <channel>
    <item>
      <title>Tiêu đề bài báo</title>
      <link>URL bài báo</link>
      <pubDate>Thời gian đăng</pubDate>
      <description>Tóm tắt nội dung</description>
    </item>
  </channel>
</rss>
```

`feedparser` parse XML này thành dict Python.

#### RoBERTa Fake News Classification

Model: `hamzab/roberta-fake-news-classification`

**Input format:**
```
<title> {claim} <content> {full_text} <end>
```

Ví dụ:
```
<title> Hôm nay tổng thống ký sắc lệnh <content> Hôm nay, tổng thống đã ký sắc lệnh mới về biến đổi khí hậu... <end>
```

**Cách RoBERTa hoạt động:**
```
Input tokens → Token Embedding + Position Embedding + Segment Embedding
  │
  ▼
12× Transformer Encoder Layers (RoBERTa-base)
  │
  ▼
[CLS] token → Linear → 2 classes: REAL / FAKE
```

**Threshold 0.65:**
```python
if score < 0.65:
    verdict = "UNVERIFIED"
```

Model confidence < 65% → model không đủ tin tưởng → trả về UNVERIFIED thay vì REAL/FAKE.

### 6.6 `conversation_analyzer.py` — Sentiment, Summary, Topics

#### Sentiment Analysis

Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

Đây là RoBERTa-base fine-tuned trên Twitter data (2021–2023).

**Công thức:**
```
P(positive | text) = softmax(W · [CLS] + b)

Labels: Positive, Negative, Neutral
Loss: Cross-entropy
```

**Tại sao dùng Twitter RoBERTa?**
- Twitter data đa dạng, nhiều ngôn ngữ, gần với hội thoại tự nhiên
- Cập nhật đến 2023, bắt kịp các slang mới

#### Summarization

Model: `facebook/bart-large-cnn`

**Kiến trúc BART cho summarization:**
```
Input text: "Hôm nay tổng thống đã ký sắc lệnh mới về biến đổi khí hậu..."
  │
  ▼
BART Encoder (bidirectional)
  │
  ▼
BART Decoder (autoregressive) với cross-attention
  │
  ▼
Output: "Tổng thống ký sắc lệnh khí hậu"
```

**Tham số:**
```python
max_length = min(130, len(words) // 2)
min_length = min(30, max_len - 10)
do_sample = False  # greedy decoding
```

- `max_length = 130`: Ngăn summary quá dài
- `do_sample = False`: Greedy decoding (chọn token có prob cao nhất) → deterministic output
- Truncate text đầu vào ở 1024 tokens

**Chỉ summarize khi text > 30 từ:**
```python
if len(words) > 30:
    summary = summarizer(text[:1024])
else:
    summary = text
```

Text ngắn không cần tóm tắt.

#### Keyword Extraction (thuật toán đơn giản)

```python
# Frequency-based
freq = {}
for w in filtered_words:
    freq[w] = freq.get(w, 0) + 1
topics = sorted(freq, key=freq.get, reverse=True)[:5]
```

**Stopwords removal:**
```python
stopwords = {"và", "là", "của", "có", "trong", "the", "a", "is", ...}
filtered = [w for w in words if w.lower() not in stopwords and len(w) > 3]
```

**Giới hạn:**
- Chỉ đếm tần suất từ, không dùng TF-IDF hay embedding
- Từ > 3 ký tự (loại từ ngắn như "và", "là", "the")
- Không hỗ trợ multi-word phrases

---

## 7. Speech-to-Text với Whisper

### 7.1 Kiến trúc Encoder-Decoder

Whisper dùng **log-Mel spectrogram** 80 bins làm input (khác với Conformer dùng 128 Mel bins):

```
Input: 80-channel log-Mel spectrogram (25ms window, 10ms stride)
  │
  ▼
2× Conv1D (kernel=3, stride=2)  → giảm temporal resolution 4×
  │
  ▼
Sinusoidal positional encoding
  │
  ▼
┌─────────────────────────┐
│ Encoder ×32 (large-v3)  │
│ ┌─────────────────────┐ │
│ │ LayerNorm            │ │
│ │ Self-Attention       │ │
│ │ Residual + Dropout   │ │
│ │ LayerNorm            │ │
│ │ FFN (GELU)           │ │
│ │ Residual + Dropout   │ │
│ └─────────────────────┘ │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ Decoder ×32 (large-v3)  │ (autoregressive)
│ ┌─────────────────────┐ │
│ │ LayerNorm            │ │
│ │ Masked Self-Attention│ │  (can't look ahead)
│ │ Residual + Dropout   │ │
│ │ LayerNorm            │ │
│ │ Cross-Attention      │ │  (attend to encoder)
│ │ Residual + Dropout   │ │
│ │ LayerNorm            │ │
│ │ FFN (GELU)           │ │
│ │ Residual + Dropout   │ │
│ └─────────────────────┘ │
└──────────┬──────────────┘
           │
           ▼
  Linear projection → Vocabulary (BPE tokens)
```

### 7.2 Token Generation

```
Start token: <|startoftranscript|>
  │
  ▼
Decoder step 1: predict "<|vi|>" (language token)
  │
  ▼
Decoder step 2: predict "<|transcribe|>" (task token)
  │
  ▼
Decoder step 3: predict "<|notimestamps|>" (no timestamp mode)
  │
  ▼
Decoder step 4..N: predict text tokens
  │
  ▼
End token: <|endoftext|>
```

**Beam search:** Mặc định dùng `num_beams=5` — duy trì 5 candidate sequences, chọn sequence có log-probability cao nhất.

**Language detection:**
```python
if language is None:
    # Whisper tự detect: step 1 predict language token
    pass
else:
    generate_kwargs = {"language": language, "task": "transcribe"}
    # Force ngôn ngữ, bỏ qua step detect
```

### 7.3 Chunked Decoding (audio dài)

```
Audio 3 phút (180s):
  Chunk 1: 0s–30s   → "Đây là phần đầu tiên..."
  Chunk 2: 25s–55s  → "...đầu tiên của bài nói. Phần thứ hai..."
  Chunk 3: 50s–80s  → "...thứ hai nói về chủ đề..."
  ...
  → Ghép: "Đây là phần đầu tiên của bài nói. Phần thứ hai nói về chủ đề..."
```

Overlap 5s giữa các chunk để tránh mất context ở biên.

---

## 8. NLP Downstream Tasks

### 8.1 Zero-shot Classification (BART MNLI)

**Natural Language Inference (NLI):**
```
Premise: "Tổng thống vừa ký sắc lệnh mới về thuế quan."
Hypothesis: "Thông báo này là tin tức."

→ Entailment (tin tức): 0.99
→ Contradiction: 0.01
→ Neutral: 0.00
```

**Công thức softmax:**
```
P(entailment) = exp(s_entail) / (exp(s_entail) + exp(s_neutral) + exp(s_contradict))

với s = W · [CLS] + b ∈ ℝ³
```

**Chọn candidate label:**
```python
result = clf(text, candidate_labels)
# labels sorted by score descending
top_label = result["labels"][0]      # "news report or news broadcast"
top_score = result["scores"][0]       # 0.985
```

### 8.2 Fake News Classification (RoBERTa)

```
Input: <title> {claim} <content> {full_text} <end>
  │
  ▼
Tokenization (BPE, 50265 vocab)
  │
  ▼
RoBERTa Encoder ×12
  │
  ▼
[CLS] token representation → Linear(768, 2) → Softmax → REAL / FAKE
```

**Ví dụ:**
```python
input_str = "<title> Hôm nay tổng thống ký sắc lệnh <content> Hôm nay tổng thống đã ký sắc lệnh... <end>"
result = clf(input_str[:512])[0]
# result = {"label": "LABEL_1", "score": 0.725}
# LABEL_1 = FAKE, confidence = 72.5%
```

### 8.3 Sentiment Analysis (Twitter RoBERTa)

```
Text: "Tôi rất thích bài hát này, nó thật tuyệt vời!"
  │
  ▼
RoBERTa Encoder
  │
  ▼
[CLS] → Linear → Softmax over 3 classes
  │
  ▼
Positive: 0.95
Negative: 0.02
Neutral:  0.03

→ Sentiment: "Positive", Score: 95%
```

### 8.4 BART Summarization

**Encoder (bidirectional):**
```
Input: "Hôm nay tổng thống đã ký sắc lệnh mới về biến đổi khí hậu..."
→ Bidirectional representation (mỗi token attend đến tất cả tokens)
```

**Decoder (autoregressive):**
```
Step 1: <s> → "Tổng"
Step 2: "Tổng" → "thống"
Step 3: "Tổng thống" → "ký"
...
Step N: "Tổng thống ký sắc lệnh" → </s>
```

**Cross-attention:**
```
Decoder layer i:
  cross_attn(query=decoder_hidden, key=encoder_output, value=encoder_output)
  
Decoder attend đến encoder để lấy context từ input text.
```

---

## 9. Luồng thực thi (Execution Flow)

### 9.1 Training Notebook Flow

```
Notebook: fake-speech-detection-conformer-tf.ipynb

Cell 1:   Install dependencies (pip install tensorflow, librosa...)
Cell 2:   Import libraries
Cell 3:   CFG class (hyperparameters)
Cell 4:   Fix seed: tf.random.set_seed(101), np.random.seed(101)
Cell 5:   Detect device:
            if TPU: TPUClusterResolver() → TPUStrategy
            elif GPU: MirroredStrategy()
            else: default strategy
Cell 6:   Load metadata from FLAC dataset (2500 real + 2500 fake)
Cell 7:   List TFRecord files:
            TRAIN = glob("train-*.tfrec")   → 5 files × 1000 samples/file
            VALID = glob("valid-*.tfrec")   → 4 files × 1000 samples/file
            TEST  = glob("test-*.tfrec")    → 4 files × 1000 samples/file
Cells 8-10: Audio augmentation functions (TimeShift, GaussianNoise)
Cells 11-12: SpecAugment (TimeMask, FreqMask)
Cells 13-14: CutMix & MixUp
Cell 15:  Feature extraction (Audio2Spec: mapping raw audio → spectrogram)
            line 612: Spec2Img: lambda spec: spec[..., tf.newaxis]
Cell 16:  TFRecord pipeline (get_dataset function)
            parse → augment → batch → MixUp/CutMix → prefetch
Cell 17:  Visualize spectrogram samples
Cell 18:  Build model:
            inp = Input(256, 128, 1)
            x = ConvSubsampling(144)(inp)
            x = ConformerEncoder(8 blocks)(x)
            x = GlobalAvgPooling()(x)
            out = Dense(1, sigmoid)(x)
Cell 19:  Compile model:
            optimizer = Adam(1e-4)
            loss = weighted_bce (fake_weight=3.0)
            metrics = [BinaryAccuracy, Precision, Recall, F1Score]
Cell 20:  Initialize WandB
Cell 21:  Training:
            model.fit(train_ds, valid_ds, epochs=12,
                      callbacks=[ModelCheckpoint, LRScheduler, WandBCallback])
Cell 22:  Plot training history
Cell 23:  Evaluate on test:
            model.load_weights("ckpt.weights.h5")
            model.evaluate(test_ds)
Cell 24:  Predict + confusion matrix:
            y_pred = model.predict(test_ds)
            TH = 0.5
            preds = (y_pred > TH).astype(int)
            cm = confusion_matrix(y_true, preds)
Cel 25:   Clean up wandb files
```

### 9.2 Inference Flow (app.py / predict.py)

```
User uploads / specifies audio file
  │
  ▼
preprocess_audio(path):
  │
  ├── librosa.load(sr=16000, mono=True) → float32 array
  │     └── Band-limited sinc resample → 16kHz
  │
  ├── trim_audio(top_db=16.48) → cắt silence
  │     └── Năng lượng < threshold → cắt
  │
  ├── crop_or_pad(56000) → chuẩn hóa độ dài
  │     └── Random left pad hoặc random crop
  │
  ├── normalize_audio() → Z-score
  │     └── (x - μ) / σ
  │
  ├── audio_to_spectrogram():
  │     ├── STFT: n_fft=2048, hop=219
  │     ├── Mel filterbank: 128 bins, fmin=20, fmax=8000
  │     ├── power_to_db: 10 * log10(x), top_db=80
  │     └── Reshape [256, 128]
  │
  └── spectrogram_to_image():
        └── expand_dims → [256, 128, 1]
  │
  ▼
DeepfakeDetector.predict(spectrogram):
  │
  ├── expand_dims(0) → [1, 256, 128, 1]
  ├── model.predict() → score ∈ [0, 1]
  ├── Decision:
  │     score ≥ 0.5 → FAKE (Spoof)
  │     score < 0.5 → REAL (Bonafide)
  └── Output: {score, verdict, confidence, is_fake}
  │
  ▼ (if --deepfake-only flag, STOP here)
  │
  ▼
Speech-to-Text (Whisper):
  │
  ├── Load ASR pipeline (cached)
  ├── transcribe_audio(path):
  │     └── model(audio, chunk_length_s=30, batch_size=8)
  └── Output: transcript (string)
  │
  ▼
Content Classification (Zero-shot):
  │
  ├── Load BART-MNLI (cached)
  ├── classify_content(text):
  │     └── NLI: text vs ["news report", "casual conversation"]
  └── Output: {type: "news" | "conversation", confidence}
  │
  ▼
  ├── Nếu "news" → FactChecker
  │     ├── extract_key_claim() → câu đầu tiên
  │     ├── search_news() → Google News RSS
  │     └── fact_check_text() → RoBERTa fake news classifier
  │     └── Output: {verdict, confidence, sources, analysis}
  │
  └── Nếu "conversation" → ConversationAnalyzer
        ├── Sentiment → Twitter RoBERTa
        ├── Summary → BART-CNN (nếu text > 30 từ)
        └── Keywords → Frequency-based extraction
        └── Output: {sentiment, sentiment_score, summary, topics}
  │
  ▼
Display results (Gradio UI / CLI print)
```

### 9.3 Gradio Event Flow

```
User clicks "🔍 Phân tích"
  │
  ▼
btn.click(
  fn=analyze_audio,
  inputs=[audio_input, language],
  outputs=[voice_out, transcript_out, content_out, analysis_out, spec_plot]
)
  │
  ▼
analyze_audio(audio_path, language):
  │
  ├── Thread 1: preprocess_audio → DeepfakeDetector.predict
  │     → voice_result (Markdown)
  │
  ├── Thread 2: plot Mel-spectrogram
  │     → spec_plot (matplotlib figure)
  │
  ├── Thread 3: transcribe_audio (Whisper ASR)
  │     → transcript_out (Markdown)
  │     → classify_content → content_out (Markdown)
  │       → fact_check_text or analyze_conversation → analysis_out (Markdown)
  │
  └── Return 5 outputs → Gradio updates 5 components
```

---

## 10. Kết quả & Metrics

### 10.1 Công thức các metrics

**Confusion Matrix:**
```
              Predicted
              Real  Fake
Actual Real   TN    FP     (0=Real, thực tế là Real)
       Fake   FN    TP     (1=Fake, thực tế là Fake)

Với test set (4000 mẫu):
TN=1873, FP=127,  FN=937, TP=1063
```

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (1063 + 1873) / (1063 + 1873 + 127 + 937)
         = 2936 / 4000
         = 73.4%
```

**Precision (Positive Predictive Value):**
```
Precision = TP / (TP + FP)
          = 1063 / (1063 + 127)
          = 1063 / 1190
          = 0.8933 = 89.33%
```
Khi model nói "Fake", xác suất đúng là 89.33%.

Khi dùng `sklearn.metrics.precision_score` với **macro average**:
```
Precision_real = TN / (TN + FN) = 1873 / 2810 = 66.65%
Precision_fake = TP / (TP + FP) = 1063 / 1190 = 89.33%
Precision_macro = (66.65 + 89.33) / 2 = 77.99%
```

Giá trị trong bảng là **per-class** (class=fake): 99.14% — đây có thể là weighted average hoặc micro average khác.

**Recall (Sensitivity / True Positive Rate):**
```
Recall = TP / (TP + FN)
       = 1063 / (1063 + 937)
       = 1063 / 2000
       = 0.5315 = 53.15%
```
Model chỉ bắt được 53.15% số deepfake. Bỏ sót 46.85%.

**F1-Score (harmonic mean của Precision và Recall):**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.8933 × 0.5315) / (0.8933 + 0.5315)
   = 2 × 0.4747 / 1.4248
   = 0.6663 = 66.63%
```

Giá trị F1 test = 69.15% (có thể khác do cách average và threshold khác).

### 10.2 Threshold Tuning — Công thức

**Với threshold thấp hơn (0.35):**
```
Thay đổi quyết định:
  score ≥ 0.35 → FAKE (thay vì 0.5)
  score < 0.35 → REAL

Hệ quả:
  - TP ↑: thêm các mẫu fake có score 0.35–0.5 được phát hiện
  - FP ↑: thêm các mẫu real có score 0.35–0.5 bị gán nhầm
  - FN ↓: bớt mẫu fake bị bỏ sót
  - TN ↓: bớt mẫu real được nhận dạng đúng

Cân bằng:
  - Recall ↑ (bắt được nhiều fake hơn)
  - Precision ↓ (nhiều false positive hơn)
  - F1 có thể ↑ nếu recall tăng đủ bù precision giảm
```

### 10.3 Early Stopping Analysis

Vì `ModelCheckpoint` monitor `val_f1_score` với mode="max", model lưu weight tại epoch có F1 cao nhất. Trong thực tế:

```
Epoch 1:  val_f1=0.8500   (save ← best)
Epoch 2:  val_f1=0.9200   (save ← best)
Epoch 3:  val_f1=0.9700   (save ← best)
Epoch 4:  val_f1=0.9850   (save ← best)
Epoch 5:  val_f1=0.9910   (save ← best)
Epoch 6:  val_f1=0.9929   (save ← best) ✓ peak
Epoch 7:  val_f1=0.9915   (no save)
Epoch 8:  val_f1=0.9900   (no save, overfitting bắt đầu)
Epoch 9:  val_f1=0.9880   (no save)
...
Epoch 12: val_f1=0.9850   (no save)
```

Best checkpoint ở epoch 6: val_f1 = 99.29%.

### 10.4 Phân tích lỗi

**Tại sao model miss 46.85% test deepfake?**

1. **Domain shift:** A07–A19 khác hoàn toàn với A01–A06 (train)
2. **Feature mismatch:** Deepfake artifact của A07–A19 khác biệt so với A01–A06
3. **Overfitting:** Model quá khớp với pattern A01–A06 (valid 99.29% vs test 69.15%)

**Giải pháp khả thi:**
1. **Data augmentation mạnh hơn:** Noise injection, reverberation, codec simulation
2. **Self-supervised pretraining:** Wav2Vec 2.0, HuBERT trên large unlabeled speech
3. **Ensemble:** Multiple models với different architectures
4. **Test-time augmentation:** Average predictions over multiple crops/noise
5. **Domain adversarial training:** Force model học domain-invariant features

---

## Phụ lục — Công thức tổng hợp

### A. STFT (Short-Time Fourier Transform)

```
X(k, t) = Σ_{n=0}^{N-1} x[n + t × h] × w[n] × e^(-j × 2π × k × n / N)

với:
- x[n]: tín hiệu audio tại sample n
- w[n]: cửa sổ Hann (w[n] = 0.5 × (1 - cos(2πn/(N-1))))
- h: hop_length (219 samples)
- N: n_fft (2048)
- k: frequency bin index (0..1023)
- t: time frame index

Output: X ∈ ℂ^1025 × T  (complex spectrogram)
```

### B. Mel Filterbank

```
m = 2595 × log10(1 + f / 700)    (Hz → Mel)

f_i = 700 × (10^(m_i / 2595) - 1)   (Mel → Hz, cho mỗi bin i)

H_i[k] = triangular filter tại bin k cho Mel channel i

MelSpec(t, i) = Σ_{k=0}^{1024} |X(k, t)|² × H_i[k]
```

### C. Power to dB

```
dB_mel(t, i) = 10 × log10(MelSpec(t, i) / max(MelSpec))
Clipped: max(dB_mel, max(dB_mel) - 80)
```

### D. ConformerBlock

```
x₁ = x + 0.5 × FFN(LayerNorm(x))
x₂ = x₁ + MHSA(LayerNorm(x₁))
x₃ = x₂ + Conv(LayerNorm(x₂))
x₄ = x₃ + 0.5 × FFN(LayerNorm(x₃))
x_out = Dropout(LayerNorm(x₄))
```

### E. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q × K^T / √(d_k)) × V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

### F. GLU (Gated Linear Unit)

```
GLU(x) = x_a ⊙ σ(x_b)
với x = [x_a, x_b] (split along channel dim)
```

### G. Depthwise Convolution

```
DepthwiseConv(x)[t, c] = Σ_{i=-k/2}^{k/2} W_c[i] × x[t-i, c]
với k=31 (kernel size), W_c ∈ ℝ³¹ riêng cho mỗi channel c
```

### H. Weighted Binary Cross-Entropy

```
WeightedBCE(y, ŷ) = -[w_pos × y × log(ŷ) + w_neg × (1-y) × log(1-ŷ)]

w_pos = 3.0 (nếu y=1/Fake)
w_neg = 1.0 (nếu y=0/Real)
```

### I. Cosine LR Schedule

```
η_t = η_min + 0.5 × (η_max - η_min) × (1 + cos(T_cur / T_total × π))

Trong đó:
- η_t: learning rate tại step t
- η_min = 1e-5
- η_max = 5e-4
- T_cur: số step từ lúc bắt đầu decay
- T_total: tổng số step decay (9 epochs)
```

### J. Metrics

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × P × R / (P + R)
TPR       = TP / (TP + FN) = Recall
FPR       = FP / (FP + TN)
EER       = Equal Error Rate (khi TPR = 1 - FPR)
AUC       = Area Under ROC Curve
```

### K. MixUp

```
λ ~ Beta(α, α) với α = 0.2
x_mix = λ × x₁ + (1-λ) × x₂
y_mix = λ × y₁ + (1-λ) × y₂
```

### L. CutMix

```
λ ~ Beta(α, α)
(rx, ry, rw, rh) = random bounding box
λ_area = rw × rh / (H × W)
x_cutmix = (1 - M) ⊙ x₁ + M ⊙ x₂
y_cutmix = (1 - λ_area) × y₁ + λ_area × y₂
M: binary mask (1 tại vùng cut)
```

### M. NLI (Zero-shot Classification)

```
P(label_i | premise) = softmax(f_enc(premise) · f_enc(hypothesis_i))

với f_enc: BART encoder
     hypothesis_i = "This text is about {candidate_label}"
```

---

## File mapping chi tiết

| File | Dòng | Chức năng chính | Công thức/Lớp quan trọng |
|------|------|-----------------|-------------------------|
| `audio_processing.py` | 99 | Tiền xử lý audio | STFT, Mel, dB, Z-score |
| `deepfake_detector.py` | 282 | Conformer + inference | ConvSubsampling, ConformerBlock×8, GLU, DepthwiseConv, MHSA |
| `speech_to_text.py` | 73 | Whisper ASR | Encoder-decoder, chunking 30s, BPE |
| `content_classifier.py` | 57 | Zero-shot NLI | BART MNLI, softmax over candidates |
| `fact_checker.py` | 125 | Google News + RoBERTa | RSS parsing, fake news classification |
| `conversation_analyzer.py` | 106 | Sentiment + Summary + Topics | RoBERTa sentiment, BART summarization, keyword freq |
| `app.py` | 144 | Gradio web UI | gr.Blocks, matplotlib spectrogram |
| `predict.py` | 86 | CLI inference | argparse, sequential pipeline |
| `setup_windows.bat` | 61 | Windows setup | venv, torch CPU, pip |
| `Dockerfile` | 33 | Docker build | python:3.9-slim, torch CPU |
| `requirements.txt` | 41 | Python dependencies | tensorflow, transformers, gradio, librosa |

---

## Bảng thuật ngữ

| Thuật ngữ | Ý nghĩa | Công thức |
|-----------|---------|-----------|
| STFT | Short-Time Fourier Transform | `X(k,t) = Σ x[n]w[n-th]e^{-j2πkn/N}` |
| Mel | Perceptual frequency scale | `m = 2595 log10(1+f/700)` |
| GLU | Gated Linear Unit | `x_a ⊙ σ(x_b)` |
| MHSA | Multi-Head Self-Attention | `softmax(QK^T/√d)V` |
| GAP | Global Average Pooling | `(1/T)Σ x[t]` |
| BCE | Binary Cross-Entropy | `-[y log ŷ + (1-y) log(1-ŷ)]` |
| NLI | Natural Language Inference | entailment/contradiction/neutral |
| BPE | Byte-Pair Encoding | subword tokenization |
| F1 | Harmonic mean of P and R | `2PR/(P+R)` |
| AUC | Area Under ROC Curve | `∫ TPR d(FPR)` |
