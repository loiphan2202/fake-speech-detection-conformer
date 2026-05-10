"""
audio_processing.py
-------------------
Pipeline tiền xử lý âm thanh cho hệ thống Deepfake Speech Detection.
Tất cả thông số phải khớp CHÍNH XÁC với CFG trong notebook training.

NOTE: Không dùng tensorflow-io (Linux-only). Thay bằng librosa + tf.signal.
"""

import numpy as np
import librosa


# ── Cấu hình (phải khớp với CFG trong notebook training) ─────────────
SAMPLE_RATE = 16000
DURATION    = 3.5
AUDIO_LEN   = int(SAMPLE_RATE * DURATION)   # 56,000 samples
SPEC_TIME   = 256
SPEC_FREQ   = 128
N_FFT       = 2048
FMIN        = 20
FMAX        = SAMPLE_RATE // 2              # 8000 Hz
SPEC_SHAPE  = [SPEC_TIME, SPEC_FREQ]


def load_audio(file_path: str) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def trim_audio(audio: np.ndarray, epsilon: float = 0.15) -> np.ndarray:
    top_db = -20 * np.log10(epsilon)
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if len(trimmed) > 0 else audio


def crop_or_pad(audio: np.ndarray, target_len: int = AUDIO_LEN) -> np.ndarray:
    audio_len = len(audio)
    if audio_len < target_len:
        diff = target_len - audio_len
        pad_left  = np.random.randint(0, diff + 1)
        pad_right = diff - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif audio_len > target_len:
        diff  = audio_len - target_len
        start = np.random.randint(0, diff + 1)
        audio = audio[start:start + target_len]
    return audio.reshape(target_len)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
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
    spec_time = spec_shape[0]
    spec_freq = spec_shape[1]
    audio_len = len(audio)
    hop_length = audio_len // (spec_time - 1)

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=spec_freq, fmin=fmin, fmax=fmax, center=True,
    )
    db_mel_spec = librosa.power_to_db(mel_spec, top_db=80)
    db_mel_spec = db_mel_spec.T

    if db_mel_spec.shape[0] > spec_time:
        db_mel_spec = db_mel_spec[:spec_time, :]
    elif db_mel_spec.shape[0] < spec_time:
        pad_width = spec_time - db_mel_spec.shape[0]
        db_mel_spec = np.pad(db_mel_spec, ((0, pad_width), (0, 0)), mode='constant')

    return db_mel_spec.astype(np.float32)


def spectrogram_to_image(spec: np.ndarray) -> np.ndarray:
    return spec[..., np.newaxis]


def preprocess_audio(file_path: str) -> np.ndarray:
    audio = load_audio(file_path)
    audio = trim_audio(audio)
    audio = crop_or_pad(audio, target_len=AUDIO_LEN)
    audio = normalize_audio(audio)
    spec  = audio_to_spectrogram(audio)
    img   = spectrogram_to_image(spec)
    return img
