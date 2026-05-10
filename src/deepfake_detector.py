"""
deepfake_detector.py
--------------------
Load Conformer model (weights-only .h5) và thực hiện dự đoán deepfake.

QUAN TRỌNG: ckpt.h5 là weights-only checkpoint.
Phải rebuild architecture giống hệt notebook training rồi mới load_weights().
"""

import numpy as np
import tensorflow as tf

# Shape cố định phải khớp với training CFG
SPEC_SHAPE  = [256, 128]
INPUT_SHAPE = (256, 128, 1)


def build_conformer_model(input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    """
    Rebuild kiến trúc Conformer GIỐNG HỆT notebook training.
    Dùng audio_classification_models (cần pip install audio_classification_models).
    """
    import audio_classification_models as acm
    from audio_classification_models.models import conformer
    from audio_classification_models.layers import multihead_attention

    # --- MONKEY PATCH FOR KERAS 2.13+ MASK ISSUE ---
    # In newer TF/Keras, when inputs is a list, mask is implicitly passed as a list of Nones.
    # We clean it up so that we don't crash when len(mask.shape) is called on a list.
    def clean_mask(mask):
        if isinstance(mask, list):
            # If it's a list and all elements are None, just return None
            flattened = tf.nest.flatten(mask)
            if all(m is None for m in flattened):
                return None
        return mask

    orig_cb_call = conformer.ConformerBlock.call
    def patched_cb_call(self, inputs, training=False, mask=None, **kwargs):
        return orig_cb_call(self, inputs, training=training, mask=clean_mask(mask), **kwargs)
    conformer.ConformerBlock.call = patched_cb_call

    orig_mhsa_call = conformer.MHSAModule.call
    def patched_mhsa_call(self, inputs, training=False, mask=None, **kwargs):
        return orig_mhsa_call(self, inputs, training=training, mask=clean_mask(mask), **kwargs)
    conformer.MHSAModule.call = patched_mhsa_call

    orig_rpmha_call = multihead_attention.RelPositionMultiHeadAttention.call
    def patched_rpmha_call(self, inputs, training=False, mask=None, **kwargs):
        return orig_rpmha_call(self, inputs, training=training, mask=clean_mask(mask), **kwargs)
    multihead_attention.RelPositionMultiHeadAttention.call = patched_rpmha_call
    # ----------------------------------------------

    # Khởi tạo backbone là ConformerEncoder thay vì toàn bộ mô hình Conformer
    backbone = conformer.ConformerEncoder()
    
    inp = tf.keras.Input(shape=input_shape)
    x   = backbone(inp)
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=out)


class DeepfakeDetector:
    """
    Phát hiện giọng nói deepfake dùng Conformer model.
    Score: 0.0 = Real (Bonafide), 1.0 = Fake (Spoof)
    """

    def __init__(self, model_path: str = "models/ckpt.h5"):
        print(f"Loading model weights from: {model_path}")
        self.model = build_conformer_model()
        # Dùng load_weights vì ckpt.h5 là weights-only
        self.model.load_weights(model_path)
        print(f"✅ Model weights loaded successfully")
        print(f"   Input shape  : {self.model.input_shape}")
        print(f"   Output shape : {self.model.output_shape}")
        # Warm-up inference
        dummy = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("   Warm-up done ✓")

    def predict(self, spectrogram: np.ndarray) -> dict:
        """
        Args:
            spectrogram: np.ndarray shape [256, 128, 1]
        Returns:
            dict with keys: score, verdict, confidence, is_fake
        """
        inp   = np.expand_dims(spectrogram, axis=0).astype(np.float32)
        score = float(self.model.predict(inp, verbose=0)[0][0])

        verdict    = "FAKE (Spoof)"    if score >= 0.5 else "REAL (Bonafide)"
        confidence = score             if score >= 0.5 else (1.0 - score)

        return {
            "score"     : round(score, 4),
            "verdict"   : verdict,
            "confidence": round(confidence * 100, 2),
            "is_fake"   : score >= 0.5,
        }

    def predict_from_file(self, audio_path: str) -> dict:
        """Nhận đường dẫn file audio, trả về kết quả dự đoán."""
        from src.audio_processing import preprocess_audio
        spec = preprocess_audio(audio_path)
        return self.predict(spec)

    def predict_batch(self, spectrograms: np.ndarray) -> list:
        """
        Dự đoán theo batch cho nhiều file cùng lúc.
        Args:
            spectrograms: np.ndarray shape [N, 256, 128, 1]
        Returns:
            list of dicts
        """
        scores  = self.model.predict(spectrograms.astype(np.float32), verbose=0).flatten()
        results = []
        for score in scores:
            score = float(score)
            results.append({
                "score"     : round(score, 4),
                "verdict"   : "FAKE (Spoof)" if score >= 0.5 else "REAL (Bonafide)",
                "confidence": round((score if score >= 0.5 else 1 - score) * 100, 2),
                "is_fake"   : score >= 0.5,
            })
        return results
