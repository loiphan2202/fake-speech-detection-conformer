"""
deepfake_detector.py
--------------------
Load Conformer model (weights-only .h5) và thực hiện dự đoán deepfake.

QUAN TRỌNG: ckpt.weights.h5 là weights-only checkpoint.
Phải rebuild architecture giống hệt notebook training rồi mới load_weights().
"""

import re
import h5py
import numpy as np
import tensorflow as tf

# Shape cố định phải khớp với training CFG
SPEC_SHAPE  = [256, 128]
INPUT_SHAPE = (256, 128, 1)

# ── Các layer tự định nghĩa giống hệt notebook training ─────────────

class ConformerConvSubsampling(tf.keras.layers.Layer):
    def __init__(self, d_model=144, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            d_model, 3, 2, padding='same', activation='relu', name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            d_model, 3, 2, padding='same', activation='relu', name='conv2')
        self.linear = tf.keras.layers.Dense(d_model, name='linear')
    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        b, t, f, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [b, t, f * c])
        x = self.linear(x)
        return x

class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self, d_model=144, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_normalization'),
            tf.keras.layers.Dense(d_model * 4, activation=tf.nn.swish, name='dense'),
            tf.keras.layers.Dropout(dropout, name='dropout'),
            tf.keras.layers.Dense(d_model, name='dense_1'),
            tf.keras.layers.Dropout(dropout, name='dropout_1'),
        ]
    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training) if isinstance(layer, tf.keras.layers.Dropout) else layer(x)
        return x

class MultiHeadSelfAttentionModule(tf.keras.layers.Layer):
    def __init__(self, d_model=144, n_head=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')
        self.attn = tf.keras.layers.MultiHeadAttention(n_head, d_model, dropout=dropout, name='attn')
        self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')
    def call(self, x, training=False):
        x_norm = self.norm(x)
        attn_out = self.attn(x_norm, x_norm)
        x = x + self.dropout(attn_out, training=training)
        return x

class ConvolutionModule(tf.keras.layers.Layer):
    def __init__(self, d_model=144, kernel_size=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')
        self.pw_conv1 = tf.keras.layers.Conv1D(d_model * 2, 1, name='pw_conv1')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same', name='dw_conv')
        self.batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm')
        self.pw_conv2 = tf.keras.layers.Conv1D(d_model, 1, name='pw_conv2')
        self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')
    def call(self, x, training=False):
        x = self.norm(x)
        x = self.pw_conv1(x)
        x_glu, x_lin = tf.split(x, 2, axis=-1)
        x = x_glu * tf.nn.sigmoid(x_lin)
        x = self.dw_conv(x)
        x = self.batch_norm(x, training=training)
        x = tf.nn.swish(x)
        x = self.pw_conv2(x)
        x = self.dropout(x, training=training)
        return x

class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model=144, n_head=4, conv_kernel=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ffn1 = FeedForwardModule(d_model, dropout, name='ffn1')
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_head, dropout, name='mhsa')
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout, name='conv')
        self.ffn2 = FeedForwardModule(d_model, dropout, name='ffn2')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')
        self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')
    def call(self, x, training=False, mask=None):
        x = x + 0.5 * self.ffn1(x, training=training)
        x = x + self.mhsa(x, training=training)
        x = x + self.conv(x, training=training)
        x = x + 0.5 * self.ffn2(x, training=training)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return x

class ConformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=144, n_blocks=8, n_head=4,
                 conv_kernel=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.subsampling = ConformerConvSubsampling(d_model, name='subsampling')
        self.linear = tf.keras.layers.Dense(d_model, name='linear')
        self.blocks = [ConformerBlock(d_model, n_head, conv_kernel, dropout,
                                      name=f'conformer_block_{i}' if i > 0 else 'conformer_block')
                       for i in range(n_blocks)]
    def call(self, x, training=False, mask=None):
        x = self.subsampling(x, training=training)
        x = self.linear(x)
        for block in self.blocks:
            x = block(x, training=training)
        return x


def _load_weights_manual(model: tf.keras.Model, filepath: str):
    """Load weights từ HDF5 checkpoint, map tên variables từ model
    sang đường dẫn trong file (do khác TF version nên tên layer không khớp)."""
    # ── Đọc toàn bộ weight data từ file (bỏ qua optimizer) ──
    with h5py.File(filepath, "r") as f:
        ckpt_data = {}
        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset) and not name.startswith("optimizer/"):
                ckpt_data[name] = obj[()]
        f.visititems(_collect)

    # ── Map: model variable name → file dataset path ──
    # Model: "conformer_encoder/subsampling/linear/kernel:0"
    # File:  "layers/conformer_encoder/subsampling/linear/vars/0"
    # Rule:  thêm "layers/" prefix, thay var_name → vars/{idx}
    VAR_TO_IDX = {
        "kernel": "0", "bias": "1",
        "gamma": "0", "beta": "1",
        "moving_mean": "2", "moving_variance": "3",
        "depthwise_kernel": "0",
    }

    # Một số tên layer trong model không khớp với file (TF version khác).
    # File dùng: key_dense, query_dense, value_dense, output_dense
    # TF mới dùng: key, query, value, attention_output
    MHA_REMAP = {
        "key": "key_dense",
        "query": "query_dense",
        "value": "value_dense",
        "attention_output": "output_dense",
    }

    loaded = 0
    errors = []

    for v in model.weights:
        vname = v.name
        stem = re.sub(r":\d+$", "", vname)
        parts = stem.split("/")

        var_name = parts[-1]
        var_idx = VAR_TO_IDX.get(var_name)
        if var_idx is None:
            errors.append(f"Unknown variable type '{var_name}' in {vname}")
            continue

        layer_parts = parts[:-1]

        # Remap MHA internal names (key→key_dense, etc.)
        for i, p in enumerate(layer_parts):
            if p in MHA_REMAP:
                layer_parts[i] = MHA_REMAP[p]

        # Checkpoint có thêm "blocks/" giữa conformer_encoder và conformer_block
        # và thêm "layers/" giữa ffn1/ffn2 và layer con (do list attribute)
        # Model:  conformer_encoder/conformer_block/ffn1/dense/kernel:0
        # File:   layers/conformer_encoder/blocks/conformer_block/ffn1/layers/dense/vars/0
        inserted = []
        for p in layer_parts:
            if p.startswith("conformer_block") and inserted and inserted[-1] == "conformer_encoder":
                inserted.append("blocks")
            # Thêm "layers/" trước layer_normalization, dense, dense_1 ngay sau ffn1/ffn2
            if p in ("layer_normalization", "dense", "dense_1") and inserted and inserted[-1] in ("ffn1", "ffn2"):
                inserted.append("layers")
            inserted.append(p)

        fpath = "layers/" + "/".join(inserted) + f"/vars/{var_idx}"

        if fpath in ckpt_data:
            data = ckpt_data[fpath]
            if list(data.shape) == v.shape.as_list():
                v.assign(data)
                loaded += 1
                continue
            else:
                errors.append(
                    f"Shape mismatch for {fpath}: file {data.shape} vs model {v.shape}"
                )
        else:
            errors.append(f"Path not found: {fpath}")

    if loaded == len(model.weights):
        print(f"   ✅ Loaded {loaded}/{len(model.weights)} weights")
    else:
        print(f"   ⚠ Loaded {loaded}/{len(model.weights)} weights "
              f"({len(errors)} errors)")
        for e in errors:
            print(f"      {e}")


def build_conformer_model(input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    """Rebuild kiến trúc Conformer GIỐNG HỆT notebook training."""
    inp = tf.keras.Input(shape=input_shape)
    backbone = ConformerEncoder(name='conformer_encoder')
    out = backbone(inp)
    out = tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d')(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='dense')(out)
    return tf.keras.Model(inputs=inp, outputs=out)


class DeepfakeDetector:
    """
    Phát hiện giọng nói deepfake dùng Conformer model.
    Score: 0.0 = Real (Bonafide), 1.0 = Fake (Spoof)
    """

    def __init__(self, model_path: str = "models/ckpt.weights.h5"):
        print(f"Loading model weights from: {model_path}")
        self.model = build_conformer_model()
        _load_weights_manual(self.model, model_path)
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
