"""
model.py
--------
Defines the two-stream hybrid CNN architecture for image manipulation detection.

Architecture:
    Stream 1 (RGB)   : ResNet50 backbone pretrained on ImageNet
                       Extracts semantic and visual features
    Stream 2 (Noise) : SRM (Spatial Rich Model) constrained conv layers
                       Extracts noise residuals and texture artefacts
    Fusion           : Concatenates both streams
    Classifier head  : Fully connected layers → 3-class softmax output
                       Classes: authentic | copy_move | splicing

Functions:
    build_srm_filters : returns the SRM filter bank as a numpy array
    build_model       : builds and returns the full two-stream Keras model

Usage (from other modules):
    from source.model import build_model
    model = build_model()
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from settings.SettingsAssistant import CONFIG


# ── SRM Filter Bank ────────────────────────────────────────────────────────────
# SRM (Spatial Rich Model) filters extract noise residuals left by manipulation.
# These are fixed, non-trainable filters — they are not learned during training.

def build_srm_filters():
    """
    Returns a set of 3 SRM high-pass filters as a numpy array.
    Shape: (3, 3, 1, 3) — 3x3 kernel, 1 input channel, 3 filters.
    Applied per channel independently in the noise stream.
    """
    # Filter 1: Simple high-pass (centre surround)
    f1 = np.array([
        [ 0,  0,  0],
        [ 0,  1, -1],
        [ 0,  0,  0]
    ], dtype=np.float32)

    # Filter 2: Horizontal edge
    f2 = np.array([
        [ 0,  0,  0],
        [ 0,  2, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    # Filter 3: Laplacian (detects noise residuals in all directions)
    f3 = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32) / 8.0

    # Stack into shape (3, 3, 1, 3)
    filters = np.stack([f1, f2, f3], axis=-1)        # (3, 3, 3)
    filters = np.expand_dims(filters, axis=2)         # (3, 3, 1, 3)
    filters = np.transpose(filters, (0, 1, 2, 3))     # (3, 3, 1, 3)

    return filters


tf.keras.utils.register_keras_serializable()
def _srm_conv_layer(x):
    """
    Apply SRM filters to each colour channel independently,
    then concatenate the results.
    Input shape : (batch, H, W, 3)
    Output shape: (batch, H, W, 9)  — 3 filters x 3 channels
    """
    srm_filters = build_srm_filters()  # (3, 3, 1, 3)

    channels = tf.split(x, num_or_size_splits=3, axis=-1)
    outputs  = []

    for ch in channels:
        filtered = tf.nn.conv2d(
            ch,
            filters=tf.constant(srm_filters),
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        outputs.append(filtered)

    return tf.concat(outputs, axis=-1)

# ── Model Builder ──────────────────────────────────────────────────────────────

def build_model():
    """
    Build and return the two-stream hybrid CNN model.

    Returns:
        Compiled Keras model ready for training.
    """
    cfg        = CONFIG["model"]
    img_size   = CONFIG["image"]["size"]
    num_classes = cfg["num_classes"]
    lr         = CONFIG["training"]["learning_rate"]

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # ── Stream 1: RGB ──────────────────────────────────────────────
    # ResNet50 pretrained on ImageNet, top layers removed
    # Preprocess input using ResNet50's expected normalisation
    rgb_preprocessed = keras.applications.resnet50.preprocess_input(inputs)

    backbone = ResNet50(
        include_top=False,
        weights="imagenet" if cfg["pretrained"] else None,
        input_tensor=rgb_preprocessed,
        pooling="avg"          # Global average pooling → flat feature vector
    )

    # Freeze backbone initially — fine-tune later if needed
    backbone.trainable = True

    rgb_features = backbone.output   # Shape: (batch, 2048)

    # ── Stream 2: Noise / SRM ──────────────────────────────────────
    # Apply fixed SRM filters to extract noise residuals
    noise = layers.Lambda(_srm_conv_layer, name="srm_filters")(inputs)
                                                        # (batch, H, W, 9)

    # Learnable conv layers on top of SRM output
    noise = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                          name="noise_conv1")(noise)
    noise = layers.BatchNormalization(name="noise_bn1")(noise)
    noise = layers.MaxPooling2D((2, 2), name="noise_pool1")(noise)

    noise = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                          name="noise_conv2")(noise)
    noise = layers.BatchNormalization(name="noise_bn2")(noise)
    noise = layers.MaxPooling2D((2, 2), name="noise_pool2")(noise)

    noise = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                          name="noise_conv3")(noise)
    noise = layers.BatchNormalization(name="noise_bn3")(noise)

    noise = layers.GlobalAveragePooling2D(name="noise_gap")(noise)
                                                        # (batch, 128)

    # ── Fusion ─────────────────────────────────────────────────────
    # Concatenate RGB and noise feature vectors
    fused = layers.Concatenate(name="fusion")([rgb_features, noise])
                                                        # (batch, 2048 + 128)

    # ── Classifier Head ────────────────────────────────────────────
    x = layers.Dense(512, activation="relu", name="fc1")(fused)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    # ── Compile ────────────────────────────────────────────────────
    model = keras.Model(inputs=inputs, outputs=outputs, name="IMD_TwoStream")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Quick sanity check — print model summary
    model = build_model()
    model.summary()