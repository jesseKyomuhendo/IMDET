"""
model.py
--------
Defines the two-stream hybrid CNN architecture for image manipulation detection.

Architecture:
    Stream 1 (RGB)   : ResNet50 backbone pretrained on ImageNet (frozen by default)
                       Extracts semantic and visual features
    Stream 2 (Noise) : SRM (Spatial Rich Model) constrained conv layers
                       Extracts noise residuals and texture artefacts
                       Two specialised heads: copy-move and splicing
    Attention        : Dynamically weights RGB vs noise stream contribution
    Fusion           : Weighted combination of both streams
    Classifier head  : Fully connected layers → 3-class softmax output
                       Classes: authentic | copy_move | splicing

Functions:
    build_model : builds and returns the full two-stream Keras model

Usage (from other modules):
    from source.model import build_model
    model = build_model()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

from settings.SettingsAssistant import CONFIG


# ── SRM Filter ─────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
def _srm_conv_layer(inp):
    """
    Apply a depthwise Laplacian SRM filter to extract noise residuals.
    Input shape : (batch, H, W, 3)
    Output shape: (batch, H, W, 3)
    """
    kernel = tf.constant(
        [[[-1.0], [-1.0], [-1.0]],
         [[-1.0], [ 8.0], [-1.0]],
         [[-1.0], [-1.0], [-1.0]]],
        dtype=tf.float32
    )
    kernel = tf.repeat(kernel, repeats=3, axis=2)   # (3, 3, 3)
    kernel = tf.reshape(kernel, (3, 3, 3, 1))        # (3, 3, 3, 1)

    inp = tf.cast(inp, tf.float32)
    inp = tf.nn.depthwise_conv2d(inp, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    return inp


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_model(train_backbone=False):
    """
    Build and return the two-stream hybrid CNN model.

    Args:
        train_backbone : whether to unfreeze ResNet50 weights for fine-tuning.
                         Default False — backbone frozen, only head trains.

    Returns:
        Compiled Keras model ready for training.
    """
    cfg         = CONFIG["model"]
    img_size    = CONFIG["image"]["size"]
    num_classes = cfg["num_classes"]
    lr          = CONFIG["training"]["learning_rate"]

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")

    # Normalize pixel values [0-255] → [0.0-1.0] inside the model
    rescaled = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # ── Stream 1: RGB ──────────────────────────────────────────────
    backbone = ResNet50(
        include_top=False,
        weights="imagenet" if cfg["pretrained"] else None,
        pooling="avg"
    )
    backbone.trainable = train_backbone

    rgb_features = backbone(rescaled)   # Shape: (batch, 2048)

    # ── Stream 2: Noise / SRM ──────────────────────────────────────
    x_noise = layers.Lambda(
        _srm_conv_layer,
        output_shape=lambda s: s,
        name="srm_filters"
    )(rescaled)
    x_noise = layers.BatchNormalization(name="noise_bn0")(x_noise)

    n = layers.Conv2D(32,  3, padding="same", activation="relu", name="noise_conv1")(x_noise)
    n = layers.MaxPooling2D(name="noise_pool1")(n)

    n = layers.Conv2D(64,  3, padding="same", activation="relu", name="noise_conv2")(n)
    n = layers.MaxPooling2D(name="noise_pool2")(n)

    n = layers.Conv2D(128, 3, padding="same", activation="relu", name="noise_conv3")(n)
    n = layers.GlobalAveragePooling2D(name="noise_gap")(n)   # (batch, 128)

    # Specialised copy-move head
    cm = layers.Dense(128, activation="relu", name="cm_fc1")(n)
    cm = layers.Dense(64,  activation="relu", name="cm_fc2")(cm)

    # Specialised splicing head
    sp = layers.Dense(128, activation="relu", name="sp_fc1")(n)
    sp = layers.Dense(64,  activation="relu", name="sp_fc2")(sp)

    noise_features = layers.Concatenate(name="noise_fusion")([cm, sp])  # (batch, 128)

    # ── Attention ──────────────────────────────────────────────────
    fused_input = layers.Concatenate(name="pre_attention")([rgb_features, noise_features])

    att       = layers.Dense(2, activation="softmax", name="attention")(fused_input)
    att_rgb   = layers.Lambda(lambda t: t[:, 0:1], output_shape=lambda s: (s[0], 1), name="att_rgb")(att)
    att_noise = layers.Lambda(lambda t: t[:, 1:2], output_shape=lambda s: (s[0], 1), name="att_noise")(att)

    rgb_exp   = layers.Reshape((1, 2048), name="rgb_exp")(rgb_features)
    noise_exp = layers.Reshape((1, 128),  name="noise_exp")(noise_features)

    att_rgb_exp   = layers.Reshape((1, 1), name="att_rgb_exp")(att_rgb)
    att_noise_exp = layers.Reshape((1, 1), name="att_noise_exp")(att_noise)

    rgb_weighted   = layers.Multiply(name="rgb_weighted")([rgb_exp,   att_rgb_exp])
    noise_weighted = layers.Multiply(name="noise_weighted")([noise_exp, att_noise_exp])

    # ── Fusion ─────────────────────────────────────────────────────
    fused = layers.Concatenate(axis=-1, name="fusion")([
        layers.Flatten(name="rgb_flat")(rgb_weighted),
        layers.Flatten(name="noise_flat")(noise_weighted),
    ])

    # ── Classifier Head ────────────────────────────────────────────
    fused   = layers.Dense(128, activation="relu", name="fc1")(fused)
    fused   = layers.Dropout(0.5, name="dropout")(fused)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(fused)

    # ── Compile ────────────────────────────────────────────────────
    m = keras.Model(inputs=inputs, outputs=outputs, name="IMD_TwoStream")
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return m


if __name__ == "__main__":
    summary_model = build_model()
    summary_model.summary()