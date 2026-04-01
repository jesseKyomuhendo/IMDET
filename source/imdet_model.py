import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_augmentation import get_data_augmentation


def srm_layer(x):
    """Apply a simple SRM-like high-pass filter to the RGB image.

    This approximates a noise/residual stream by emphasizing high-frequency content
    (edges, manipulation traces). We use a fixed 3x3 kernel and keep it non-trainable.
    """
    # Simple Laplacian-style high-pass kernel
    kernel = tf.constant(
        [[[-1.0], [-1.0], [-1.0]],
         [[-1.0], [8.0],  [-1.0]],
         [[-1.0], [-1.0], [-1.0]]],
        dtype=tf.float32,
    )  # shape (3, 3, 1)

    # Apply the same kernel to each channel independently
    kernel = tf.repeat(kernel, repeats=3, axis=2)  # (3,3,3)
    kernel = tf.reshape(kernel, (3, 3, 3, 1))      # (3,3,3,in_channels) per-filter

    x = tf.cast(x, tf.float32)
    x = tf.nn.depthwise_conv2d(
        x,
        filter=kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )
    return x


def build_imdet_model(input_shape=(224, 224, 3), train_backbone=False):
    """Build the IMDET two-stream model.

    Architecture (high-level):
    - Shared RGB input
    - RGB stream: ResNet50 (ImageNet, semantic features)
    - Noise stream: SRM-like filter + small CNN backbone
        * Copy-Move head (features)
        * Splicing head (features)
    - Attention fusion between RGB and noise features
    - Final 3-class classifier: Authentic / Copy-Move / Splicing
    """

    # Shared input and augmentation
    inputs = keras.Input(shape=input_shape, name="input_image")
    x_rgb = get_data_augmentation()(inputs)
    x_rgb = layers.Rescaling(1.0 / 255.0)(x_rgb)

    # -----------------
    # RGB stream (semantic)
    # -----------------
    rgb_backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    rgb_backbone.trainable = train_backbone

    rgb_features = rgb_backbone(x_rgb)  # shape (batch, D_rgb)

    # -----------------
    # Noise stream (forensic)
    # -----------------
    x_noise = layers.Lambda(srm_layer, name="srm_filter")(inputs)

    # Small CNN to extract shared noise features
    n = layers.Conv2D(32, 3, padding="same", activation="relu")(x_noise)
    n = layers.MaxPooling2D()(n)
    n = layers.Conv2D(64, 3, padding="same", activation="relu")(n)
    n = layers.MaxPooling2D()(n)
    n = layers.Conv2D(128, 3, padding="same", activation="relu")(n)
    n = layers.GlobalAveragePooling2D()(n)

    # Copy-Move head
    cm = layers.Dense(128, activation="relu", name="copy_move_head_dense1")(n)
    cm = layers.Dense(64, activation="relu", name="copy_move_head_dense2")(cm)

    # Splicing head
    sp = layers.Dense(128, activation="relu", name="splicing_head_dense1")(n)
    sp = layers.Dense(64, activation="relu", name="splicing_head_dense2")(sp)

    noise_features = layers.Concatenate(name="noise_features")([cm, sp])

    # -----------------
    # Attention fusion
    # -----------------
    fused_input = layers.Concatenate(name="fusion_concat")([
        rgb_features,
        noise_features,
    ])

    # Two attention weights: one for RGB, one for Noise
    att_logits = layers.Dense(2, activation="softmax", name="stream_attention")(fused_input)
    att_rgb = layers.Lambda(lambda t: t[:, 0:1], name="att_rgb")(att_logits)
    att_noise = layers.Lambda(lambda t: t[:, 1:2], name="att_noise")(att_logits)

    # Reshape for broadcasting
    att_rgb_exp = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(att_rgb)
    att_noise_exp = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(att_noise)

    rgb_weighted = layers.Multiply()([att_rgb_exp, tf.expand_dims(rgb_features, axis=1)])
    noise_weighted = layers.Multiply()([
        att_noise_exp,
        tf.expand_dims(noise_features, axis=1),
    ])

    fused = layers.Concatenate(axis=-1, name="fused_features")([
        layers.Flatten()(rgb_weighted),
        layers.Flatten()(noise_weighted),
    ])

    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.5)(fused)

    # -----------------
    # Final classifier (3 classes)
    # -----------------
    outputs = layers.Dense(3, activation="softmax", name="classifier")(fused)

    model = keras.Model(inputs=inputs, outputs=outputs, name="IMDET_two_stream")
    return model


if __name__ == "__main__":
    model = build_imdet_model()
    model.summary()
