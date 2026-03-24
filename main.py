import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# Config
# -----------------------------
DATASET_PATH = "data/Prepared Datasets/CASIA2.0"

IMAGE_SIZE = 128
BATCH_SIZE = 2
NUM_CLASSES = 3
EPOCHS = 1
MAX_IMAGES = 30
VAL_SPLIT = 0.2
SEED = 42


# -----------------------------
# Dataset loading
# -----------------------------
def load_dataset(path, image_size=128, batch_size=2):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )
    return dataset


def limit_dataset_exact_images(dataset, max_images):
    return dataset.unbatch().take(max_images)


def split_dataset_by_count(dataset, max_images, val_split, batch_size):
    val_images = max(1, int(max_images * val_split))
    train_images = max_images - val_images

    train_dataset = dataset.take(train_images).batch(batch_size, drop_remainder=False)
    val_dataset = dataset.skip(train_images).take(val_images).batch(batch_size, drop_remainder=False)

    train_steps = math.ceil(train_images / batch_size)
    val_steps = math.ceil(val_images / batch_size)

    return train_dataset, val_dataset, train_steps, val_steps, train_images, val_images


def normalize_dataset(dataset):
    def map_fn(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def make_dual_input(dataset):
    def map_fn(images, labels):
        blurred = tf.nn.avg_pool2d(images, ksize=3, strides=1, padding="SAME")
        residual = images - blurred
        return (images, residual), labels

    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def prepare_dataset(dataset):
    return dataset.prefetch(tf.data.AUTOTUNE)


# -----------------------------
# Model
# -----------------------------
def make_resnet_branch(branch_name, image_size):
    return keras.applications.ResNet50(
        include_top=False,
        weights=None,   
        pooling="avg",
        input_shape=(image_size, image_size, 3),
        name=branch_name
    )


def build_dual_resnet50(num_classes=3, image_size=128):
    print("Building model...")

    rgb_input = keras.Input(shape=(image_size, image_size, 3), name="rgb_input")
    aux_input = keras.Input(shape=(image_size, image_size, 3), name="aux_input")

    rgb_base = make_resnet_branch("rgb_resnet50", image_size)
    aux_base = make_resnet_branch("aux_resnet50", image_size)

    rgb_features = rgb_base(rgb_input)
    aux_features = aux_base(aux_input)

    fused = layers.Concatenate(name="fusion")([rgb_features, aux_features])
    x = layers.Dense(128, activation="relu", name="fc1")(fused)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(
        inputs=[rgb_input, aux_input],
        outputs=output,
        name="dual_resnet50_model"
    )
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading dataset...")
    dataset = load_dataset(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Class names:", dataset.class_names)

    print(f"Limiting dataset to {MAX_IMAGES} images...")
    dataset = limit_dataset_exact_images(dataset, MAX_IMAGES)

    print("Splitting into train/validation...")
    train_dataset, val_dataset, train_steps, val_steps, train_images, val_images = split_dataset_by_count(
        dataset,
        MAX_IMAGES,
        VAL_SPLIT,
        BATCH_SIZE
    )

    print(f"Train images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Train batches: {train_steps}")
    print(f"Validation batches: {val_steps}")

    print("Normalizing dataset...")
    train_dataset = normalize_dataset(train_dataset)
    val_dataset = normalize_dataset(val_dataset)

    print("Creating dual-input dataset with residual second branch...")
    train_dataset = make_dual_input(train_dataset)
    val_dataset = make_dual_input(val_dataset)

    train_dataset = prepare_dataset(train_dataset)
    val_dataset = prepare_dataset(val_dataset)

    print("Building dual model...")
    model = build_dual_resnet50(
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE
    )

    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Testing one forward pass...")
    for (rgb_batch, aux_batch), label_batch in train_dataset.take(1):
        preds = model.predict([rgb_batch, aux_batch], verbose=1)
        print("Prediction shape:", preds.shape)
        print("Label batch shape:", label_batch.shape)

    print(f"Training for {EPOCHS} epoch(s)...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=1
    )

    print("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_dataset, steps=val_steps, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    print("Finished successfully.")


if __name__ == "__main__":
    main()