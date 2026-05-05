import os
import os

import tensorflow as tf
from tensorflow import keras

# Configuration
DATA_DIR = os.path.join("..", "data")
CLASS_NAMES = ["authentic", "copy_move", "splicing"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 123


def compute_class_counts_and_weights(data_dir: str = DATA_DIR):
    """Count images per class and compute inverse-frequency class weights."""
    counts = []
    for cls in CLASS_NAMES:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder not found: {folder}")

        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
        counts.append(len(files))

    max_count = max(counts) if counts else 0
    class_weights = {
        idx: (max_count / count) if count > 0 else 0.0
        for idx, count in enumerate(counts)
    }
    return counts, class_weights


def create_tf_datasets(
    data_dir: str = DATA_DIR,
    image_size=IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VAL_SPLIT,
    seed: int = SEED,
):
    """Create TensorFlow training and validation datasets from folders."""
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds


if __name__ == "__main__":
    # 1) Compute and print class counts + weights
    counts, class_weights = compute_class_counts_and_weights(DATA_DIR)

    print("Class counts:")
    for name, count in zip(CLASS_NAMES, counts):
        print(f"  {name}: {count}")

    print("\nClass weights (for later use in model.fit):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx} ({name}): {class_weights[idx]}")

    # 2) Build TensorFlow datasets (no model here)
    train_ds, val_ds = create_tf_datasets(DATA_DIR)

    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = tf.data.experimental.cardinality(val_ds).numpy()

    print("\nPrepared TensorFlow datasets:")
    print(f"  Train batches: {train_batches}")
    print(f"  Val batches:   {val_batches}")