"""
dataset.py
----------
Builds tf.data.Dataset pipelines from the CSV split files.
Reads train.csv, val.csv, test.csv and returns batched datasets
ready for training and evaluation.

Functions:
    build_dataset   : builds a tf.data.Dataset from a CSV file
    get_datasets    : returns train, val, test datasets in one call

Usage (from other modules):
    from source.dataset import get_datasets
    train_ds, val_ds, test_ds, class_names = get_datasets()
"""

import os
import csv
import numpy as np
from pathlib import Path

import tensorflow as tf

from settings.SettingsAssistant import CONFIG


# ── Label mapping ──────────────────────────────────────────────────
# Map class name string → integer index (order matches config.yaml classes list)
CLASS_NAMES = CONFIG["classes"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}


def _load_image(filepath, label):
    """
    Load and preprocess a single image.
    - Reads file (supports jpg, tif, bmp, png)
    - Resizes to image.size x image.size
    - Normalizes to [0, 1]
    """
    size = CONFIG["image"]["size"]

    raw   = tf.io.read_file(filepath)

    # Decode — try jpeg first, fall back to png/bmp via decode_image
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [size, size])
    image.set_shape([size, size, 3])

    return image, label


def _augment(image, label):
    """
    Apply random augmentations during training only.
    - Random horizontal flip
    - Random vertical flip
    - Random brightness adjustment
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _read_csv(csv_path: Path):
    """Read a CSV file and return (filepaths, labels) as numpy arrays."""
    filepaths = []
    labels    = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepaths.append(row["filepath"])
            labels.append(CLASS_TO_IDX[row["label"]])

    return np.array(filepaths), np.array(labels, dtype=np.int32)


def build_dataset(csv_path: Path, augment: bool = False):
    """
    Build a tf.data.Dataset from a CSV split file.

    Args:
        csv_path : path to the CSV file (train.csv / val.csv / test.csv)
        augment  : whether to apply data augmentation (training only)

    Returns:
        Batched tf.data.Dataset of (image, label) pairs
    """
    batch_size = CONFIG["training"]["batch_size"]
    seed       = CONFIG["split"]["seed"]

    filepaths, labels = _read_csv(csv_path)

    print(f"  Loaded {len(filepaths)} samples from {csv_path.name}")

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if augment:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=seed)

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_datasets():
    """
    Build and return train, val, and test datasets.

    Returns:
        train_ds   : augmented and shuffled training dataset
        val_ds     : validation dataset (no augmentation)
        test_ds    : test dataset (no augmentation)
        class_names: list of class name strings
    """
    split_dir = Path(CONFIG["data"]["split_dir"])

    print("\nBuilding datasets...")
    train_ds = build_dataset(split_dir / "train.csv", augment=True)
    val_ds   = build_dataset(split_dir / "val.csv",   augment=False)
    test_ds  = build_dataset(split_dir / "test.csv",  augment=False)

    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Mapping : {CLASS_TO_IDX}\n")

    return train_ds, val_ds, test_ds, CLASS_NAMES