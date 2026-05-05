import tensorflow as tf
from tensorflow import keras


# Simple data augmentation pipeline for IMDET images
# You can import `get_data_augmentation()` in your model script


def get_data_augmentation():
    """Return a Keras Sequential with common augmentations.

    This does NOT change the dataset on disk; it only applies
    random transformations during training to reduce overfitting.
    """

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


if __name__ == "__main__":
    # Small demo: apply augmentation to one dummy batch
    aug = get_data_augmentation()
    dummy_images = tf.random.uniform(shape=(4, 224, 224, 3))  # 4 random images
    augmented = aug(dummy_images, training=True)
    print("Input shape:", dummy_images.shape)
    print("Augmented shape:", augmented.shape)
    print("Data augmentation pipeline is ready to be used in your model.")