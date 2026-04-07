"""
test.py
-------
Loads the trained IMD model and evaluates it on the test set.

Requirements:
    - Trained model must exist at path defined in config.yaml (model.save_path)
    - Test split CSV must exist at data/splits/test.csv
    - Dataset must be placed as defined in config.yaml (data.casia_dir)

Usage:
    python test.py
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

os.chdir(Path(__file__).resolve().parent)

from source.model    import _srm_conv_layer
from source.dataset  import get_datasets
from source.evaluate import evaluate_model, print_results, save_results
from settings.SettingsAssistant import CONFIG


def main():
    save_path = CONFIG["model"]["save_path"]

    # ── Check model exists ────────────────────────────────────────
    if not Path(save_path).exists():
        print(f"\nError: No trained model found at '{save_path}'")
        print("Please train the model first by running: python source/train.py")
        return

    # ── Load model ────────────────────────────────────────────────
    print(f"\nLoading model from: {save_path}")
    model = keras.models.load_model(
        save_path,
        custom_objects={"_srm_conv_layer": _srm_conv_layer}
    )
    print(f"  Model loaded successfully: {model.name}")
    print(f"  Total params: {model.count_params():,}\n")

    # ── Load test dataset ─────────────────────────────────────────
    _, _, test_ds, class_names = get_datasets()

    # ── Evaluate ──────────────────────────────────────────────────
    print("Running evaluation on test set...")
    results = evaluate_model(model, test_ds, class_names)

    print_results(results, split_name="Test")
    save_results(results, split_name="test")


if __name__ == "__main__":
    main()