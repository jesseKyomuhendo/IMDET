"""
train.py
--------
Main training script for the IMD two-stream model.
Pulls together dataset.py, model.py, and evaluate.py.

Steps:
    1. Load datasets (train, val, test) from CSV splits
    2. Build two-stream model
    3. Train with early stopping and model checkpointing
    4. Evaluate on validation set after each epoch
    5. Evaluate final model on test set
    6. Save results to results/

Usage:
    Press IDE run button or: python source/train.py
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

os.chdir(Path(__file__).resolve().parent.parent)

from source.dataset  import get_datasets
from source.model    import build_model
from source.evaluate import evaluate_model, print_results, save_results
from settings.SettingsAssistant import CONFIG


def main():
    # ── Config ────────────────────────────────────────────────────
    epochs    = CONFIG["training"]["epochs"]
    patience  = CONFIG["training"]["early_stopping_patience"]
    save_path = CONFIG["model"]["save_path"]

    # ── Datasets ──────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_names = get_datasets()

    # ── Model ─────────────────────────────────────────────────────
    print("Building model...")
    model = build_model(train_backbone=False)
    print(f"  Model: {model.name}")
    print(f"  Total params: {model.count_params():,}\n")

    # ── Class Weights ─────────────────────────────────────────────
    # Handles class imbalance: authentic=7491, copy_move=3295, splicing=1828
    # Higher weight = model penalised more for getting that class wrong
    class_weight = {
        0: 1.0,            # authentic  — majority class, no boost
        1: 7491 / 3295,    # copy_move  — ~2.3x weight
        2: 7491 / 1828,    # splicing   — ~4.1x weight
    }
    print(f"Class weights:")
    for cls, w in zip(class_names, class_weight.values()):
        print(f"  {cls:<14} : {w:.4f}")
    print()

    # ── Callbacks ─────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save best model based on val_loss
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        # Stop training if val_loss does not improve for N epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── Training ──────────────────────────────────────────────────
    print(f"Starting training — {epochs} epochs max, early stopping patience={patience}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # ── Evaluation on test set ────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_ds, class_names)
    print_results(test_results, split_name="Test")
    save_results(test_results, split_name="test")

    # ── Evaluation on val set ─────────────────────────────────────
    print("Evaluating on validation set...")
    val_results = evaluate_model(model, val_ds, class_names)
    print_results(val_results, split_name="Validation")
    save_results(val_results, split_name="val")

    print(f"\nTraining complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()