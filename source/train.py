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
    6. Save results and training plots to results/

Imbalance strategy is controlled by config.yaml:
    imbalance_strategy: class_weights  → class weights passed to model.fit
    imbalance_strategy: oversampling   → balanced dataset, no class weights needed

Usage:
    Press IDE run button or: python source/train.py
"""

import os
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

os.chdir(Path(__file__).resolve().parent.parent)

from source.dataset  import get_datasets
from source.model    import build_model
from source.evaluate import (
    evaluate_model, print_results, save_results,
    save_confusion_matrix, save_roc_curve
)
from settings.SettingsAssistant import CONFIG


def save_plots(history, results_dir: Path):
    """
    Save training/validation loss and accuracy curves to results/.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs    = range(1, len(history.history["loss"]) + 1)

    # ── Loss plot ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.history["loss"],     "b-", label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = results_dir / f"loss_curve_{timestamp}.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  Loss curve saved to: {loss_path}")

    # ── Accuracy plot ──────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.history["accuracy"],     "b-", label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = results_dir / f"accuracy_curve_{timestamp}.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"  Accuracy curve saved to: {acc_path}")


def main():
    # ── Config ────────────────────────────────────────────────────
    epochs      = CONFIG["training"]["epochs"]
    patience    = CONFIG["training"]["early_stopping_patience"]
    save_path   = CONFIG["model"]["save_path"]
    strategy    = CONFIG["training"].get("imbalance_strategy", "class_weights")
    results_dir = Path(CONFIG["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_names = get_datasets()

    # ── Model ─────────────────────────────────────────────────────
    print("Building model...")
    model = build_model(train_backbone=True)
    print(f"  Model: {model.name}")
    print(f"  Total params: {model.count_params():,}\n")

    # ── Class Weights ─────────────────────────────────────────────
    if strategy == "class_weights":
        class_weight = {
            0: 1.0,
            1: 7491 / 3295,
            2: 7491 / 1828,
        }
        print("Class weights:")
        for cls, w in zip(class_names, class_weight.values()):
            print(f"  {cls:<14} : {w:.4f}")
        print()
    else:
        class_weight = None
        print("Imbalance strategy: oversampling — no class weights applied\n")

    # ── Callbacks ─────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
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

    if strategy == "oversampling":
        batch_size      = CONFIG["training"]["batch_size"]
        total_train     = 8828
        steps_per_epoch = total_train // batch_size
    else:
        steps_per_epoch = None

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # ── Save training plots ───────────────────────────────────────
    print("\nSaving training plots...")
    save_plots(history, results_dir)

    # ── Evaluation on test set ────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_ds, class_names)
    print_results(test_results, split_name="Test")
    save_results(test_results, split_name="test")
    save_confusion_matrix(test_results, class_names, results_dir, split_name="test")
    save_roc_curve(test_results, class_names, results_dir, split_name="test")

    # ── Evaluation on val set ─────────────────────────────────────
    print("Evaluating on validation set...")
    val_results = evaluate_model(model, val_ds, class_names)
    print_results(val_results, split_name="Validation")
    save_results(val_results, split_name="val")
    save_confusion_matrix(val_results, class_names, results_dir, split_name="val")
    save_roc_curve(val_results, class_names, results_dir, split_name="val")

    print(f"\nTraining complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()