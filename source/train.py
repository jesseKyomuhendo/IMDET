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

Usage:
    Press IDE run button or: python source/train.py
"""

import os
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on Colab and headless servers
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

os.chdir(Path(__file__).resolve().parent.parent)

from source.dataset  import get_datasets
from source.model    import build_model
from source.evaluate import evaluate_model, print_results, save_results
from settings.SettingsAssistant import CONFIG


def save_plots(history, results_dir: Path):
    """
    Save training/validation loss and accuracy curves to results/.
    Useful for the report to show learning progress and detect overfitting.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs     = range(1, len(history.history["loss"]) + 1)

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
    plt.savefig(loss_path)
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
    plt.savefig(acc_path)
    plt.close()
    print(f"  Accuracy curve saved to: {acc_path}")


def save_confusion_matrix_plots(results, class_names, results_dir: Path, split_name="test"):
    """
    Save per-class confusion matrix visualisations to results/.
    Shows TP, FP, TN, FN for each class individually.
    """
    from sklearn.metrics import multilabel_confusion_matrix
    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rebuild y_true and y_pred from confusion matrix
    cm        = results["confusion_matrix"]
    n_classes = len(class_names)

    # Convert multi-class CM to per-class binary CMs
    y_true_bin = []
    y_pred_bin = []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            y_true_bin.extend([true_idx] * count)
            y_pred_bin.extend([pred_idx] * count)

    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin,
                                      labels=list(range(n_classes)))

    for i, binary_cm in enumerate(mcm):
        plt.figure(figsize=(5, 4))
        plt.imshow(binary_cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix — {class_names[i]}")
        plt.colorbar()

        tick_marks = np.arange(2)
        axis_labels = ["Not class", "Class"]
        plt.xticks(tick_marks, axis_labels)
        plt.yticks(tick_marks, axis_labels)

        label_matrix = [["TN", "FP"], ["FN", "TP"]]
        for x in range(2):
            for y in range(2):
                plt.text(y, x, f"{label_matrix[x][y]}\n{binary_cm[x, y]}",
                         ha="center", va="center", fontsize=12)

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plot_path = results_dir / f"{split_name}_cm_{class_names[i]}_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  Confusion matrix ({class_names[i]}) saved to: {plot_path}")


def main():
    # ── Config ────────────────────────────────────────────────────
    epochs      = CONFIG["training"]["epochs"]
    patience    = CONFIG["training"]["early_stopping_patience"]
    save_path   = CONFIG["model"]["save_path"]
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
    # Handles class imbalance: authentic=7491, copy_move=3295, splicing=1828
    # Higher weight = model penalised more for getting that class wrong
    class_weight = {
        0: 1.0,            # authentic  — majority class, no boost
        1: 7491 / 3295,    # copy_move  — ~2.3x weight
        2: 7491 / 1828,    # splicing   — ~4.1x weight
    }
    print("Class weights:")
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

    # ── Save training plots ───────────────────────────────────────
    print("\nSaving training plots...")
    save_plots(history, results_dir)

    # ── Evaluation on test set ────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_ds, class_names)
    print_results(test_results, split_name="Test")
    save_results(test_results, split_name="test")
    save_confusion_matrix_plots(test_results, class_names, results_dir, split_name="test")

    # ── Evaluation on val set ─────────────────────────────────────
    print("Evaluating on validation set...")
    val_results = evaluate_model(model, val_ds, class_names)
    print_results(val_results, split_name="Validation")
    save_results(val_results, split_name="val")
    save_confusion_matrix_plots(val_results, class_names, results_dir, split_name="val")

    print(f"\nTraining complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()