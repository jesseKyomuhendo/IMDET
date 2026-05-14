"""
evaluate.py
-----------
Reusable evaluation functions for the IMD project.
Called by both train.py (on validation set) and test.py (on test set).

Functions:
    evaluate_model          : runs model on a dataset, returns all metrics
    print_results           : prints a formatted summary of results
    save_results            : saves results to a JSON file in results/
    save_confusion_matrix   : saves seaborn heatmap confusion matrix
    save_roc_curve          : saves per-class ROC curve plot

Usage (from other modules):
    from source.evaluate import evaluate_model, print_results, save_results
    from source.evaluate import save_confusion_matrix, save_roc_curve
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

from settings.SettingsAssistant import CONFIG


def evaluate_model(model, dataset, class_names):
    """
    Run model on a tf.data.Dataset and compute all evaluation metrics.

    Args:
        model       : trained Keras model
        dataset     : tf.data.Dataset (batched, not shuffled)
        class_names : list of class name strings e.g. ['authentic', 'copy_move', 'splicing']

    Returns:
        dict with keys: accuracy, f1_per_class, f1_macro, auc, confusion_matrix,
                        report, y_true, y_pred, y_prob
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in dataset:
        probs  = model(images, training=False).numpy()
        preds  = np.argmax(probs, axis=1)
        labels = labels.numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # F1 per class and macro average
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(len(class_names))))
    f1_macro     = f1_score(all_labels, all_preds, average="macro")

    # AUC-ROC (one-vs-rest for multi-class)
    try:
        auc_score = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_score = None

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    # Full classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )

    return {
        "accuracy"        : round(float(accuracy), 4),
        "f1_per_class"    : {class_names[i]: round(float(f1_per_class[i]), 4) for i in range(len(class_names))},
        "f1_macro"        : round(float(f1_macro), 4),
        "auc"             : round(float(auc_score), 4) if auc_score is not None else "N/A",
        "confusion_matrix": cm.tolist(),
        "report"          : report,
        "y_true"          : all_labels.tolist(),
        "y_pred"          : all_preds.tolist(),
        "y_prob"          : all_probs.tolist(),
    }


def print_results(results, split_name="Test"):
    """Print a formatted summary of evaluation results."""
    print(f"\n{'─' * 50}")
    print(f"  Evaluation Results — {split_name} Set")
    print(f"{'─' * 50}")
    print(f"  Accuracy   : {results['accuracy']:.4f}")
    print(f"  F1 (macro) : {results['f1_macro']:.4f}")
    print(f"  AUC-ROC    : {results['auc']}")
    print(f"\n  F1 per class:")
    for cls, score in results["f1_per_class"].items():
        print(f"    {cls:<14} : {score:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
    print(f"\n  Classification Report:")
    print(results["report"])
    print(f"{'─' * 50}\n")


def save_results(results, split_name="test"):
    """Save evaluation results to a JSON file in the results directory."""
    results_dir = Path(CONFIG["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = results_dir / f"{split_name}_results_{timestamp}.json"

    # Save without raw arrays to keep JSON clean
    clean = {k: v for k, v in results.items() if k not in ("y_true", "y_pred", "y_prob")}
    with open(filename, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"  Results saved to: {filename}")
    return filename


def save_confusion_matrix(results, class_names, results_dir: Path, split_name="test"):
    """
    Save a seaborn heatmap confusion matrix to results/.
    Single matrix showing all classes — matches the style in the team's notebook.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm        = np.array(results["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    path = results_dir / f"{split_name}_confusion_matrix_{timestamp}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {path}")


def save_roc_curve(results, class_names, results_dir: Path, split_name="test"):
    """
    Save a per-class ROC curve plot to results/.
    Shows one curve per class with AUC score in the legend.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    y_true     = np.array(results["y_true"])
    y_prob     = np.array(results["y_prob"])
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure(figsize=(7, 6))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Per-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    path = results_dir / f"{split_name}_roc_curve_{timestamp}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve saved to: {path}")