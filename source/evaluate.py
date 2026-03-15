"""
evaluate.py
-----------
Reusable evaluation functions for the IMD project.
Called by both train.py (on validation set) and test.py (on test set).

Functions:
    evaluate_model  : runs model on a dataloader, returns all metrics
    print_results   : prints a formatted summary of results
    save_results    : saves results to a JSON file in results/

Usage (from other modules):
    from source.evaluate import evaluate_model, print_results, save_results
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from settings.SettingsAssistant import CONFIG


def evaluate_model(model, dataloader, device, class_names):
    """
    Run model on a dataloader and compute all evaluation metrics.

    Args:
        model       : trained PyTorch model
        dataloader  : DataLoader for the split to evaluate
        device      : torch.device (cpu or cuda)
        class_names : list of class name strings e.g. ['authentic', 'copy_move', 'splicing']

    Returns:
        dict with keys: accuracy, f1_per_class, f1_macro, auc, confusion_matrix, report
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        # Can fail if a class has no samples in the split
        auc = None

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    # Full classification report (precision, recall, f1 per class)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )

    return {
        "accuracy"     : round(float(accuracy), 4),
        "f1_per_class" : {class_names[i]: round(float(f1_per_class[i]), 4) for i in range(len(class_names))},
        "f1_macro"     : round(float(f1_macro), 4),
        "auc"          : round(float(auc), 4) if auc is not None else "N/A",
        "confusion_matrix": cm.tolist(),
        "report"       : report,
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

    # confusion_matrix is already a list (JSON serializable)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {filename}")
    return filename


