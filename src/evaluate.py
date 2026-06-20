import os
import logging
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

logger = logging.getLogger("ModelEvaluation")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes key performance metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Generates and saves a confusion matrix plot.
    """
    logger.info(f"Plotting confusion matrix to {output_path}")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Non-Fraud", "Fraud"], 
        yticklabels=["Non-Fraud", "Fraud"]
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> None:
    """
    Generates and saves an ROC curve plot.
    """
    logger.info(f"Plotting ROC curve to {output_path}")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> None:
    """
    Generates and saves a Precision-Recall curve plot.
    """
    logger.info(f"Plotting PR curve to {output_path}")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="green", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
