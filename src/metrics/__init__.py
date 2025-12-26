"""Evaluation metrics for drug-drug interaction prediction."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, probabilities: Optional[torch.Tensor] = None) -> None:
        """Update stored predictions and labels.
        
        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            probabilities: Predicted probabilities (optional)
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            logger.warning("No predictions available for metric computation")
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, average='binary', zero_division=0)
        metrics['f1'] = f1_score(labels, predictions, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # ROC and PR metrics (if probabilities available)
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            
            # Handle both single probability and probability array cases
            if probabilities.ndim == 1:
                # Single probability per sample
                prob_positive = probabilities
            else:
                # Probability array - take probability of positive class
                prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            
            try:
                metrics['roc_auc'] = roc_auc_score(labels, prob_positive)
                metrics['pr_auc'] = average_precision_score(labels, prob_positive)
            except ValueError as e:
                logger.warning(f"Could not compute ROC/PR AUC: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.predictions:
            logger.warning("No predictions available for confusion matrix")
            return
        
        cm = confusion_matrix(self.labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Interaction', 'Interaction'],
                   yticklabels=['No Interaction', 'Interaction'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.probabilities:
            logger.warning("No probabilities available for ROC curve")
            return
        
        probabilities = np.array(self.probabilities)
        labels = np.array(self.labels)
        
        # Handle probability format
        if probabilities.ndim == 1:
            prob_positive = probabilities
        else:
            prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        fpr, tpr, _ = roc_curve(labels, prob_positive)
        roc_auc = roc_auc_score(labels, prob_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.probabilities:
            logger.warning("No probabilities available for PR curve")
            return
        
        probabilities = np.array(self.probabilities)
        labels = np.array(self.labels)
        
        # Handle probability format
        if probabilities.ndim == 1:
            prob_positive = probabilities
        else:
            prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        precision, recall, _ = precision_recall_curve(labels, prob_positive)
        pr_auc = average_precision_score(labels, prob_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, save_path: Optional[str] = None) -> None:
        """Plot calibration curve.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.probabilities:
            logger.warning("No probabilities available for calibration curve")
            return
        
        probabilities = np.array(self.probabilities)
        labels = np.array(self.labels)
        
        # Handle probability format
        if probabilities.ndim == 1:
            prob_positive = probabilities
        else:
            prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, prob_positive, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title("Calibration Plot")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def compute_uncertainty_metrics(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    method: str = "entropy"
) -> Dict[str, float]:
    """Compute uncertainty metrics.
    
    Args:
        predictions: Model predictions
        probabilities: Predicted probabilities
        labels: Ground truth labels
        method: Uncertainty method ("entropy", "max_prob", "variance")
        
    Returns:
        Dictionary of uncertainty metrics
    """
    metrics = {}
    
    # Convert to numpy
    probs = probabilities.cpu().numpy()
    preds = predictions.cpu().numpy()
    true_labels = labels.cpu().numpy()
    
    if method == "entropy":
        # Entropy-based uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        metrics['mean_entropy'] = np.mean(entropy)
        metrics['std_entropy'] = np.std(entropy)
        
        # Uncertainty vs accuracy
        correct = (preds == true_labels)
        metrics['entropy_correct'] = np.mean(entropy[correct])
        metrics['entropy_incorrect'] = np.mean(entropy[~correct])
    
    elif method == "max_prob":
        # Maximum probability uncertainty
        max_probs = np.max(probs, axis=1)
        metrics['mean_max_prob'] = np.mean(max_probs)
        metrics['std_max_prob'] = np.std(max_probs)
        
        # Confidence vs accuracy
        correct = (preds == true_labels)
        metrics['max_prob_correct'] = np.mean(max_probs[correct])
        metrics['max_prob_incorrect'] = np.mean(max_probs[~correct])
    
    return metrics


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> str:
    """Create a formatted leaderboard from results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Formatted leaderboard string
    """
    if not results:
        return "No results available"
    
    # Get all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    # Create header
    header = "Model".ljust(20)
    for metric in sorted(all_metrics):
        header += metric.rjust(12)
    
    lines = [header, "=" * len(header)]
    
    # Add model results
    for model_name, model_results in results.items():
        line = model_name.ljust(20)
        for metric in sorted(all_metrics):
            value = model_results.get(metric, 0.0)
            line += f"{value:.4f}".rjust(12)
        lines.append(line)
    
    return "\n".join(lines)
