"""
Validation utilities for face recognition.
"""

import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

def calculate_optimal_threshold(y_true, y_scores, method='roc'):
    """
    Calculate optimal threshold based on ROC or PR curve.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Similarity scores
        method: 'roc' or 'pr' (precision-recall)
    
    Returns:
        Optimal threshold
    """
    if method == 'roc':
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find optimal threshold (maximizing TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold from ROC: {optimal_threshold:.4f}")
        return optimal_threshold
    
    elif method == 'pr':
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find optimal threshold (maximizing F1 score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        
        # PR curve returns one less threshold than precision/recall points
        if optimal_idx >= len(thresholds):
            optimal_threshold = thresholds[-1]
        else:
            optimal_threshold = thresholds[optimal_idx]
            
        logger.info(f"Optimal threshold from PR curve: {optimal_threshold:.4f}")
        return optimal_threshold
    
    else:
        logger.warning(f"Unknown method: {method}")
        return 0.5

def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Similarity scores
        save_path: Path to save plot
    
    Returns:
        AUC score
    """
    try:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Mark optimal threshold
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
                 label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved ROC curve to {save_path}")
        
        plt.close()
        
        return roc_auc
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")
        return None

def evaluate_model(y_true, y_scores, threshold=0.5):
    """
    Evaluate face recognition model.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Similarity scores
        threshold: Classification threshold
    
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Convert scores to predictions
        y_pred = (np.array(y_scores) >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Calculate AUC
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr_curve, tpr_curve)
        
        # Create results dictionary
        results = {
            'threshold': threshold,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'sensitivity': tpr,
            'specificity': tnr,
            'precision': precision,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'auc': roc_auc
        }
        
        logger.info(f"Evaluation results at threshold {threshold}:")
        for k, v in results.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        
        return results
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def create_threshold_vs_metrics_plot(y_true, y_scores, save_path=None, num_thresholds=100):
    """
    Create plot of metrics vs threshold.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Similarity scores
        save_path: Path to save plot
        num_thresholds: Number of thresholds to evaluate
    """
    try:
        # Create threshold values
        thresholds = np.linspace(0, 1, num_thresholds)
        
        # Initialize metric arrays
        precision_values = []
        recall_values = []
        f1_values = []
        accuracy_values = []
        
        # Calculate metrics for each threshold
        for threshold in thresholds:
            results = evaluate_model(y_true, y_scores, threshold)
            precision_values.append(results['precision'])
            recall_values.append(results['sensitivity'])
            f1_values.append(results['f1_score'])
            accuracy_values.append(results['accuracy'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precision_values, label='Precision', color='blue')
        plt.plot(thresholds, recall_values, label='Recall', color='green')
        plt.plot(thresholds, f1_values, label='F1 Score', color='red')
        plt.plot(thresholds, accuracy_values, label='Accuracy', color='purple')
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved threshold vs metrics plot to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error creating threshold vs metrics plot: {e}")