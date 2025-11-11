from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict, List

def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Computes metrics for a single-label classification task.
    """
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_multi_label_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes metrics for a multi-label classification task.
    """
    preds = (predictions > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_ner_metrics(predictions: np.ndarray, labels: np.ndarray, label_map: Dict[int, str]) -> Dict[str, float]:
    """
    Computes metrics for a NER task.
    """
    preds = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_map[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [label_map[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]

    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(flat_labels, flat_labels, average='weighted', zero_division=0)
    acc = accuracy_score(flat_labels, flat_predictions)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
