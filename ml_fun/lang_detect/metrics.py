import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    return float(np.mean(predictions == labels, dtype=np.float32))


def precision(predictions: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    true_positives = float(np.sum((predictions == class_id) & (labels == class_id)))
    false_positives = float(np.sum((predictions == class_id) & (labels != class_id)))
    return (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )


def recall(predictions: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    true_positives = float(np.sum((predictions == class_id) & (labels == class_id)))
    false_negatives = float(np.sum((predictions != class_id) & (labels == class_id)))
    return (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )


def f1_score(predictions: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    precision_ = precision(predictions, labels, class_id)
    recall_ = recall(predictions, labels, class_id)
    f1 = (
        2.0 * (precision_ * recall_) / (precision_ + recall_)
        if (precision_ + recall_) > 0
        else 0.0
    )
    return f1
