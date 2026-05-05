def accuracy(predictions: list[int], labels: list[int]) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    return (
        sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)
        if labels
        else 0.0
    )


def f1_score(predictions: list[int], labels: list[int], class_id: int) -> float:
    assert len(predictions) == len(
        labels
    ), "Length of predictions and labels must be the same."
    true_positives = float(
        sum(
            (pred == class_id) and (label == class_id)
            for pred, label in zip(predictions, labels)
        )
    )
    false_positives = float(
        sum(
            (pred == class_id) and (label != class_id)
            for pred, label in zip(predictions, labels)
        )
    )
    false_negatives = float(
        sum(
            (pred != class_id) and (label == class_id)
            for pred, label in zip(predictions, labels)
        )
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1
