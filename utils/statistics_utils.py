from dataclasses import dataclass

import numpy as np
from sklearn import metrics


@dataclass
class ProbingStatistics:
    accuracy: float = None
    recall_at_5: float = None
    macro_accuracy: float = None


def get_statistics_results(gt_array: np.ndarray, predictions: np.ndarray, predictions_scores: np.ndarray) -> ProbingStatistics:
    accuracy = metrics.accuracy_score(gt_array, predictions)
    macro_accuracy = metrics.balanced_accuracy_score(gt_array, predictions)
    if predictions_scores.shape[1] > 5:
        recall_at_5 = metrics.top_k_accuracy_score(gt_array, predictions_scores, k=5, labels=np.arange(predictions_scores.shape[1]))
    else:
        recall_at_5 = 0.

    return ProbingStatistics(accuracy=accuracy, recall_at_5=recall_at_5, macro_accuracy=macro_accuracy)
