import numpy as np

FIXED_QUOTA = 1000


def average_precision(targets, scores):
    """Integrate precision against recall, grouping tied scores at one threshold."""
    positives = targets.sum()
    if not positives:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    true_positives = np.cumsum(targets[order])[ends]
    precision = true_positives / (ends + 1)
    return float(np.diff(true_positives, prepend=0) @ precision / positives)


def quota_metrics(targets, scores, quota, returns=None):
    """Winners among the `quota` highest scores, against random picks at the split's winner rate.

    With `returns` (net percent per entry, aligned with targets), also report the mean net
    return of the selected entries and of taking every entry in the split.
    """
    calls = min(quota, len(targets))
    cutoff = np.partition(scores, len(scores) - calls)[-calls]
    above, tied = scores > cutoff, scores == cutoff
    # Average over ties at the cutoff so row order cannot change the result.
    def top_sum(values):
        return float(values[above].sum() + (calls - above.sum()) * values[tied].mean())

    result = {"calls": calls, "positives": top_sum(targets), "baseline_positives": calls * float(targets.mean())}
    if returns is not None and np.isfinite(returns).all():
        result.update(net_return_pct_per_trade=top_sum(returns) / calls,
                      all_net_return_pct_per_trade=float(returns.mean()))
    return result


class Metrics:
    def __init__(self, num_classes=2):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.loss_sum = 0.0
        self.count = 0

    def update(self, targets, predictions, loss):
        targets, predictions = np.asarray(targets), np.asarray(predictions)
        n = len(self.matrix)
        self.matrix += np.bincount(n * targets + predictions, minlength=n * n).reshape(n, n)
        self.count += len(targets)
        self.loss_sum += float(loss) * len(targets)

    def result(self):
        if not self.count:
            raise ValueError("Cannot calculate metrics on an empty dataset")
        if len(self.matrix) != 2:
            support = self.matrix.sum(axis=1)
            calls = self.matrix.sum(axis=0)
            hits = self.matrix.diagonal()
            seen = support > 0
            return {
                "loss": self.loss_sum / self.count, "accuracy": int(hits.sum()) / self.count,
                "majority_accuracy": int(support.max()) / self.count,
                "balanced_accuracy": float(np.mean(hits[seen] / support[seen])),
                "confusion_matrix": self.matrix.tolist(), "count": self.count,
                "class_counts": {str(i): int(n) for i, n in enumerate(support)},
                "per_class": [
                    {"support": int(total), "calls": int(called),
                     "recall": int(hit) / int(total) if total else None,
                     "precision": int(hit) / int(called) if called else None}
                    for total, called, hit in zip(support, calls, hits)
                ],
            }
        tn, fp, fn, tp = map(int, self.matrix.ravel())
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return {
            "loss": self.loss_sum / self.count,
            "accuracy": (tp + tn) / self.count,
            "always_skip_accuracy": (tn + fp) / self.count,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "precision_enter": tp / (tp + fp) if tp + fp else 0.0,
            "recall_enter": recall,
            "balanced_accuracy": (recall + specificity) / 2 if tp + fn and tn + fp else None,
            "profitable_above_baseline": (tp * self.count - (tp + fp) * (tp + fn)) / self.count,
            "allowed": tp + fp, "rejected": tn + fn, "count": self.count,
            "class_counts": {"0": tn + fp, "1": tp + fn},
        }
