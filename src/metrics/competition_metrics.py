# auc scoreを計算する関数
import torch
from torchmetrics.classification.accuracy import Accuracy


class CompetitionMetrics:
    def __init__(self):
        self.metrics = Accuracy(task="multiclass", num_classes=10)

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return self.metrics(y_pred, y_true)


if __name__ == "__main__":
    metrics = CompetitionMetrics()
    y_true = torch.randint(0, 10, (100,))
    y_pred = torch.randint(0, 10, (100,))

    print(metrics(y_true, y_pred))
