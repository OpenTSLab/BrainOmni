import torch
from sklearn import metrics as sk_metrics

class MetricsComputer:
    def __init__(self, num_classes, is_binary):
        self.num_classes = num_classes
        self.is_binary = is_binary
        self.count = 0
        self.preds = []
        self.gts = []
        if self.is_binary:
            self.metrics = {
                "pr_auc": 0.0,  # precision-recall curve
                "roc_auc": 0.0,  # operating characteristic curve
                "f1": 0.0,  # f1 for pos class
                "f1_macro": 0.0,  # f1_macro
                "accuracy": 0.0,  # acc
                "balanced_accuracy": 0.0,  # bacc
                "cohen_kappa": 0.0,  # kappa
            }
        else:
            self.metrics = {
                "cohen_kappa": 0.0,  # kappa
                "f1_weighted": 0.0,  # f1_weighted
                "f1_macro": 0.0,  # f1_macro
                "balanced_accuracy": 0.0,  # bacc
                "accuracy": 0.0,  # acc
            }

    def compute_metrics(self, prob: torch.Tensor, gts: torch.Tensor):
        y_true = gts.cpu().float().numpy()
        y_prob = prob.cpu().float().numpy()
        y_pred = torch.argmax(prob, dim=-1).cpu().float().numpy()
        metrics = {}

        for key in self.metrics:
            if key == "roc_auc":
                metrics["roc_auc"] = sk_metrics.roc_auc_score(
                    y_true=y_true, y_score=y_prob[:, 1]
                )
            elif key == "pr_auc":
                metrics["pr_auc"] = sk_metrics.average_precision_score(
                    y_true=y_true, y_score=y_prob[:, 1]
                )
            elif key == "f1":
                metrics["f1"] = sk_metrics.f1_score(
                    y_true=y_true, y_pred=y_pred, average="binary"
                )
            elif key == "f1_macro":
                metrics["f1_macro"] = sk_metrics.f1_score(
                    y_true=y_true, y_pred=y_pred, average="macro"
                )
            elif key == "accuracy":
                metrics["accuracy"] = sk_metrics.accuracy_score(
                    y_true=y_true, y_pred=y_pred
                )
            elif key == "balanced_accuracy":
                metrics["balanced_accuracy"] = sk_metrics.balanced_accuracy_score(
                    y_true=y_true, y_pred=y_pred
                )
            elif key == "cohen_kappa":
                metrics["cohen_kappa"] = sk_metrics.cohen_kappa_score(y_true, y_pred)
            elif key == "f1_weighted":
                metrics["f1_weighted"] = sk_metrics.f1_score(
                    y_true=y_true, y_pred=y_pred, average="weighted"
                )

        return metrics

    def step(self, pred: torch.Tensor, gts: torch.Tensor):
        self.preds.append(pred.cpu().float())
        self.gts.append(gts.cpu().float())

    def get_metrics(self):
        self.preds = torch.concat(self.preds)
        self.gts = torch.concat(self.gts)
        metrics = self.compute_metrics(self.preds, self.gts)
        return metrics

    def reset(self):
        self.preds = []
        self.gts = []
