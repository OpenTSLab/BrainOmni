import torch
import torch.nn as nn

class DownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        frozen: bool,
        n_dim: int,
        num_classes: int,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        assert num_classes >= 2
        self.backbone = backbone
        self.frozen = frozen
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.class_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(n_dim),
            nn.SELU(),
            nn.Linear(n_dim, num_classes),
        )

    def forward(self, input_dict, y):
        if self.frozen:
            self.backbone.eval()
        x = self.backbone.encode(**input_dict)
        if x.ndim == 4:  # B C W D
            x = x.mean(2)
        x = x.contiguous().view(x.shape[0], -1)
        logits = self.class_head(x)
        loss = self.criterion(logits, y.long())
        return logits, loss

    @torch.no_grad()
    def predict(self, input_dict):
        x = self.backbone.encode(**input_dict)
        if x.ndim == 4:
            x = x.mean(2)
        x = x.contiguous().view(x.shape[0], -1)
        logits = self.class_head(x)
        return logits, x

    @torch.jit.ignore
    def get_parameters_groups(
        self, weight_decay: float, backbone_lr: float, head_lr: float
    ):
        # Get backbone parameter groups
        backbone_groups = self.backbone.get_parameters_groups(
            weight_decay=weight_decay, lr=backbone_lr
        )
        # Get head parameter groups (treated as final layer)
        head_params = []
        for p in self.class_head.parameters():
            head_params.append(p)
        return backbone_groups + [
            {"weight:decay": weight_decay, "params": head_params, "lr": head_lr}
        ]
