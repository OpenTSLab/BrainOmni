import torch


class MetricsComputer:
    """
    assign this to a specific setting and it can be used to compute all the metrics
    """

    def __init__(self):
        self.evaluate_func = {
            "mae": compute_mae,
            "mse": compute_mse,
            "amp": compute_amp,
            "phase": compute_phase,
            "pcc": compute_pcc,
        }
        self.record = []

    def step(
        self,
        rec: torch.Tensor,
        raw: torch.Tensor,
        sensor_type: torch.Tensor,
    ):
        rec = rec.detach().float()
        raw = raw.detach().float()

        cur_metrics = {
            key: self.evaluate_func[key](rec, raw).item()
            for key in self.evaluate_func.keys()
        }
        cur_metrics["is_eeg"] = (sensor_type == 0).all()
        self.record.append(cur_metrics)

    def get_metrics(self):
        metrics = {"all": {}, "eeg": {}, "meg": {}}
        for key in self.evaluate_func.keys():
            metrics["all"][key] = (
                torch.tensor([i[key] for i in self.record]).mean().item()
            )
            metrics["eeg"][key] = (
                torch.tensor([i[key] for i in self.record if i["is_eeg"]]).mean().item()
            )
            metrics["meg"][key] = (
                torch.tensor([i[key] for i in self.record if not i["is_eeg"]])
                .mean()
                .item()
            )
        return metrics

    def reset(self):
        self.record = []


def compute_mae(rec, raw):
    mae = torch.abs(rec - raw)
    return torch.mean(mae)


def compute_mse(rec, raw):
    mse = torch.square(rec - raw)
    return torch.mean(mse)


def compute_amp(rec, raw):
    window = torch.hamming_window(rec.shape[-1], device=rec.device)

    pred_fft = torch.fft.rfft(rec * window, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(raw * window, dim=-1, norm="ortho")

    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    return compute_mae(pred_magnitude, target_magnitude)


def compute_phase(rec, raw):
    window = torch.hamming_window(rec.shape[-1], device=rec.device)

    pred_fft = torch.fft.rfft(rec * window, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(raw * window, dim=-1, norm="ortho")

    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    return compute_mae(pred_phase, target_phase)


def compute_pcc(rec: torch.Tensor, raw: torch.Tensor):
    # B C W
    B, C, W, D = rec.shape
    x = rec.reshape(B * C * W, 1, D)
    y = raw.reshape(B * C * W, 1, D)
    c = (
        (x - x.mean(dim=-1, keepdim=True))
        @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
        * (1.0 / (D - 1))
    ).squeeze()
    sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
    return (c / sigma).mean()
