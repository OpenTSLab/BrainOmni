import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model_utils.attn import RMSNorm
from model_utils.loss import get_time_loss, get_pcc, get_frequency_domain_loss
from model_utils.module import (
    BrainSensorModule,
    BrainTokenizerEncoder,
    BrainQuantizer,
    BrainTokenizerDecoder,
)


class BrainTokenizer(nn.Module):
    def __init__(
        self,
        window_length,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        n_dim,
        n_neuro,
        n_head,
        dropout,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        **kwargs,
    ):
        super().__init__()
        self.window_length = window_length
        self.n_dim = n_dim
        self.sensor_embed = BrainSensorModule(n_dim)
        self.mask_ratio = 0.25  # hard coded

        self.encoder = BrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
        )
        self.quantizer = BrainQuantizer(
            n_dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        self.decoder = BrainTokenizerDecoder(
            n_dim=n_dim,
            n_head=n_head,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=dropout,
        )
        # --------------------------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    @torch.jit.ignore
    def get_parameters_groups(self, lr: float, codebook_lr: float, weight_decay: float):
        normal_params = []
        no_decay_params = []
        codebook_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if "norm" in n or n in [
                    "sensor_embed.sensor_embedding_layer.weight",
                ]:
                    no_decay_params.append(p)
                elif "quantizer" in n:
                    codebook_params.append(p)
                else:
                    normal_params.append(p)
        return [
            {"params": normal_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
            {"params": codebook_params, "lr": codebook_lr, "weight_decay": 0.0},
        ]

    def unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))
        if overlap_ratio > 0.0:
            stride = int(self.window_length * (1 - overlap_ratio))
            right_remain = (x.shape[-1] - self.window_length) % stride
            if right_remain > 0:
                x = F.pad(x, pad=(0, stride - right_remain))
        return x.unfold(
            dimension=-1,
            size=self.window_length,
            step=int(self.window_length * (1 - overlap_ratio)),
        )

    def norm_target(self, x: torch.Tensor):
        """
        x: B C N L
        """
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1,keepdim=True)+1e-6)
        return x

    def add_noise(self, x: torch.Tensor):
        return x + torch.randn_like(x) * 0.1

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        """
        x: B C (N L)
        pos: B C 6
        sensor_type: B C
        """
        x = self.unfold(x)

        sensor_embedding = self.sensor_embed(pos, sensor_type)
        random_index = torch.randperm(x.shape[1], device=x.device)
        x = x.index_select(dim=1, index=random_index)
        sensor_embedding = sensor_embedding.index_select(dim=1, index=random_index)
        n_mask_channel = max(int(x.shape[1] * self.mask_ratio), 1)
        feature = self.encoder(
            self.add_noise(x[:, n_mask_channel:]),
            sensor_embedding[:, n_mask_channel:],
        )

        feature, indices, commitment_loss = self.quantizer(feature)

        x_rec = self.decoder(feature, sensor_embedding)

        x_rec = x_rec.float()
        x = self.norm_target(x)

        time_loss = get_time_loss(x_rec, x)
        pcc = get_pcc(x_rec, x)
        amp_loss, phase_loss = get_frequency_domain_loss(x_rec, x)
        return {
            "loss": time_loss
            + torch.exp(-pcc)
            + commitment_loss
            + amp_loss
            + 0.5 * phase_loss,
            "time_loss": time_loss.detach(),
            "pcc": pcc.detach(),
            "amp_loss": amp_loss.detach(),
            "phase_loss": phase_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "judge_loss": (
                time_loss
                + torch.exp(-pcc)
                + commitment_loss
                + amp_loss
                + 0.5 * phase_loss
            ).detach(),
        }, indices

    @torch.no_grad()
    def visualize(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C
        """
        x = self.unfold(x)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        x_rec = self.decoder(feature, sensor_embedding)
        return {
            "x": self.norm_target(x),
            "x_rec": x_rec.float(),
            "sensor_type": sensor_type,
        }

    @torch.no_grad()
    def tokenize(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        overlap_ratio: float,
        **kwargs,
    ):
        """
        x: B C T
        pos: B C 6
        sensor_type: B C
        """
        self.eval()
        x = self.unfold(x, overlap_ratio=overlap_ratio)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        feature = rearrange(feature, "B C N T D->B C (N T) D")
        indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
        return feature, indices

    def get_finetune_parameter_groups(self, weight_decay, layer_decay):
        del self.decoder
        del self.quantizer
        parameter_groups = {}

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            this_weight_decay = weight_decay
            group_name = "decay"

            # Create group if it doesn't exist
            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": layer_decay,
                }

            parameter_groups[group_name]["params"].append(p)

        return list(parameter_groups.values())
