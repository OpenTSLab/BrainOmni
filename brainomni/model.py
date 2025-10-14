import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from braintokenizer.model import BrainTokenizer
from model_utils.attn import SelfAttnBlock, RMSNorm, SpatialTemporalAttentionBlock


class BrainOmni(nn.Module):
    def __init__(
        self,
        # tokenizer parameter
        window_length: int,
        n_filters: int,
        ratios: List[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        n_neuro: int,
        dropout: float,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        # lm model parameter
        overlap_ratio: float,
        lm_dim: int,
        lm_head: int,
        lm_depth: int,
        lm_dropout: float,
        mask_ratio: float,
        num_quantizers_used: int,
        **kwargs,
    ):
        super().__init__()
        self.lm_dim = lm_dim
        self.window_length = window_length
        self.overlap_ratio = overlap_ratio
        self.mask_ratio = mask_ratio
        self.num_quantizers_used = (
            num_quantizers_used if num_quantizers_used != None else num_quantizers
        )
        # B C T -> unfold -> B C T' -> tokenizer -> (B C) W D -> next predict
        self.tokenizer = BrainTokenizer(
            window_length,
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            n_dim,
            n_neuro,
            n_head,
            dropout,
            codebook_dim,
            codebook_size,
            num_quantizers,
            rotation_trick,
            quantize_optimize_method,
        )
        self.mask_token = nn.Parameter(torch.randn(n_dim))
        self.projection = nn.Linear(n_dim, lm_dim) if n_dim != lm_dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(lm_dim, lm_head, lm_dropout, causal=False)
                for _ in range(lm_depth)
            ]
        )
        self.predict_head = nn.Linear(lm_dim, num_quantizers_used * codebook_size)
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
    def load_frozen_tokenizer_ckpt(self, tokenizer_ckpt_path: str):
        self.tokenizer.load_state_dict(
            torch.load(tokenizer_ckpt_path, weights_only=True)
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        return None

    @torch.jit.ignore
    def get_parameters_groups(self, lr: float, weight_decay: float):
        no_decay_params = []
        normal_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if (
                    "norm" in n
                    or "predict_head" in n
                    or n in ["projection.weight", "projection.bias", "mask_token"]
                ):
                    no_decay_params.append(p)
                else:
                    normal_params.append(p)

        return [
            {"params": normal_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        **kwargs,
    ):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C
        """
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, D = x.shape

        mask = (
            torch.rand(size=(B, C, W), device=x.device) > self.mask_ratio
        )  # true in mask will be preserve, false in mask will be masked
        # 20% random select a token from the minibatch
        x = torch.where(
            mask.unsqueeze(-1).repeat(1, 1, 1, D),
            x,
            rearrange(
                x.view(-1, D)[torch.randperm(B * C * W, device=x.device)],
                "(B C W) D -> B C W D",
                B=B,
                C=C,
            ),
        )
        # 80% use mask token
        tmp_mask = (mask.float() + torch.rand(size=(B, C, W), device=x.device)) > 0.8
        tmp_mask = tmp_mask.unsqueeze(-1).type_as(x)
        mask_token = self.mask_token.type_as(x)
        x = x * tmp_mask + mask_token * (1 - tmp_mask)

        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro

        x = self.projection(x)

        for block in self.blocks:
            x = block(x)

        # (batch channel) window (num_quant logit_dim)  -> batch channel window num_quant logit_dim
        logits = rearrange(
            self.predict_head(x),
            "B C W (N D) -> B C W N D",
            N=self.num_quantizers_used,
        )
        loss, acc = self.compute_cross_entropy(logits, label_indices, mask)
        output_dict = {"loss": loss, "acc_all": acc.mean()}
        for i in range(self.num_quantizers_used):
            output_dict[f"acc_{i}"] = acc[i]
        return output_dict

    def encode(self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C

        output: B W D
        """
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, _ = x.shape
        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro
        x = self.projection(x)

        for block in self.blocks[:-1]:
            x = block(x)

        return F.normalize(
            x,
            p=2.0,
            dim=-1,
            eps=1e-6,
        )

    def compute_cross_entropy(
        self, logits: torch.Tensor, label: torch.Tensor, mask: torch.Tensor
    ):
        """
        logits: B C W num_quantizers_used codebook_size
        label:  B C W num_quantizers_used
        mask:   B C W
        """
        B, C, W, N = label.shape
        logits = logits[~mask]
        label = label[~mask]
        #  X is masked num , N is codebook depth, M is codebook size
        logits = rearrange(logits, "X N M -> (X N) M")
        label = label.view(-1)

        loss = F.cross_entropy(logits.float(), label, reduction="mean")

        acc = (
            rearrange((logits.argmax(dim=-1)) == label, "(X N) -> N X", N=N)
            .float()
            .mean(dim=-1)
        )

        return loss, acc
