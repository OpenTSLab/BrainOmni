import os
import json
import torch
from typing import Optional
from brainomni.model import BrainOmni


def get_brainomni(
    pretrained: bool = True,
    ckpt_path: Optional[str] = None,
    *awargs,
    **kwargs
):
    assert ckpt_path is not None
    model_config_path = os.path.join(ckpt_path, "model_cfg.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = BrainOmni(**model_config)
    if pretrained:
        checkpoint = torch.load(os.path.join(ckpt_path, "BrainOmni.pt"), map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        for p in model.tokenizer.parameters():
            p.requires_grad=False
    return model, model.lm_dim
