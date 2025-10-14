import os
import json
import torch
import random
import argparse
import numpy as np
from constant import SEED, PROJECT_ROOT_PATH
from datetime import datetime
from brainomni.trainer import Trainer
from brainomni.config import BrainOmniTrainerConfig


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arg():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--launcher", type=str)
    parser.add_argument("--signal_type", type=str)
    parser.add_argument("--model_size", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--num_quantizers_used", type=int)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    return args


def record_datetime_exp(rank, cfg_exp_name, model_cfg_json):
    exp_name = "exp_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    exp_path = os.path.join(
        PROJECT_ROOT_PATH, "train_omni_results", cfg_exp_name, exp_name[:-3]
    )
    if rank == 0:
        os.system(f"mkdir -p {exp_path}")
        os.makedirs(os.path.join(exp_path, cfg_exp_name), exist_ok=True)
        json.dump(
            model_cfg_json,
            open(os.path.join(exp_path, cfg_exp_name, "model_cfg.json"), "w"),
            indent=4,
        )
    return exp_path


if __name__ == "__main__":
    args = parse_arg()

    seed_everything(SEED)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    cfg = BrainOmniTrainerConfig(
        signal_type=args.signal_type,
        epoch=args.epoch,
        model_size=args.model_size,
        tokenizer_path=args.tokenizer_path,
        num_quantizers_used=args.num_quantizers_used,
        world_size=world_size,
    )

    exp_path = record_datetime_exp(
        rank,
        cfg.exp_name,
        model_cfg_json=cfg.get_model_cfg(),
    )

    Trainer(
        cfg,
        local_rank,
        rank,
        world_size,
        exp_path=exp_path,
    ).main()
