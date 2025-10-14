import os
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from constant import SEED, PROJECT_ROOT_PATH
from braintokenizer.trainer import Trainer
from braintokenizer.config import BrainTokenizerTrainerConfig

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
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--n_neuro", type=int, default=16)
    parser.add_argument("--codebook_size", type=int,default=512)
    parser.add_argument("--codebook_dim", type=int,default=256)
    parser.add_argument("--num_quantizers", type=int,default=4)
    args = parser.parse_args()
    return args


def record_datetime_exp(rank, cfg_exp_name, model_cfg_json):
    exp_name = "exp_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    exp_path = os.path.join(
        PROJECT_ROOT_PATH, "train_tokenizer_results", cfg_exp_name, exp_name[:-3]
    )
    if rank == 0:
        os.system(f"mkdir -p {exp_path}")
        os.makedirs(os.path.join(exp_path, cfg_exp_name), exist_ok=True)
        json.dump(
            model_cfg_json,
            open(os.path.join(exp_path, cfg_exp_name, 'model_cfg.json'), "w"),
            indent=4,
        )
    return exp_path


if __name__ == "__main__":
    args = parse_arg()
    
    seed_everything(SEED)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    cfg = BrainTokenizerTrainerConfig(
        signal_type=args.signal_type,
        epoch=args.epoch,
        n_neuro=args.n_neuro,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        num_quantizers=args.num_quantizers,
        world_size=world_size,
    )
    exp_path = record_datetime_exp(
        rank,
        cfg.exp_name,
        model_cfg_json=cfg.get_model_cfg(),
    )

    Trainer(cfg, local_rank, rank, world_size, exp_path=exp_path).main()
