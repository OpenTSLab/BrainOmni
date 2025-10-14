import os
import torch
import random
import argparse
import numpy as np
from constant import PROJECT_ROOT_PATH
from downstream.tester import DownStreamTester
from downstream.config import DownstreamConfig
from constant import SEED


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
    parser.add_argument("--launcher", type=str,default='pdsh')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ckpt_path", type=str, default="not_need_ckpt")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--n_fold", type=int)
    parser.add_argument("--fold_index", type=int)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--backbone_lr", type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    seed = args.seed
    seed_everything(seed)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    cfg = DownstreamConfig(
        dataset_name=args.dataset_name,
        ckpt_path=args.ckpt_path,
        pretrained=args.pretrained,
        frozen=args.frozen,
        epoch=args.epoch,
        n_fold=args.n_fold,
        fold_index=args.fold_index,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        world_size=world_size,
    )

    exp_path = os.path.join(
        PROJECT_ROOT_PATH,
        "exp_results_downstream",
        cfg.model_name,
        cfg.ckpt_path.split("/")[-1] + ("_frozen" if args.frozen else ""),
        cfg.dataset_name,
        f"seed{seed}_backbone{cfg.backbone_lr}_head{cfg.head_lr}",
        f"total_{cfg.n_fold}_fold",
        f"{cfg.fold_index}_fold",
    )
    os.makedirs(exp_path, exist_ok=True)
    if not os.path.exists(os.path.join(exp_path, "metrics.json")):
        DownStreamTester(cfg, rank, local_rank, world_size, exp_path).main()
