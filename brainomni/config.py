import os
import json
from constant import (
    SAMPLE_RATE,
    LOW,
    HIGH,
    PRETRAIN_METADATA_PATH,
)


class BrainOmniTrainerConfig:
    def __init__(
        self,
        signal_type: str,
        model_size: str,
        tokenizer_path: str,
        num_quantizers_used: int,
        epoch: str,
        world_size: int,
    ):
        self.signal_type = signal_type
        self.exp_name = f"BrainOmni_epoch{epoch}_model_{model_size}_signal_{signal_type}_num_quantizers_{num_quantizers_used}_tokenizer_{tokenizer_path.split('/')[-1]}"

        # dataset
        TIME = 30
        STRIDE = 30
        self.pretrain_metadata_path = os.path.join(
            PRETRAIN_METADATA_PATH,
            f"sfreq_{SAMPLE_RATE}_low_{LOW}_high_{HIGH}_time_{TIME}_stride_{STRIDE}",
        )

        # tokenizer parameter
        self.tokenizer_ckpt_path = os.path.join(tokenizer_path, "BrainTokenizer.pt")
        tokenizer_parameter = json.load(
            open(os.path.join(tokenizer_path, "model_cfg.json"), "r")
        )
        self.window_length = tokenizer_parameter["window_length"]
        self.n_filters = tokenizer_parameter["n_filters"]
        self.ratios = tokenizer_parameter["ratios"]
        self.kernel_size = tokenizer_parameter["kernel_size"]
        self.last_kernel_size = tokenizer_parameter["last_kernel_size"]
        self.n_dim = tokenizer_parameter["n_dim"]
        self.n_neuro = tokenizer_parameter["n_neuro"]
        self.n_head = tokenizer_parameter["n_head"]
        self.dropout = tokenizer_parameter["dropout"]
        self.codebook_dim = tokenizer_parameter["codebook_dim"]
        self.codebook_size = tokenizer_parameter["codebook_size"]
        self.num_quantizers = tokenizer_parameter["num_quantizers"]
        self.rotation_trick = tokenizer_parameter["rotation_trick"]
        self.quantize_optimize_method = tokenizer_parameter["quantize_optimize_method"]

        # LM parameter
        self.overlap_ratio = 0.25
        self.num_quantizers_used = num_quantizers_used
        self.lm_dropout = 0.1
        self.mask_ratio = 0.5
        # tokenizer 5M
        if model_size == "tiny":
            self.lm_dim = 256
            self.lm_head = 8
            self.lm_depth = 12
            self.lr = 5e-4
        elif model_size == "base":
            self.lm_dim = 512
            self.lm_head = 16
            self.lm_depth = 12
            self.lr = 4e-4

        self.batch_size = 8
        # 训练参数
        self.weight_decay = 0.05
        self.total_batch_per_update = 256

        assert self.total_batch_per_update // (self.batch_size * world_size) > 0
        self.gradient_accumulation_steps = self.total_batch_per_update // (
            self.batch_size * world_size
        )

        self.num_workers = 32
        self.epoch = epoch
        self.train_data_ratio = 1.0
        self.valid_data_ratio = 1.0
        self.test_data_ratio = 1.0
        self.scheduler_warm_ratio = 0.1

        self.ds_config = {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "bf16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0.0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {"betas": [0.9, 0.95], "eps": 1e-6},
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": None,  # set when trainning
                    "warmup_num_steps": None,  # set when trainning
                    "warmup_min_ratio": 0.1,
                    "cos_min_ratio": 0.1,
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "overlap_comm": False,
                "allgather_partitions": True,
                "allgather_bucket_size": "auto",
                "reduce_scatter": True,
                "reduce_bucket_size": "auto",
            },
        }

    def get_model_cfg(self):
        return {
            # Tokenizer parameters
            "window_length": self.window_length,
            "n_filters": self.n_filters,
            "ratios": self.ratios,
            "kernel_size": self.kernel_size,
            "last_kernel_size": self.last_kernel_size,
            "n_dim": self.n_dim,
            "n_head": self.n_head,
            "n_neuro": self.n_neuro,
            "dropout": self.dropout,
            "codebook_dim": self.codebook_dim,
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            "rotation_trick": self.rotation_trick,
            "quantize_optimize_method": self.quantize_optimize_method,
            # LM parameters
            "overlap_ratio": self.overlap_ratio,
            "lm_dim": self.lm_dim,
            "lm_head": self.lm_head,
            "lm_depth": self.lm_depth,
            "lm_dropout": self.lm_dropout,
            "mask_ratio": self.mask_ratio,
            "num_quantizers_used": self.num_quantizers_used,
        }
