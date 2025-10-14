import os
from constant import (
    TOKENIZER_SEGMENT_TIME,
    SAMPLE_RATE,
    LOW,
    HIGH,
    PRETRAIN_METADATA_PATH,
)

class BrainTokenizerTrainerConfig:
    def __init__(
        self,
        signal_type,
        epoch,
        n_neuro,
        codebook_size,
        codebook_dim,
        num_quantizers,
        world_size,
    ):
        self.signal_type = signal_type
        assert signal_type in ["eeg", "meg", "both"]
        self.exp_name = f"braintokenizer_{signal_type}_n_neuro_{n_neuro}"
        # dataset
        TIME = 10
        STRIDE = 10
        self.pretrain_metadata_path = os.path.join(
            PRETRAIN_METADATA_PATH,
            f"sfreq_{SAMPLE_RATE}_low_{LOW}_high_{HIGH}_time_{TIME}_stride_{STRIDE}",
        )

        # model parameter
        self.window_length = int(TOKENIZER_SEGMENT_TIME * SAMPLE_RATE)
        self.n_filters = 32
        self.ratios = [8, 4, 2]
        self.kernel_size = 5
        self.last_kernel_size = 5
        self.n_dim = 256

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.rotation_trick = True
        self.quantize_optimize_method = "ema"

        self.n_neuro = n_neuro
        self.n_head = 4
        self.dropout = 0.0

        # 训练参数
        self.total_batch_per_update = 512

        self.batch_size = 16
        if self.signal_type == "eeg":
            self.batch_size = 32

        assert self.total_batch_per_update // (self.batch_size * world_size) > 0

        self.gradient_accumulation_steps = self.total_batch_per_update // (
            self.batch_size * world_size
        )

        self.num_workers = 32
        self.epoch = epoch
        self.train_data_ratio = 1.0
        self.val_data_ratio = 1.0
        self.scheduler_warm_ratio = 0.1
        self.weight_decay = 1e-2
        self.lr = 2e-4
        self.codebook_lr = 3e-4
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
                "params": {
                    "betas": [0.5, 0.9],
                    "eps": 1e-5,
                },
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": None,  # set when trainning
                    "warmup_num_steps": None,  # set when trainning
                    "warmup_min_ratio": 0.1,
                    "cos_min_ratio": 0.05,
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
            "window_length": self.window_length,
            "n_filters": self.n_filters,
            "ratios": self.ratios,
            "kernel_size": self.kernel_size,
            "last_kernel_size": self.last_kernel_size,
            "n_dim": self.n_dim,
            "n_neuro": self.n_neuro,
            "n_head": self.n_head,
            "dropout": self.dropout,
            "codebook_dim": self.codebook_dim,  # 注意这里保留了原来的self.n_dim引用
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            "rotation_trick": self.rotation_trick,
            "quantize_optimize_method": self.quantize_optimize_method,
        }
