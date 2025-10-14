DATASET_NUM_CLASSES_DICT = {
    "TUAB": 2,
    "TUEV": 6,
    "ad65": 2,
    "MDD": 2,
    "pd31": 2,
    "WBCIC_SHU": 2,
    "FACED": 3,
    "PhysioNet-MI": 4,
    "asd74": 2,
    "Somato_EMEG": 2,
    "Somato_EEG": 2,
    "Somato_MEG": 2,
    "MEG-MMI": 2,
}

class DownstreamConfig:
    def __init__(
        self,
        dataset_name,
        ckpt_path,
        pretrained,
        frozen,
        epoch,
        n_fold,
        fold_index,
        head_lr,
        backbone_lr,
        world_size,
    ):
        self.model_name = 'BrainOmni'
        self.pretrained = pretrained
        self.frozen = frozen
        self.dataset_name = dataset_name
        self.num_classes = DATASET_NUM_CLASSES_DICT[dataset_name]
        self.ckpt_path = ckpt_path
        self.n_fold = n_fold
        self.fold_index = fold_index

        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.weight_decay = 0.05

        self.epoch = epoch

        self.label_smoothing = 0.1

        self.total_batch_per_update = 128
        self.batch_size = 16
        
        if dataset_name=='MEG-MMI':
            self.batch_size=8   # prevent out of memory
              
        self.num_workers = 16
        self.gradient_accumulation_steps = int(
            self.total_batch_per_update // (self.batch_size * world_size)
        )
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
                "params": {"betas": [0.9, 0.99], "eps": 1e-6},
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
