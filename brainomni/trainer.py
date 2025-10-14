import os
import json
import torch
import logging
import deepspeed
import deepspeed.comm as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accessor import DataAccessor
from brainomni.config import BrainOmniTrainerConfig
from brainomni.model import BrainOmni
from pretrain_dataset import build_brain_bucket_dataloader
from constant import NEW_DEVICE_DATASET_LIST


class EmptyLogger:
    def info(self, *awargs, **kwargs):
        return None


class EmptyWriter:
    def add_scalar(self, *awargs, **kwargs):
        return None

    def close(self):
        return None

    def add_figure(self, *awargs, **kwargs):
        return None


class Trainer:
    def __init__(
        self,
        cfg: BrainOmniTrainerConfig,
        local_rank: int,
        rank: int,
        world_size: int,
        exp_path: str,
    ):
        # prepare basic environment
        self.cfg = cfg
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.exp_path = exp_path
        self.ckpt_path = os.path.join(self.exp_path, "checkpoint")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.accessor = DataAccessor(
            read_only=True
        )

        # configuration
        self.epoch = 0
        self.total_epoch = cfg.epoch
        self.best_eval_loss = 1000000000000
        self.logger = self.build_logger()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()

        self.logger.info("=> Building train dataloader ...")
        self.train_loader = self.build_dataloader(
            mode="train", ratio=self.cfg.train_data_ratio, persistent_workers=True
        )
        self.logger.info("=> Building val dataloader ...")
        self.val_loader = self.build_dataloader(
            mode="val", ratio=self.cfg.valid_data_ratio
        )

        self.train_step_counter = 0
        train_total_steps = (
            len(self.train_loader)
            * self.total_epoch
            // self.cfg.gradient_accumulation_steps
        )

        self.logger.info(
            "=> Building model and initializing distributed environment..."
        )
        self.cfg.ds_config["scheduler"]["params"]["total_num_steps"] = train_total_steps
        self.cfg.ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
            train_total_steps * self.cfg.scheduler_warm_ratio
        )
        self.model = self.deepspeed_initialize()

    def main(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
        while self.epoch < self.total_epoch:
            self.count_epoch()
            self.before_epoch()
            self.model.train()
            if self.rank == 0:
                with tqdm(self.train_loader, unit="batch") as tepoch:
                    for self.input_dict in tepoch:
                        tepoch.set_description(f"Epoch {self.epoch}")
                        tepoch.set_postfix(self.train_step())
                        break
            else:
                for self.input_dict in self.train_loader:
                    self.train_step()
                    break
            self.model.eval()
            if self.rank == 0:
                with tqdm(self.val_loader, unit="batch") as tepoch:
                    for self.input_dict in tepoch:
                        tepoch.set_description(f"Epoch {self.epoch}")
                        tepoch.set_postfix(self.eval_step())
                        break
            else:
                for self.input_dict in self.val_loader:
                    self.eval_step()
                    break
            self.after_epoch()
        self.logger.info(">>>>>>>>>>>>>>>> Finish Training >>>>>>>>>>>>>>>>")
        self.load_ckpt(
            load_dir=os.path.join(self.exp_path, "checkpoint"),
            tag="best",
        )
        self.model.eval()
        del self.train_loader
        del self.val_loader

        self.logger.info("=> Start Testing ...")
        for mode in ["test"] + NEW_DEVICE_DATASET_LIST:
            self.test_loader = self.build_dataloader(mode=mode, ratio=1.0)
            if len(self.test_loader) == 0:
                continue
            self.test_running_dict = {"loss": 0.0, "acc_all": 0.0}
            for i in range(self.cfg.num_quantizers_used):
                self.test_running_dict[f"acc_{i}"] = 0.0
            with torch.no_grad():
                for self.input_dict in self.test_loader:
                    input_dict = self.fetch_input_dict()
                    output_dict = self.model(**input_dict)
                    for key in self.test_running_dict.keys():
                        self.test_running_dict[key] += output_dict[key].item()
                    break

            torch.cuda.empty_cache()
            for key in self.test_running_dict.keys():
                self.test_running_dict[key] = self.test_running_dict[key] / len(
                    self.test_loader
                )
                self.test_running_dict[key] = self.scalar_comm_reduce(
                    self.test_running_dict[key]
                )
            with open(os.path.join(self.exp_path, f"{mode}_metrics.json"), "w") as f:
                json.dump(self.test_running_dict, f, indent=4)
        self.writer.close()
        dist.destroy_process_group()

    def count_epoch(self):
        self.epoch += 1

    def before_epoch(self):
        self.logger.info(f">>>>>>>>>>>>>>>> Epoch {self.epoch} >>>>>>>>>>>>>>>>")
        self.train_running_dict = {"loss": 0.0, "acc_all": 0.0}
        self.eval_running_dict = {"loss": 0.0, "acc_all": 0.0}
        for i in range(self.cfg.num_quantizers_used):
            self.train_running_dict[f"acc_{i}"] = 0.0
            self.eval_running_dict[f"acc_{i}"] = 0.0

    def fetch_input_dict(self):
        input_dict = self.input_dict
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(
                    device=self.local_rank, non_blocking=True
                )
        return input_dict

    def train_step(self):
        input_dict = self.fetch_input_dict()
        output_dict = self.model(**input_dict)
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] += output_dict[key].item()
        loss = output_dict["loss"]
        self.model.backward(loss)
        self.model.step()
        self.train_step_counter += 1
        return tqdm_dict

    @torch.no_grad()
    def eval_step(self):
        input_dict = self.fetch_input_dict()
        output_dict = self.model(**input_dict)
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] += output_dict[key].item()
        return tqdm_dict

    def scalar_comm_reduce(self, scalar, op=dist.ReduceOp.AVG):
        tensor_scalar = torch.tensor(
            [scalar], device=self.local_rank, dtype=torch.float32
        )
        dist.all_reduce(tensor_scalar, op=op)
        return tensor_scalar.item()

    def after_epoch(self):
        torch.cuda.empty_cache()
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] = self.train_running_dict[key] / len(
                self.train_loader
            )
            self.train_running_dict[key] = self.scalar_comm_reduce(
                self.train_running_dict[key]
            )
            self.writer.add_scalar(
                tag=f"train_{key}",
                scalar_value=self.train_running_dict[key],
                global_step=self.epoch,
            )
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] = self.eval_running_dict[key] / len(
                self.val_loader
            )
            self.eval_running_dict[key] = self.scalar_comm_reduce(
                self.eval_running_dict[key]
            )
            self.writer.add_scalar(
                tag=f"eval_{key}",
                scalar_value=self.eval_running_dict[key],
                global_step=self.epoch,
            )
            self.logger.info(
                f"train {key}:{self.train_running_dict[key]} eval {key}:{self.eval_running_dict[key]}"
            )

        self.logger.info("")
        if self.epoch % 20 == 0:
            self.save_ckpt(tag=f"epoch_{self.epoch}")
        if self.epoch == self.total_epoch:
            self.save_ckpt(tag="last")
        if self.eval_running_dict["loss"] < self.best_eval_loss:
            self.best_eval_loss = self.eval_running_dict["loss"]
            self.save_ckpt(tag="best")

    def save_ckpt(self, tag: str):
        """
        epoch
        best_eval_loss
        save summary writter
        save logger
        ckpt
        optimizer
        scheduler
        """
        self.model.save_checkpoint(
            save_dir=self.ckpt_path,
            tag=tag,
        )

    def load_ckpt(self, load_dir: str, tag: str):
        load_path, client_state = self.model.load_checkpoint(load_dir=load_dir, tag=tag)

    def deepspeed_initialize(self):
        model = BrainOmni(**self.cfg.get_model_cfg())
        model.load_frozen_tokenizer_ckpt(
            tokenizer_ckpt_path=self.cfg.tokenizer_ckpt_path
        )
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.get_parameters_groups(
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            ),
            config=self.cfg.ds_config,
        )
        n_parameters = sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and "predict_head" not in n
        )
        self.logger.info(f"Num params: {n_parameters/1.0e9} B")
        return model

    def build_writer(self):
        if self.rank != 0:
            return EmptyWriter()
        writer = SummaryWriter(self.exp_path)
        self.logger.info(f"Tensorboard writer logging dir: {self.exp_path}")
        return writer

    def build_dataloader(self, mode, ratio, batch_size=None, persistent_workers=False):
        return build_brain_bucket_dataloader(
            mode=mode,
            ratio=ratio,
            metadata_path=self.cfg.pretrain_metadata_path,
            accessor=self.accessor,
            signal_type=self.cfg.signal_type,
            rank=self.rank,
            world_size=self.world_size,
            batch_size=batch_size if batch_size else self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=persistent_workers,
        )

    def build_logger(self):
        if self.rank != 0:
            return EmptyLogger()
        logger = logging.getLogger(name="BrainGPT")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            "%H:%M:%S",
        )

        fileHandler = logging.FileHandler(
            f"{self.exp_path}/logs.txt",
            encoding="utf-8",
        )
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        screenHandler = logging.StreamHandler()
        screenHandler.setLevel(logging.INFO)
        screenHandler.setFormatter(formatter)
        logger.addHandler(screenHandler)

        logger.info(f"Save experiment in {self.exp_path}")
        return logger
