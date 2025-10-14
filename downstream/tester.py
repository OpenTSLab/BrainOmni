import os
import json
import torch
import logging
import deepspeed
from tqdm import tqdm
from accessor import DataAccessor
import deepspeed.comm as dist
from torch.utils.tensorboard import SummaryWriter
from downstream.dataset import DownStreamClassifyDataset, collate_fn
from downstream.model import DownstreamModel
from downstream.config import DownstreamConfig
from downstream.metrics import MetricsComputer
from downstream.model_collection.BrainOmni import get_brainomni
from constant import STANDARD_1020


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


class DownStreamTester:
    def __init__(
        self,
        cfg: DownstreamConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        exp_path: str,
    ):
        # prepare basic environment
        self.cfg = cfg
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.exp_path = exp_path
        self.accessor = DataAccessor(read_only=True)
        self.ckpt_path = os.path.join(exp_path, "checkpoint")
        os.makedirs(self.ckpt_path, exist_ok=True)  # use to save linear matrix

        # config
        self.epoch = 0
        self.total_epoch = cfg.epoch
        self.best_bacc = 0.0
        self.eval_worse_than_best_counter = 0
        self.current_eval_bacc = 0
        self.eval_worse_counter = 0
        self.logger = self.build_logger()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()

        # [x,pos,sensor_type,ch_names]
        self.logger.info(f"=> Building dataloader...")
        self.train_loader = self.build_dataloader(mode="train")
        self.val_loader = self.build_dataloader(mode="val")
        self.test_loader = self.build_dataloader(mode="test")

        train_total_steps = (
            len(self.train_loader)
            * self.total_epoch
            // self.cfg.gradient_accumulation_steps
        )
        # backbone is a normal nn.Module and downstream_model is warpped by deepspeed
        self.logger.info("=> Building frozen backbone and learnable linear head...")
        self.cfg.ds_config["scheduler"]["params"]["total_num_steps"] = train_total_steps
        self.cfg.ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
            train_total_steps * self.cfg.scheduler_warm_ratio
        )
        self.downstream_model = self.build_model()

        # 接收B个predict和ground truth，计算各种指标
        self.logger.info(f"=> Building classify metrics computer...")
        self.metrics_computer = MetricsComputer(
            num_classes=cfg.num_classes, is_binary=(cfg.num_classes == 2)
        )

    def main(self):
        if os.path.exists(os.path.join(self.exp_path, "metrics.json")):
            self.logger.info(
                ">>>>>>>>>>>>>>>> Train finished before. Skip training. >>>>>>>>>>>>>"
            )
        else:
            self.logger.info(">>>>>>>>>>>>>>>> Start Finetuning >>>>>>>>>>>>>")
            while self.epoch < self.total_epoch:
                self.downstream_model.train()
                self.train_epoch()
                self.downstream_model.eval()
                metrics = self.eval_epoch("val")

                if metrics["balanced_accuracy"] > self.best_bacc:
                    self.save_ckpt(tag="best")
                    self.best_bacc = metrics["balanced_accuracy"]
                    self.eval_worse_than_best_counter = 0
                else:
                    self.eval_worse_than_best_counter += 1

                if metrics["balanced_accuracy"] < self.current_eval_bacc:
                    self.eval_worse_counter += 1
                else:
                    self.eval_worse_counter = 0
                self.current_eval_bacc = metrics["balanced_accuracy"]

                # for early stop
                if self.eval_worse_counter == 5:
                    self.logger.info(
                        "val performance drop for continuous 5 epoch, early stop!"
                    )
                    break
                if (
                    self.eval_worse_than_best_counter == 8
                    and self.cfg.dataset_name in ["TUAB", "TUEV"]
                ):  # TUAB and TUEV are time consuming, add another early stop
                    self.logger.info(
                        "val performance worse than best for 8 epoch, early stop!"
                    )
                    break
                self.count_epoch()

            self.logger.info(">>>>>>>>>>>>>>>> Finish Training >>>>>>>>>>>>>>>>")
            self.load_ckpt(
                load_dir=os.path.join(self.exp_path, "checkpoint"), tag="best"
            )
            self.downstream_model.eval()
            test_metric = self.eval_epoch("test")
            with open(os.path.join(self.exp_path, "metrics.json"), "w") as f:
                json.dump(test_metric, f, indent=4)

            self.writer.close()
            dist.destroy_process_group()

    def count_epoch(self):
        self.epoch += 1

    def fetch_input_dict(self, input_dict):
        # input_dict = self.input_dict
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(
                    device=self.local_rank, non_blocking=True
                )
        y = input_dict.pop("y")
        return input_dict, y

    def train_epoch(self):
        self.downstream_model.train()
        local_labels = []
        local_logits = []
        if dist.get_rank() == 0:
            loop = tqdm(
                self.train_loader, unit="batch", desc=f"Train Epoch {self.epoch}"
            )
        else:
            loop = self.train_loader

        for i, batch in enumerate(loop):
            input_dict, label = self.fetch_input_dict(batch)
            logits, loss = self.downstream_model(input_dict, label)
            self.downstream_model.backward(loss)
            self.downstream_model.step()

            running_loss = self.scalar_comm_reduce(loss)
            self.writer.add_scalar(
                tag="train/running_loss",
                scalar_value=running_loss,
                global_step=(self.epoch * len(self.train_loader) + i),
            )
            local_labels.append(label.long())
            local_logits.append(logits.float())

        local_labels = torch.concat(local_labels)
        local_logits = torch.concat(local_logits)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gathered_label_list = [
            torch.zeros_like(local_labels) for _ in range(world_size)
        ]
        gathered_logit_list = [
            torch.zeros_like(local_logits) for _ in range(world_size)
        ]
        dist.all_gather(gathered_label_list, local_labels)
        dist.all_gather(gathered_logit_list, local_logits)
        gathered_labels = torch.concat(gathered_label_list).cpu()
        gathered_logits = torch.concat(gathered_logit_list).cpu()

        gathered_prob = torch.softmax(gathered_logits, dim=-1)
        metrics = self.metrics_computer.compute_metrics(
            prob=gathered_prob, gts=gathered_labels
        )
        loss = torch.nn.functional.cross_entropy(
            gathered_logits, gathered_labels, label_smoothing=self.cfg.label_smoothing
        )

        if self.rank == 0:
            for key in metrics:
                self.writer.add_scalar(
                    tag=f"train/{key}",
                    scalar_value=metrics[key],
                    global_step=self.epoch,
                )
            self.writer.add_scalar(
                tag="train/loss", scalar_value=loss, global_step=self.epoch
            )
            self.logger.info(
                f"Train Result in Epoch {self.epoch}: {str(metrics)}, loss: {loss}"
            )

    @torch.no_grad()
    def eval_epoch(self, mode):
        self.downstream_model.eval()
        dataloader = getattr(self, f"{mode}_loader", None)
        assert dataloader is not None, f"{mode}_loader does not exist."
        local_labels = []
        local_logits = []

        if dist.get_rank() == 0:
            loop = tqdm(
                dataloader,
                unit="batch",
                desc=f"Eval Epoch {self.epoch}" if mode == "val" else "Test",
            )
        else:
            loop = dataloader

        for batch in loop:
            input, label = self.fetch_input_dict(batch)
            logits, feature = self.downstream_model.predict(input)

            local_labels.append(label.long())
            local_logits.append(logits.float())

        local_labels = torch.concat(local_labels)
        local_logits = torch.concat(local_logits)
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        gathered_label_list = [
            torch.zeros_like(local_labels) for _ in range(world_size)
        ]
        gathered_logit_list = [
            torch.zeros_like(local_logits) for _ in range(world_size)
        ]
        # if dist.is_initialized():
        dist.all_gather(gathered_label_list, local_labels)
        dist.all_gather(gathered_logit_list, local_logits)
        gathered_labels = torch.concat(gathered_label_list).cpu()
        gathered_logits = torch.concat(gathered_logit_list).cpu()

        gathered_prob = torch.softmax(gathered_logits, dim=-1)
        metrics = self.metrics_computer.compute_metrics(
            prob=gathered_prob, gts=gathered_labels
        )
        loss = torch.nn.functional.cross_entropy(
            gathered_logits, gathered_labels, label_smoothing=self.cfg.label_smoothing
        )

        # Log
        if self.rank == 0:
            if mode == "val":
                for key in metrics:
                    self.writer.add_scalar(
                        tag=f"eval/{key}",
                        scalar_value=metrics[key],
                        global_step=self.epoch,
                    )
                self.writer.add_scalar(
                    tag="eval/loss", scalar_value=loss, global_step=self.epoch
                )
                self.logger.info(
                    f"Eval Result in Epoch {self.epoch}: {str(metrics)}, loss: {loss}"
                )

            if mode == "test":
                self.logger.info(f"Eval Result for Test: {str(metrics)}, loss: {loss}")
                torch.save(gathered_labels, os.path.join(self.exp_path, f"label.pt"))
                torch.save(gathered_logits, os.path.join(self.exp_path, f"logit.pt"))
        return metrics

    def scalar_comm_reduce(self, scalar, op=dist.ReduceOp.AVG):
        tensor_scalar = torch.tensor(
            [scalar], device=self.local_rank, dtype=torch.float32
        )
        dist.all_reduce(tensor_scalar, op=op)
        return tensor_scalar.item()

    def save_ckpt(self, tag: str):
        self.downstream_model.save_checkpoint(
            save_dir=self.ckpt_path,
            tag=tag,
        )

    def load_ckpt(self, load_dir: str, tag: str):
        load_path, client_state = self.downstream_model.load_checkpoint(
            load_dir=load_dir, tag=tag
        )

    def build_model(self):
        # initialize lazy module
        for batch in self.train_loader:
            input, _ = self.fetch_input_dict(batch)
            for key in input.keys():
                if key in ["x", "pos"]:
                    input[key] = input[key].float()
            break

        # build model and load ckpt
        backbone, n_dim = get_brainomni(
            pretrained=self.cfg.pretrained,
            ckpt_path=self.cfg.ckpt_path,
        )

        downstream_model = DownstreamModel(
            backbone=backbone,
            frozen=self.cfg.frozen,
            n_dim=n_dim,
            num_classes=self.cfg.num_classes,
            label_smoothing=self.cfg.label_smoothing,
        ).to(self.local_rank)

        # init lazy module
        downstream_model.predict(input)

        # init deepspeed
        downstream_model, _, _, _ = deepspeed.initialize(
            model=downstream_model,
            model_parameters=downstream_model.get_parameters_groups(
                backbone_lr=self.cfg.backbone_lr,
                head_lr=self.cfg.head_lr,
                weight_decay=self.cfg.weight_decay,
            ),
            config=self.cfg.ds_config,
        )
        return downstream_model

    def build_dataloader(self, mode):
        dataset = DownStreamClassifyDataset(
            model_name=self.cfg.model_name,
            dataset_name=self.cfg.dataset_name,
            mode=mode,
            n_fold=self.cfg.n_fold,
            fold_index=self.cfg.fold_index,
            accessor=self.accessor,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=collate_fn,
            sampler=torch.utils.data.DistributedSampler(
                dataset=dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            ),
        )

    def build_writer(self):
        if self.rank != 0:
            return EmptyWriter()
        writer = SummaryWriter(os.path.join(self.exp_path, "tensorboard"))
        self.logger.info(f"Tensorboard writer logging dir: {self.exp_path}")
        return writer

    def build_logger(self):
        if self.rank != 0:
            return EmptyLogger()
        logger = logging.getLogger(name="Brain")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            "%H:%M:%S",
        )

        fileHandler = logging.FileHandler(
            os.path.join(self.exp_path, "logs.txt"), mode="w"
        )
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        screenHandler = logging.StreamHandler()
        screenHandler.setLevel(logging.INFO)
        screenHandler.setFormatter(formatter)
        logger.addHandler(screenHandler)

        logger.info(f"Save evaluation in {self.exp_path}")
        return logger
