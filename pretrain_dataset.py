import os
import json
import torch
import random
from constant import SEED
from constant import PRETRAIN_DTYPE
from accessor import DataAccessor, load_torch_warpper
from torch.utils.data import DataLoader, BatchSampler

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_list, accessor: DataAccessor):
        super().__init__()
        self.metadata_list = metadata_list
        self.accessor = accessor

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        data = self.accessor.read(
            self.metadata_list[idx]["path"],
            load_torch_warpper,
        )
        data["x"] = data["x"].to(PRETRAIN_DTYPE)
        data["pos"] = data["pos"].to(PRETRAIN_DTYPE)
        data["path"] = self.metadata_list[idx]["path"]
        return data


def collate_fn(batch):
    data = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            data[key] = torch.stack([i[key] for i in batch])
    data["path"] = [i["path"] for i in batch]
    return data


class Bucket:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size

    def append(self, x):
        self.data.append(x)

    def shuffle(self):
        random.shuffle(self.data)

    def __len__(self):
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        return len(self.data) // self.batch_size + 1

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.data[i * self.batch_size : (i + 1) * self.batch_size]
                if (i + 1) * self.batch_size <= len(self.data)
                else self.data[i * self.batch_size :]
            )


class BucketBatchSampler(BatchSampler):
    def __init__(self, channel_list, batch_size, rank):
        self.rank = rank
        channel_set = sorted(list(set(channel_list)))
        self.num_buckets = len(channel_set)
        self.buckets = {i: Bucket(batch_size) for i in range(self.num_buckets)}

        for idx, channel in enumerate(channel_list):
            bucket_idx = channel_set.index(channel)
            self.buckets[bucket_idx].append(idx)

        self.bucket_sample_sequence = []
        for i in range(self.num_buckets):
            self.bucket_sample_sequence += [i] * (len(self.buckets[i]) - 1)
        self.last_sample_sequence = [i for i in range(self.num_buckets)]

    def __iter__(self):
        rand = random.Random(SEED + self.rank)
        rand.shuffle(self.bucket_sample_sequence)
        for bucket_idx in range(self.num_buckets):
            self.buckets[bucket_idx].shuffle()
        buckets = [iter(self.buckets[i]) for i in range(self.num_buckets)]
        for i in self.bucket_sample_sequence+self.last_sample_sequence:
            yield next(buckets[i])

    def __len__(self):
        return sum([len(self.buckets[i]) for i in range(self.num_buckets)])


def build_brain_bucket_dataloader(
    mode,
    ratio,
    metadata_path,
    accessor,
    signal_type: str,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool = False,
):
    with open(os.path.join(metadata_path, f"{mode}.json"), "r") as f:
        metadata_list = json.load(f)
    if signal_type == "eeg":
        metadata_list = [i for i in metadata_list if i["is_eeg"]]
    elif signal_type == "meg":
        metadata_list = [i for i in metadata_list if i["is_meg"]]
    # 多卡划分
    channels_set = sorted(set([i["channels"] for i in metadata_list]))
    replicated_metadata_list = []
    for channels in channels_set:
        channel_metadata_list = [i for i in metadata_list if i["channels"] == channels]
        random.shuffle(channel_metadata_list)
        channel_metadata_list = channel_metadata_list[
            : int(len(channel_metadata_list) * ratio)
        ]
        len_replicas = len(channel_metadata_list) // world_size
        replicated_metadata_list += channel_metadata_list[
            rank * len_replicas : (rank + 1) * len_replicas
        ]
        replicated_metadata_list += channel_metadata_list[world_size * len_replicas :]
    brain_dataset = BrainDataset(
        metadata_list=replicated_metadata_list, accessor=accessor
    )

    return DataLoader(
        dataset=brain_dataset,
        batch_sampler=BucketBatchSampler(
            [i["channels"] for i in replicated_metadata_list],
            batch_size=batch_size,
            rank=rank,
        ),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )