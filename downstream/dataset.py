import os
import json
import torch
import torch.utils
from constant import DOWNSTREAM_DTYPE, EVALUATE_METADATA_PATH
from accessor import DataAccessor, load_torch_warpper


class DownStreamClassifyDataset(torch.utils.data.Dataset):
    """
    array
    label

    path dataset subject train val
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        mode: str,
        n_fold: int,
        fold_index: int,
        accessor: DataAccessor,
    ):
        super().__init__()
        assert mode in ["train", "val", "test"]
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.accessor = accessor
        metadata_path = os.path.join(
            EVALUATE_METADATA_PATH,
            dataset_name,
            model_name,
            f"{n_fold}_fold",
            f"{mode}_{fold_index}_fold.json",
        )
        metadata_list = json.load(open(metadata_path, "r"))
        self.ch_names = None
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        data = self.accessor.read(
            self.metadata_list[idx]["path"],
            load_torch_warpper,
        )
        data["x"] = data["x"].to(DOWNSTREAM_DTYPE)
        data["pos"] = data["pos"].to(DOWNSTREAM_DTYPE) if data["pos"] != None else None
        data["y"] = torch.tensor(self.metadata_list[idx]["label"])
        data["ch_names"] = self.ch_names

        return data


def collate_fn(batch):
    data = {}
    for key in batch[0].keys():
        if batch[0][key] == None:
            continue
        if isinstance(batch[0][key], torch.Tensor):
            data[key] = torch.stack([i[key] for i in batch])
    if batch[0]["ch_names"] != None:
        data["ch_names"] = batch[0]["ch_names"]
    return data
