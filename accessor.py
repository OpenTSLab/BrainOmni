import os
import io
import mne
import json
import torch
from constant import DATA_ROOT_PATH
from tqdm import tqdm
BRAIN_EXTENSION = ["con", "fif", "set", "bdf", "edf", "vhdr", "ds"]


def is_useful_path(x):
    if "empty" in x or "noise" in x or '"split-02_meg.fif' in x or "hz.ds" in x:
        return False
    return True


def load_torch_warpper(path):
    return torch.load(path, weights_only=True)


def write_torch_warpper(data, path):
    torch.save(data, path)


def load_json_warpper(path):
    if isinstance(path, str):
        with open(path, "r") as f:
            return json.load(f)
    assert isinstance(path, io.BytesIO)
    return json.loads(path.getvalue().decode("utf-8"))


def write_json_warpper(data, path):
    if isinstance(path, str):
        with open(path, "w") as f:
            return json.dump(data, f, indent=4)
    assert isinstance(path, io.BytesIO)
    json_string = json.dumps(data, indent=4)
    return path.write(json_string.encode("utf-8"))


class DataAccessor:
    def __init__(self, read_only: bool = True):
        self.read_only = read_only
        self.brain_read_func_dict = {
            "fif": mne.io.read_raw_fif,
            "con": mne.io.read_raw_kit,
            "bdf": mne.io.read_raw_bdf,
            "edf": mne.io.read_raw_edf,
            "vhdr": mne.io.read_raw_brainvision,
            "ds": mne.io.read_raw_ctf,
            "set": mne.io.read_raw_eeglab,
            "cnt": mne.io.read_raw_cnt,
            "gdf": mne.io.read_raw_gdf,
        }

    def search_brain_files(self, root_path: str):
        brain_files = []
        for root, dir, name in tqdm(os.walk(root_path)):
            if len(name) > 0:
                for i in name:
                    if i.split(".")[-1] in BRAIN_EXTENSION and is_useful_path(i):
                        brain_files.append(
                            {
                                "path": os.path.join(root, i),
                                "dataset": self.get_dataset_folder_name(root),
                            }
                        )
            if len(dir) > 0:
                for i in dir:
                    if i.split(".")[-1] == "ds" and is_useful_path(i):
                        brain_files.append(
                            {
                                "path": os.path.join(root, i),
                                "dataset": self.get_dataset_folder_name(root),
                            }
                        )
        return brain_files

    def read_brain_file(self, path: str, preload: bool = True):
        extension = path.rsplit(".")[-1]
        return self.brain_read_func_dict[extension](
            path, verbose=False, preload=preload
        )

    def get_usage_folder_name(self, path: str):
        return path.replace(DATA_ROOT_PATH, "").split("/")[0]

    def get_dataset_folder_name(self, path: str):
        return path.replace(DATA_ROOT_PATH, "").split("/")[1]

    def replace_usage_folder_name(self, path: str, new_usage: str):
        return path.replace(f"/{self.get_usage_folder_name(path)}/", f"/{new_usage}/")

    def mkdir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def exist(self, path: str):
        return os.path.exists(path)

    def read(self, path: str, read_func):
        assert self.exist(path)
        return read_func(path)

    def write(self, data, path: str, write_func):
        if self.read_only or self.get_usage_folder_name(path) in [
            "raw",
            "evaluate",
        ]:
            return None
        write_func(data, path)

    def remove(self, path):
        if self.read_only:
            return None
        assert self.exist(path)
        os.remove(path)
