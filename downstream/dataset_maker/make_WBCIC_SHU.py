import os
import mne
import numpy as np
import json
import torch
import time
import sys
import pandas as pd
import scipy.io as sio

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from accessor import DataAccessor, write_torch_warpper
from constant import (
    SAMPLE_RATE,
    LOW,
    HIGH,
    EVALUATE_PATH,
    EVALUATE_METADATA_PATH,
    PROCESSED_EVALUATE_PATH,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from factory.utils import (
    extract_pos_sensor_type,
    get_sensor_type_mask,
    normalize_pos,
    sensortype_wise_normalize,
    accept_segment,
)

DATASET_NAME = "WBCIC_SHU"
DATASET_PATH = os.path.join(
    EVALUATE_PATH, DATASET_NAME, "derivatives", "2C dataset_processeddata"
)

ch_name_df = pd.read_csv(
    os.path.join(EVALUATE_PATH, DATASET_NAME, "task-motorimagery_channels.tsv"),
    sep="\t",
)
ch_name_df = ch_name_df[ch_name_df["name"] != "Pz"]
ch_names = ch_name_df["name"].tolist()
ch_types = ch_name_df["type"].tolist()
ch_types = [sample.lower() for sample in ch_types]
sfreq = 250
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

montage = mne.channels.read_custom_montage(
    os.path.join(EVALUATE_PATH, DATASET_NAME, "task-motorimagery_electrodes.tsv")
)


def get_logger():
    logger = logging.getLogger(name="processor")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
        "%H:%M:%S",
    )
    screenHandler = logging.StreamHandler()
    screenHandler.setLevel(logging.INFO)
    screenHandler.setFormatter(formatter)
    logger.addHandler(screenHandler)

    return logger


def preprocess_BrainOmni(accessor: DataAccessor, raw_path: str, processed_path: str):
    logger.info(f"Processing file {raw_path}")
    no_extension_processed_name = os.path.join(
        processed_path, raw_path.split("/")[-1].split(".")[0]
    )
    os.makedirs(no_extension_processed_name, exist_ok=True)

    matdata = sio.loadmat(raw_path)
    data = matdata["data"]
    label = matdata["labels"]
    data = np.transpose(data, (2, 0, 1))
    label = np.squeeze(label)

    metadata_list = []
    for i in range(data.shape[0]):
        data_to_save = data[i]
        label_to_save = int(label[i] - 1)
        raw = mne.io.RawArray(data_to_save, info, verbose=False)
        raw.set_montage(montage)

        raw = raw.notch_filter(freqs=50, verbose=False)
        raw = raw.filter(LOW, HIGH, verbose=False)
        raw = raw.resample(SAMPLE_RATE, verbose=False)

        pos, sensor_type = extract_pos_sensor_type(raw.info)
        eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
        pos = normalize_pos(pos, eeg_mask, mag_mask)

        data_to_save = raw.get_data()
        seg_data = sensortype_wise_normalize(
            data_to_save, eeg_mask, mag_mask, grad_mask
        )

        assert accept_segment(seg_data, pos)
        seg_data_path = os.path.join(no_extension_processed_name, f"{i}.pt")
        seg_data_dict = {
            "x": torch.from_numpy(seg_data),
            "pos": torch.from_numpy(pos),
            "sensor_type": torch.from_numpy(sensor_type),
        }
        accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
        metadata = {
            "path": seg_data_path,
            "subject": raw_path.split("/")[-1].split("_")[0],
            # "split": split,
            "label": label_to_save,
            "channels": seg_data.shape[0],
        }
        metadata_list.append(metadata)

    return metadata_list


def process(accessor: DataAccessor, method: str, logger, multi_process=True):
    method_preprocess_dict = {
        "BrainOmni": preprocess_BrainOmni,
    }

    processed_path = os.path.join(PROCESSED_EVALUATE_PATH, DATASET_NAME, method)
    os.makedirs(processed_path, exist_ok=True)
    file_path_list = []
    for root, dir, names in os.walk(DATASET_PATH):
        if len(names) > 0:
            file_path_list += [
                os.path.join(root, name) for name in names if name.endswith(".mat")
            ]

    metadata_list = []
    if multi_process:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for file_path in file_path_list:
                file_name = file_path.split("/")[-1]
                futures.append(
                    executor.submit(
                        method_preprocess_dict[method],
                        accessor,
                        file_path,
                        processed_path,
                    )
                )
            for future in as_completed(futures):
                try:
                    metadata_list += future.result()
                except Exception as e:
                    logger.info(f"An error occurred:{e}")
    else:
        for file_path in file_path_list:
            file_name = file_path.split("/")[-1]
            result = method_preprocess_dict[method](accessor, file_path, processed_path)
            try:
                metadata_list += result
            except Exception as e:
                logger.info(f"An error occurred:{e}")

    return metadata_list


if __name__ == "__main__":
    accessor = DataAccessor(read_only=False)
    logger = get_logger()
    for method in ["BrainOmni"]:
        info_path = os.path.join(
            EVALUATE_METADATA_PATH, DATASET_NAME, method, "info.json"
        )
        if os.path.exists(info_path):
            continue
        logger.info(f"start processing {method}")
        metadata_list = []
        os.makedirs(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method), exist_ok=True
        )
        metadata_list += process(accessor, method, logger, multi_process=True)
        metadata_list = sorted(metadata_list, key=lambda x: x["path"])
        with open(info_path, "w") as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
