import os
import mne
import json
import time
import torch
import numpy as np
import sys

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


TUAB_PATH = os.path.join(EVALUATE_PATH, "tuh_eeg_abnormal", "edf")
CHANNEL_STANDARD = "01_tcp_ar"
DATASET_NAME = "TUAB"


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


def acquire_folder_path(split: str, label: str):
    return os.path.join(TUAB_PATH, split, label, CHANNEL_STANDARD)


def preprocess_BrainOmni(
    accessor: DataAccessor, edf_path: str, split: str, label: int, processed_path: str
):

    logger.info(f"Processing file {edf_path}")
    no_extension_processed_name = os.path.join(
        processed_path, edf_path.split("/")[-1][:-4]
    )
    raw = accessor.read_brain_file(edf_path)
    # channels and pos
    raw.rename_channels(
        {
            i: i.split("-")[0].split("EEG ")[-1].replace("FP", "Fp").replace("Z", "z")
            for i in raw.info.ch_names
        }
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    exclude = [i for i in raw.info.ch_names if i not in montage.ch_names]
    indices = mne.pick_types(
        raw.info, meg=True, eeg=True, ref_meg=False, exclude=exclude
    )
    raw.pick(indices)
    raw.set_montage(montage)

    # preprocess
    raw = raw.notch_filter(60, verbose=False)
    raw = raw.filter(LOW, HIGH, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)

    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask)

    # split and save
    data = raw.get_data()
    start = 0
    end = int(SAMPLE_RATE * 10)
    stride_length = int(SAMPLE_RATE * 10)

    metadata_list = []
    os.makedirs(no_extension_processed_name, exist_ok=True)
    while end < data.shape[1]:
        seg_data = sensortype_wise_normalize(
            data[:, start:end], eeg_mask, mag_mask, grad_mask
        )
        assert accept_segment(seg_data, pos) and seg_data.shape[0] == 21
        seg_data_path = os.path.join(
            no_extension_processed_name, f"{len(metadata_list)}.pt"
        )
        seg_data_dict = {
            "x": torch.from_numpy(seg_data),
            "pos": torch.from_numpy(pos),
            "sensor_type": torch.from_numpy(sensor_type),
        }
        accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
        metadata = {
            "path": seg_data_path,
            "subject": edf_path.split("/")[-1].split("_")[0],
            "split": split,
            "label": label,
            "channels": seg_data.shape[0],
        }
        metadata_list.append(metadata)
        start += stride_length
        end += stride_length
    return metadata_list


def process(
    accessor: DataAccessor,
    split: str,
    label: str,
    method: str,
    logger,
    multi_process=True,
):
    assert split in ["train", "eval"]
    assert label in ["abnormal", "normal"]
    method_preprocess_dict = {
        "BrainOmni": preprocess_BrainOmni,
    }

    processed_path = os.path.join(PROCESSED_EVALUATE_PATH, DATASET_NAME, method)
    os.makedirs(processed_path, exist_ok=True)

    root_path = acquire_folder_path(split, label)
    files = [os.path.join(root_path, i) for i in os.listdir(root_path)]
    metadata_list = []
    if multi_process:
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for i in files:
                futures.append(
                    executor.submit(
                        method_preprocess_dict[method],
                        accessor,
                        i,
                        split,
                        1 if label == "abnormal" else 0,
                        processed_path,
                    )
                )
            for future in as_completed(futures):
                try:
                    metadata_list += future.result()
                except Exception as e:
                    logger.info(f"An error occurred:{e}")
    else:
        for i in files:
            result = method_preprocess_dict[method](
                accessor, i, split, 1 if label == "abnormal" else 0, processed_path
            )
            try:
                metadata_list += result
            except Exception as e:
                logger.info(f"An error occurred:{e}")

    return metadata_list


if __name__ == "__main__":
    accessor = DataAccessor(read_only=False)
    logger = get_logger()
    for method in ["BrainOmni"]:
        if os.path.exists(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method, "info.json")
        ):
            continue
        logger.info(f"start processing {method}")
        metadata_list = []
        os.makedirs(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method), exist_ok=True
        )
        for split in ["train", "eval"]:
            for label in ["abnormal", "normal"]:
                metadata_list += process(
                    accessor, split, label, method, logger, multi_process=True
                )
        metadata_list = sorted(metadata_list, key=lambda x: x["path"])
        with open(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method, "info.json"), "w"
        ) as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
