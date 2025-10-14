import os
import mne
import numpy as np
import json
import torch
import sys
import pickle

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from accessor import DataAccessor, write_torch_warpper
from constant import (
    SAMPLE_RATE,
    LOW,
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
)

DATASET_NAME = "PhysioNet-MI"
DATASET_PATH = os.path.join(EVALUATE_PATH, "eeg-motor-movement", "files")
image_tasks = [4, 6, 8, 10, 12, 14]

RAW_INFO=pickle.load(open('./share/custom_montages/physio_info.pkl','rb'))

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


def preprocess_BrainOmni(
    accessor: DataAccessor, raw_path: str, processed_path: str, task: int
):
    logger.info(f"Processing file {raw_path}")
    no_extension_processed_name = os.path.join(
        processed_path, raw_path.split("/")[-1].split(".")[0]
    )
    os.makedirs(no_extension_processed_name, exist_ok=True)
    metadata_list = []

    raw = accessor.read_brain_file(raw_path)

    indices = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
    raw.pick(indices)

    raw = raw.notch_filter((60), verbose=False)
    raw = raw.filter(LOW, h_freq=None, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)

    pos, sensor_type = extract_pos_sensor_type(RAW_INFO)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask)

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events_from_annot,
        event_dict,
        tmin=0,
        tmax=4.0 - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    data = epochs.get_data()
    events = epochs.events[:, 2]
    data = data[:, :, -SAMPLE_RATE * 4 :]

    for i, (seg_data, event) in enumerate(zip(data, events)):
        if event != 1:
            seg_data = sensortype_wise_normalize(
                seg_data, eeg_mask, mag_mask, grad_mask
            )
            seg_data_path = os.path.join(
                no_extension_processed_name, f"{len(metadata_list)}.pt"
            )
            label = int(event - 2 if task in [4, 8, 12] else event)
            seg_data_dict = {
                "x": torch.from_numpy(seg_data),
                "pos": torch.from_numpy(pos),
                "sensor_type": torch.from_numpy(sensor_type),
            }
            accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
            metadata_list.append(
                {
                    "path": seg_data_path,
                    "subject": int(raw_path.split("/")[-1].split("R")[0][1:]),
                    "label": label,
                    "channels": seg_data.shape[0],
                }
            )

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
            for name in names:
                if name.endswith(".edf"):
                    task = int(name.split(".")[0].split("R")[-1])
                    if task in image_tasks:
                        file_path_list.append(os.path.join(root, name))

    metadata_list = []
    if multi_process:
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for file_path in file_path_list:
                futures.append(
                    executor.submit(
                        method_preprocess_dict[method],
                        accessor,
                        file_path,
                        processed_path,
                        int(file_path.rsplit("/", 1)[-1].split(".")[0].split("R")[-1]),
                    )
                )
            for future in as_completed(futures):
                try:
                    metadata_list += future.result()
                except Exception as e:
                    logger.info(f"An error occurred:{e}")
    else:
        for file_path in file_path_list:
            result = method_preprocess_dict[method](
                accessor,
                file_path,
                processed_path,
                int(file_path.rsplit("/", 1)[-1].split(".")[0].split("R")[-1]),
            )
            metadata_list += result

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
