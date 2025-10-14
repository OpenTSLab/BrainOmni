import os
import mne
import numpy as np
import json
import torch
import time
import sys
import pandas as pd

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

DATASET_NAME = "Somato_EMEG" # or Somato_EEG or Somato_MEG
DATASET_PATH = os.path.join(EVALUATE_PATH, "ds006035-1.0.0")


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


def event_processing(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t")
    onset_list = []
    label_list = []

    for _, row in df.iterrows():
        onset = row["onset"]
        trial_type = row["trial_type"]
        if trial_type == "somatosensory":
            label = 0
        elif trial_type == "Finger":
            label = 1
        else:
            continue
        onset_list.append(onset)
        label_list.append(label)

    return onset_list, label_list


def preprocess_BrainOmni(accessor: DataAccessor, raw_path: str, processed_path: str):
    logger.info(f"Processing file {raw_path}")
    no_extension_processed_name = os.path.join(
        processed_path, raw_path.split("/")[-1].split(".")[0]
    )
    raw = accessor.read_brain_file(raw_path)

    indices = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
    raw.pick(indices)

    if "EEG" in DATASET_NAME:
        indices = mne.pick_types(raw.info, meg=False, eeg=True, ref_meg=False)
        raw.pick(indices)
        # some have more than 66, trucation
        indices = np.arange(66)
        raw.pick(indices)
    elif "MEG" in DATASET_NAME:
        indices = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False)
        raw.pick(indices)
    elif "EMEG" in DATASET_NAME:
        # trucation
        indices = np.arange(372)
        raw.pick(indices)

    # preprocess
    raw = raw.notch_filter(freqs=[50], verbose=False)
    raw = raw.filter(LOW, HIGH, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)

    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask)

    data = raw.get_data()
    # save
    os.makedirs(no_extension_processed_name, exist_ok=True)
    metadata_list = []

    evt_file = raw_path.replace("_meg.fif", "_events.tsv")
    onset_list, label_list = event_processing(evt_file)

    for onset, label in zip(onset_list, label_list):
        start = int((onset - 1) * SAMPLE_RATE)
        end = int(start + 2 * SAMPLE_RATE)
        if end >= data.shape[1]:
            continue
        seg_data = sensortype_wise_normalize(
            data[:, start:end], eeg_mask, mag_mask, grad_mask
        )
        assert accept_segment(seg_data, pos)
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
            "subject": raw_path.split("/")[-1].split("_")[0],
            "label": label,
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
                os.path.join(root, name)
                for name in names
                if (name.split(".")[-1] == "fif")
            ]

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
                    )
                )
            for future in as_completed(futures):
                try:
                    metadata_list += future.result()
                except Exception as e:
                    logger.info(f"An error occurred:{e}")
    else:
        for file_path in file_path_list:
            # try:
            result = method_preprocess_dict[method](accessor, file_path, processed_path)
            metadata_list += result
            # except Exception as e:
            #     logger.info(f"An error occurred:{e}")

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
