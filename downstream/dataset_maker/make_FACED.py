import os
import mne
import json
import torch
import time
import sys
import pickle

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from accessor import DataAccessor, write_torch_warpper
from constant import (
    SAMPLE_RATE,
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


DATASET_NAME = "FACED"
DATASET_PATH = os.path.join(EVALUATE_PATH, "FACED", "Processed_data")

ch_names = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "Cz",
    "C3",
    "C4",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
    "A1",
    "A2",
]
ch_types = ["eeg" for sample in ch_names]
sfreq = 250
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

labels_3 = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
]

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
    metadata_list = []

    with open(raw_path, "rb") as f:
        pkldata = pickle.load(f)

    for i in range(pkldata.shape[0]):
        raw = mne.io.RawArray(pkldata[i], info=info, verbose=False)
        label = labels_3[i]
        montage = mne.channels.make_standard_montage("standard_1020")
        exclude = [i for i in raw.info.ch_names if i not in montage.ch_names]
        indices = mne.pick_types(
            raw.info, meg=True, eeg=True, ref_meg=False, exclude=exclude
        )
        raw.pick(indices)
        raw.set_montage(montage)

        raw = raw.resample(SAMPLE_RATE, verbose=False)
        pos, sensor_type = extract_pos_sensor_type(raw.info)
        eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
        pos = normalize_pos(pos, eeg_mask, mag_mask)

        data = raw.get_data()
        start = 0
        end = int(SAMPLE_RATE * 10)
        stride_length = int(SAMPLE_RATE * 10)

        while end <= data.shape[1]:
            seg_data = sensortype_wise_normalize(
                data[:, start:end], eeg_mask, mag_mask, grad_mask
            )
            if accept_segment(seg_data, pos):
                seg_data_path = os.path.join(
                    no_extension_processed_name, f"{len(metadata_list)}.pt"
                )
                seg_data_dict = {
                    "x": torch.from_numpy(seg_data),
                    "pos": torch.from_numpy(pos),
                    "sensor_type": torch.from_numpy(sensor_type),
                    # "mean_std_dict": mean_std_dict,
                }
                accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
                time.sleep(0.02)
                metadata = {
                    "path": seg_data_path,
                    "subject": raw_path.split("/")[-1].split(".")[0],
                    # "split": split,
                    "label": label,
                    "channels": seg_data.shape[0],
                    "size": os.path.getsize(seg_data_path) / (1024.0 * 1024.0),  # MB
                }
                metadata_list.append(metadata)
            start += stride_length
            end += stride_length

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
                os.path.join(root, name) for name in names if name.endswith(".pkl")
            ]

    metadata_list = []
    if multi_process:
        with ProcessPoolExecutor(max_workers=32) as executor:
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
            result = method_preprocess_dict[method](accessor, file_path, processed_path)
            # try:
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
        metadata_list = sorted(metadata_list,key=lambda x:x['path'])
        with open(info_path, "w") as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
