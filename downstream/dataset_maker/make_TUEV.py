import os
import mne
import numpy as np
import json
import torch
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
from scipy.signal import resample
import logging
from factory.utils import (
    extract_pos_sensor_type,
    get_sensor_type_mask,
    normalize_pos,
    sensortype_wise_normalize,
    accept_segment,
)

DATASET_NAME = "TUEV"
TUEV_PATH = os.path.join(EVALUATE_PATH, "tuh_eeg_events", "edf")

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


def BuildEvents(signals, times, EventData, fs):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    features = []
    labels = []
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features.append(
            signals[:, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)]
        )
        labels.append(int(EventData[i, 3]))
    return features, labels


def preprocess_BrainOmni(
    accessor: DataAccessor, edf_path: str, split: str, processed_path: str
):
    logger.info(f"Processing file {edf_path}")
    no_extension_processed_name = os.path.join(
        processed_path, edf_path.rsplit("/", 1)[-1].split(".")[0]
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

    if len(raw.ch_names) != 21:
        return []

    # preprocess
    raw = raw.notch_filter(60.0, verbose=False)
    raw = raw.filter(LOW, HIGH, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)

    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask)

    _, times = raw[:]
    data = raw.get_data()
    rec_file = edf_path.rsplit(".", 1)[0] + ".rec"
    event_data = np.genfromtxt(rec_file, delimiter=",")
    data, labels = BuildEvents(data, times, event_data, SAMPLE_RATE)
    # each data is channel 5s*samplerate    label

    # save
    os.makedirs(no_extension_processed_name, exist_ok=True)
    metadata_list = []
    for i in range(len(data)):
        seg_data = sensortype_wise_normalize(
            data[i], eeg_mask, mag_mask, grad_mask
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
        metadata_list.append(
            {
                "path": seg_data_path,
                "subject": edf_path.split("/")[-1].split("_")[0],
                "split": split,
                "label": labels[i] - 1,
                "channels": seg_data.shape[0],
            }
        )
    return metadata_list


def process(
    accessor: DataAccessor, split: str, method: str, logger, multi_process=True
):
    assert split in ["train", "eval"]
    method_preprocess_dict = {
        "BrainOmni": preprocess_BrainOmni,
    }
    # import pdb;pdb.set_trace()
    processed_path = os.path.join(PROCESSED_EVALUATE_PATH, DATASET_NAME, method)
    os.makedirs(processed_path, exist_ok=True)
    root_path = os.path.join(TUEV_PATH, split)
    files = []
    for root, dir, names in os.walk(root_path):
        if len(names) > 0:
            files += [
                os.path.join(root, i) for i in names if i.rsplit(".", 1)[-1] == "edf"
            ]
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
                        processed_path,
                    )
                )
            for future in as_completed(futures):
                # try:
                metadata_list += future.result()
                # except Exception as e:
                #     logger.info(f"An error occurred:{e}")
    else:
        for i in files:
            result = method_preprocess_dict[method](accessor, i, split, processed_path)
            # try:
            metadata_list += result
            # except Exception as e:
            #     logger.info(f"An error occurred:{e}")

    return metadata_list


if __name__ == "__main__":
    accessor = DataAccessor(read_only=False)
    logger = get_logger()
    for method in ["BrainOmni"]:
        logger.info(f"start processing {method}")
        metadata_list = []
        os.makedirs(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method), exist_ok=True
        )
        if os.path.exists(os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method, "info.json")):
            continue
        for split in ["train", "eval"]:
            metadata_list += process(
                accessor, split, method, logger, multi_process=True
            )
        metadata_list = sorted(metadata_list,key=lambda x:x['path'])
        with open(
            os.path.join(EVALUATE_METADATA_PATH, DATASET_NAME, method, "info.json"),
            "w",
        ) as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
