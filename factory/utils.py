import os
import mne
import time
import random
import numpy as np
import torch
from constant import SAMPLE_RATE, LOW, HIGH, NEW_DEVICE_DATASET_LIST
from factory.brain_constant import (
    EXCLUDE_DICT,
    RENAME_DICT,
    HPI_LIST,
    MONTAGE_DICT,
    CUSTOM_MONTAGE_DICT,
    SENSOR_TYPE_DICT,
)
from accessor import DataAccessor, write_torch_warpper

def filter_channel(raw, dataset: str):
    exclude = []
    if dataset in EXCLUDE_DICT.keys():
        exclude = EXCLUDE_DICT[dataset]

    for i in ["HEO", "VEO", "EKG", "EMG"]:
        if i in raw.info.ch_names and i not in exclude:
            exclude.append(i)

    if dataset == "Omega":
        indices = mne.pick_types(
            raw.info, meg=True, eeg=False, ref_meg=False, exclude=exclude
        )
    else:
        indices = mne.pick_types(
            raw.info, meg=True, eeg=True, ref_meg=False, exclude=exclude
        )
    raw.pick(indices)
    return raw


def rename_channel(raw, dataset: str):
    if dataset in RENAME_DICT.keys():
        raw.rename_channels(RENAME_DICT[dataset])
    return raw


def set_montage(raw, dataset: str):
    if dataset not in MONTAGE_DICT.keys() and dataset not in CUSTOM_MONTAGE_DICT.keys():
        return raw
    if dataset in CUSTOM_MONTAGE_DICT.keys():
        montage = mne.channels.read_custom_montage(CUSTOM_MONTAGE_DICT[dataset])
        raw.set_montage(montage)
        return raw
    montage = mne.channels.make_standard_montage(MONTAGE_DICT[dataset])
    raw.set_montage(montage)
    return raw


def extract_pos_sensor_type(info):
    """
    kind = {1(FIFFV_MEG_CH), 2(FIFFV_EEG_CH)}
    coil_type = {
        1(FIFFV_COIL_EEG),
        4001(FIFFV_COIL_MAGNES_MAG),
        3012(FIFFV_COIL_VV_PLANAR_T1),
        201609,                          #(AXIAL_GRAD)
        5001,                            #(AXIAL_GRAD)
        3022(FIFFV_COIL_VV_MAG_T1),
        3024(FIFFV_COIL_VV_MAG_T3),
        6001(FIFFV_COIL_KIT_GRAD),
    }
    """
    pos = []
    sensor_type = []
    # kind_dict = {1: "meg", 2: "eeg"}
    for i in info["chs"]:
        kind = int(i["kind"])
        assert kind in [1, 2], f"Unknown sensor kind:{i['kind']}"
        coil_type = str(i["coil_type"])
        # eeg
        if kind == 2:
            pos.append(np.hstack([i["loc"][:3], np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])
        # meg
        else:
            xyz = i["loc"][:3]
            dir_idx = 3
            if "PLANAR" in coil_type:
                dir_idx = 1
            dir = i["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos.append(np.hstack([xyz, dir]))

            if "MAG" in coil_type:
                sensor_type.append(SENSOR_TYPE_DICT["MAG"])
            else:
                sensor_type.append(SENSOR_TYPE_DICT["GRAD"])

    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.array(sensor_type).astype(np.int32)

    return pos, sensor_type


def get_sensor_type_mask(sensor_type: np.ndarray):
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]
    meg_mask = mag_mask | grad_mask
    return eeg_mask, mag_mask, grad_mask, meg_mask


def _auto_detect_bad_channels(raw_data: mne.io.Raw, threshold: int = 10):
    spectrum = raw_data.compute_psd(tmax=1000000, average="mean", verbose=False)  # fmax
    data = spectrum.data + 1e-16
    ch_names = np.array(spectrum.ch_names)
    log_data = np.log(data)
    # Euclidean distance between channel pairs
    distances = np.linalg.norm(log_data[:, None, :] - log_data[None, :, :], axis=2)
    mean_distances = np.mean(distances, axis=1)

    # Use IQR (interquartile range) to identify outliers
    Q1 = np.percentile(mean_distances, 25)
    Q3 = np.percentile(mean_distances, 75)
    IQR = Q3 - Q1
    threshold_upper = Q3 + threshold * IQR
    threshold_lower = Q1 - threshold * IQR

    outlier_indices = np.where(
        (mean_distances > threshold_upper) | (mean_distances < threshold_lower)
    )[0]
    bad_channels = ch_names[outlier_indices].tolist()

    return bad_channels


def auto_detect_bad_channels(raw: mne.io.Raw, eeg_mask, mag_mask, grad_mask):
    bad_channels = []
    if eeg_mask.any():
        bad_channels += _auto_detect_bad_channels(
            raw.copy().pick(picks=mne.pick_types(raw.info, eeg=True))
        )
    return bad_channels


def filter_resample_preprocess(raw, dataset: str):
    notch_freqs = [50, 60]
    if len(notch_freqs) > 0:
        raw = raw.notch_filter(freqs=notch_freqs, verbose=False)
    if dataset in HPI_LIST:
        raw = mne.chpi.filter_chpi(raw, include_line=False, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False, n_jobs=2)
    raw = raw.filter(LOW, HIGH, verbose=False)
    return raw


def normalize_pos(pos: np.ndarray, eeg_mask, meg_mask):
    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        pos[eeg_mask, :3] /= eeg_scale
    if meg_mask.any():
        meg_mean = np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] -= meg_mean
        meg_scale = np.sqrt(3 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
        pos[meg_mask, :3] /= meg_scale
    return pos


def sensortype_wise_normalize(
    _data: np.ndarray, eeg_mask, mag_mask, grad_mask
):
    data = _data.copy()
    if eeg_mask.any():
        eeg_data = data[eeg_mask, :]
        eeg_mean = np.mean(eeg_data, axis=0, keepdims=True)
        eeg_data = eeg_data - eeg_mean
        eeg_std = np.std(eeg_data) + 1.0e-5
        data[eeg_mask, :] = eeg_data / (eeg_std)

    if mag_mask.any():
        mag_data = data[mag_mask, :]
        mag_mean = np.mean(mag_data, axis=0, keepdims=True)
        mag_data = mag_data - mag_mean
        mag_std = np.std(mag_data) + 1.0e-13
        data[mag_mask, :] = mag_data / mag_std

    if grad_mask.any():
        grad_data = data[grad_mask, :]
        grad_mean = np.mean(grad_data, axis=0, keepdims=True)
        grad_data = grad_data - grad_mean
        grad_std = np.std(grad_data) + 1.0e-13
        data[grad_mask, :] = grad_data / grad_std

    return data.astype(np.float32)


def accept_segment(seg_data: np.ndarray, pos: np.ndarray):
    bad = (np.isnan(seg_data).any()) | (np.isnan(pos).any())
    return ~bad


def split_to_segments_save(
    accessor: DataAccessor,
    data: np.ndarray,
    pos: np.ndarray,
    sensor_type: np.ndarray,
    eeg_mask: np.ndarray,
    mag_mask: np.ndarray,
    grad_mask: np.ndarray,
    meg_mask: np.ndarray,
    path: str,
    dataset: str,
    ready_path: str,
    TIME:int,
    STRIDE:int
):
    segments_metadata = []
    start = 0
    end = int(start + TIME * SAMPLE_RATE)
    stride_length = int(STRIDE * SAMPLE_RATE)
    dataset_name = accessor.get_dataset_folder_name(path)
    dataset_to_file_path = f"{dataset_name}/" + path.split(f"/{dataset_name}/")[-1]
    brain_file_folder_path = os.path.join(ready_path, dataset_to_file_path).rsplit(
        ".", 1
    )[0]
    accessor.mkdir(brain_file_folder_path)

    while end < data.shape[1]:
        seg_data, _ = sensortype_wise_normalize(
            data[:, start:end], eeg_mask, mag_mask, grad_mask, meg_mask
        )
        if accept_segment(seg_data, pos):
            seg_data_path = os.path.join(
                brain_file_folder_path, f"{len(segments_metadata)}_data.pt"
            )
            seg_data_dict = {
                "x": torch.from_numpy(seg_data),
                "pos": torch.from_numpy(pos),
                "sensor_type": torch.from_numpy(sensor_type),
            }
            accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
            time.sleep(0.01)
            metadata = {
                "dataset": dataset,
                "path": seg_data_path,
                "channels": seg_data.shape[0],
                "is_eeg": bool((sensor_type == SENSOR_TYPE_DICT["EEG"]).all()),
                "is_meg": bool(
                    (
                        (sensor_type == SENSOR_TYPE_DICT["MAG"])
                        | (sensor_type == SENSOR_TYPE_DICT["GRAD"])
                    ).all()
                ),
            }
            segments_metadata.append(metadata)
        start += stride_length
        end += stride_length
    return segments_metadata


def split_pretrain_metadata(data):
    new_device_dataset_dict = {}
    for dataset in NEW_DEVICE_DATASET_LIST:
        new_device_dataset_dict[dataset] = [i for i in data if i["dataset"] == dataset]
    data = [i for i in data if i["dataset"] not in NEW_DEVICE_DATASET_LIST]
    random.shuffle(data)
    N = len(data)
    train = data[: int(N * 0.85)]
    val = data[int(N * 0.85) : int(N * 0.95)]
    test = data[int(N * 0.95) :]
    return train, val, test, new_device_dataset_dict

def process(
    accessor: DataAccessor,
    path: str,
    dataset: str,
    ready_path: str,
    TIME:int,
    STRIDE:int
):
    raw = accessor.read_brain_file(path)
    raw = rename_channel(raw, dataset)
    raw = filter_channel(raw, dataset)
    raw = set_montage(raw, dataset)

    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, meg_mask)

    raw = filter_resample_preprocess(raw, dataset)

    bad_channels = auto_detect_bad_channels(raw, eeg_mask, mag_mask, grad_mask)
    if len(bad_channels) > 0:
        raw.info["bads"] += bad_channels
    raw.interpolate_bads(
        reset_bads=True, mode="accurate", origin=(0.0, 0.0, 0.04), verbose=False
    )

    data = raw.get_data()
    del raw
    # data float64 everything not normalized
    segments_metadata = split_to_segments_save(
        accessor,
        data,
        pos,
        sensor_type,
        eeg_mask,
        mag_mask,
        grad_mask,
        meg_mask,
        path,
        dataset,
        ready_path,
        TIME,
        STRIDE
    )
    return segments_metadata, path