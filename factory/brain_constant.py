import os
from constant import CUSTOM_MONTAGE_PATH

MONTAGE_DICT = {
    "ds004902-1.0.5": "brainproducts-RNP-BA-128",
    "ds002778-1.0.5": "biosemi32",
    "ds003775-1.2.1": "biosemi64",
    "ds002721-1.0.3": "standard_1020",
    "ds005420-1.0.0": "standard_1020",
    "ds005620-1.0.0": "standard_1020",
    "ds003555-1.0.1": "standard_1020",
}
CUSTOM_MONTAGE_DICT = {
    "ds003478-1.1.0": os.path.join(CUSTOM_MONTAGE_PATH, "ds003478-1.1.0.tsv"),
    "ds004395-2.0.0": os.path.join(CUSTOM_MONTAGE_PATH, "ds004395-2.0.0.tsv"),
    "ds005697-1.0.2": os.path.join(CUSTOM_MONTAGE_PATH, "ds005697-1.0.2.tsv"),
    "stroke": os.path.join(CUSTOM_MONTAGE_PATH, "stroke.tsv"),
    "ds004148": os.path.join(CUSTOM_MONTAGE_PATH, "ds004148.tsv"),
}
SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}
# raw.rename_channels
RENAME_DICT = {
    "ds002721-1.0.3": {"FP1": "Fp1", "FP2": "Fp2"},
    "ds005420-1.0.0": {
        "EEG Fp1-A1A2": "Fp1",
        "EEG Fp2-A1A2": "Fp2",
        "EEG Fz-A1A2": "Fz",
        "EEG F3-A1A2": "F3",
        "EEG F4-A1A2": "F4",
        "EEG F7-A1A2": "F7",
        "EEG F8-A1A2": "F8",
        "EEG Cz-A1A2": "Cz",
        "EEG C3-A1A2": "C3",
        "EEG C4-A1A2": "C4",
        "EEG T3-A1A2": "T3",
        "EEG T4-A1A2": "T4",
        "EEG Pz-A1A2": "Pz",
        "EEG P3-A1A2": "P3",
        "EEG P4-A1A2": "P4",
        "EEG T5-A1A2": "T5",
        "EEG T6-A1A2": "T6",
        "EEG O1-A1A2": "O1",
        "EEG O2-A1A2": "O2",
    },
}


EXCLUDE_DICT = {
    "MEG-Narrative-Dataset": [
        "EEG057-4302",
        "EEG058-4302",
        "EEG059-4302",
        "EEG060-4302",
        "EEG061-4302",
        "EEG062-4302",
        "EEG063-4302",
        "EEG064-4302",
    ],
    "ds000117-1.0.6": ["Cz2", "Cpz"],
    "ds000247-1.0.2": ["ECG", "VEOG", "HEOG"],
    "ds002778-1.0.5": ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"],
    "ds003478-1.1.0": ["CB1", "CB2", "HEOG", "VEOG"],
    "ds004148": ["Cpz"],
    "ds004186-2.0.0": ["Cz"],
    "ds004395-2.0.0": ["E8", "E25", "E126", "E127", "E129"],
    "ds004998-1.2.2": [
        "EEG002",
        "EEG003",
        "EEG004",
        "EEG005",
        "EEG006",
        "EEG007",
        "EEG008",
    ],
    "ds005420-1.0.0": ["EEG LOC-ROC"],
    "ds005505-1.0.0": ["Cz"],
    "ds005506-1.0.0": ["Cz"],
    "ds005507-1.0.0": ["Cz"],
    "ds005508-1.0.0": ["Cz"],
    "ds005509-1.0.0": ["Cz"],
    "ds005510-1.0.0": ["Cz"],
    "ds005511-1.0.0": ["Cz"],
    "ds005512-1.0.0": ["Cz"],
    "ds005620-1.0.0": ["EMG", "HEOG", "VEOG"],
    "ds005697-1.0.2": ["CB1", "CB2", "Trigger"],
    "ds003555-1.0.1": ["T1", "T2"],
}

HPI_LIST = [
    "camcan1630",
    "ds000117-1.0.6",
    "ds004330-1.0.0",
]

montage = [
    "standard_1005",
    "standard_1020",
    "standard_alphabetic",
    "standard_postfixed",
    "standard_prefixed",
    "standard_primed",
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "biosemi160",
    "biosemi256",
    "easycap-M1",
    "easycap-M10",
    "easycap-M43",
    "EGI_256",
    "GSN-HydroCel-32",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-65_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-129",
    "GSN-HydroCel-256",
    "GSN-HydroCel-257",
    "mgh60",
    "mgh70",
    "artinis-octamon",
    "artinis-brite23",
    "brainproducts-RNP-BA-128",
]
