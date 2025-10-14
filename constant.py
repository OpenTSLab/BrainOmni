import os
import torch

LOW = 0.1
HIGH = 96
SAMPLE_RATE = 256
SEED = 42
TOKENIZER_SEGMENT_TIME = 2.0
PRETRAIN_DTYPE = torch.bfloat16
DOWNSTREAM_DTYPE = torch.bfloat16

STANDARD_1020 = [
    "Fp1","Fpz","Fp2","AF9","AF7","AF5","AF3","AF1","AFz","AF2","AF4","AF6","AF8","AF10","F9","F7","F5",
    "F3","F1","Fz","F2","F4","F6","F8","F10","FT9","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8",
    "FT10","T9","T7","C5","C3","C1","Cz","C2","C4","C6","T8","T10","TP9","TP7","CP5","CP3","CP1","CPz",
    "CP2","CP4","CP6","TP8","TP10","P9","P7","P5","P3","P1","Pz","P2","P4","P6","P8","P10","PO9","PO7",
    "PO5","PO3","PO1","POz","PO2","PO4","PO6","PO8","PO10","O1","Oz","O2","O9","CB1","CB2","Iz","O10","T3",
    "T5","T4","T6","M1","M2","A1","A2","CFC1","CFC2","CFC3","CFC4","CFC5","CFC6","CFC7","CFC8","CCP1",
    "CCP2","CCP3","CCP4","CCP5","CCP6","CCP7","CCP8","T1","T2","FTT9h","TTP7h","TPP9h","FTT10h","TPP8h",
    "TPP10h","Fp1-F7","F7-T7","T7-P7","P7-O1","Fp2-F8","F8-T8","T8-P8","P8-O2","Fp1-F3","F3-C3","C3-P3",
    "P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2",
]
# eeg:5697       meg: 5261
NEW_DEVICE_DATASET_LIST = ["ds005261-1.0.0", "ds005697-1.0.2"]

PROJECT_ROOT_PATH = "./"  # must end with /
DATA_ROOT_PATH = "./data/"  # must end with /

RAW_PATH = os.path.join(DATA_ROOT_PATH, "raw") + "/"
PROCESSED_PRETRAIN_PATH=os.path.join(DATA_ROOT_PATH, "processed_pretrain") + "/"
EVALUATE_PATH = os.path.join(DATA_ROOT_PATH, "evaluate") + "/"
PROCESSED_EVALUATE_PATH=os.path.join(DATA_ROOT_PATH, "processed_evaluate") + "/"

PRETRAIN_METADATA_PATH = os.path.join(PROJECT_ROOT_PATH, "share", "metadata",'pretrain')
EVALUATE_METADATA_PATH=os.path.join(PROJECT_ROOT_PATH,'share',"metadata",'evaluate')
CUSTOM_MONTAGE_PATH = os.path.join(PROJECT_ROOT_PATH, "share", "custom_montages")
