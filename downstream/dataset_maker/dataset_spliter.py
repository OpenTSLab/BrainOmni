import os
import json
import random
from typing import List
import sys
import argparse
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from constant import SEED, EVALUATE_METADATA_PATH

# cross subject split
def n_subject_fold_split(metadata_list: List, n_fold: int, n_valid_fold: int):
    """
    if split in metadata_list.keys(),use the default one
    """
    rand = random.Random(SEED)
    n_fold_subject_split_list = []
    # TUAB and TUEV has train&evel themselves
    if "split" in metadata_list[0].keys() and metadata_list[0]["split"] != None:
        train_val_subject_list = sorted(
            list(set([i["subject"] for i in metadata_list if i["split"] == "train"]))
        )
        rand.shuffle(train_val_subject_list)
        subject_per_fold = len(train_val_subject_list) // n_fold
        for fold_index in range(n_fold):
            if fold_index <= n_fold - n_valid_fold:
                train_subject_list = (
                    train_val_subject_list[: int(fold_index * subject_per_fold)]
                    + train_val_subject_list[
                        int((fold_index + n_valid_fold) * subject_per_fold) :
                    ]
                )
                val_subject_list = train_val_subject_list[
                    int(fold_index * subject_per_fold) : int(
                        (fold_index + n_valid_fold) * subject_per_fold
                    )
                ]
            else:
                val_subject_list = (
                    train_val_subject_list[int(fold_index * subject_per_fold) :]
                    + train_val_subject_list[
                        : int((fold_index - n_fold + n_valid_fold) * subject_per_fold)
                    ]
                )
                train_subject_list = train_val_subject_list[
                    int((fold_index - n_fold + n_valid_fold) * subject_per_fold) : int(
                        fold_index * subject_per_fold
                    )
                ]
            test_subject_list = sorted(
                list(set([i["subject"] for i in metadata_list if i["split"] == "eval"]))
            )
            assert (
                not set(train_subject_list)
                & set(val_subject_list)
                & set(test_subject_list)
            ), "shouldn't have overlap subject"
            n_fold_subject_split_list.append(
                [train_subject_list, val_subject_list, test_subject_list]
            )
    else:
        # sort to make sure result is the same
        subject_list = sorted(list(set([i["subject"] for i in metadata_list])))
        rand.shuffle(subject_list)  # seed has been set before
        subject_per_fold = len(subject_list) // n_fold

        for fold_index in range(n_fold):
            if fold_index <= n_fold - n_valid_fold - 1:
                train_subject_list = (
                    subject_list[: int(fold_index * subject_per_fold)]
                    + subject_list[
                        int((fold_index + n_valid_fold + 1) * subject_per_fold) :
                    ]
                )
                val_subject_list = subject_list[
                    int(fold_index * subject_per_fold) : int(
                        (fold_index + n_valid_fold) * subject_per_fold
                    )
                ]
                test_subject_list = subject_list[
                    int((fold_index + n_valid_fold) * subject_per_fold) : int(
                        (fold_index + n_valid_fold + 1) * subject_per_fold
                    )
                ]
            elif fold_index == n_fold - n_valid_fold:
                test_subject_list = subject_list[: int(subject_per_fold)]
                val_subject_list = subject_list[int(fold_index * subject_per_fold) :]
                train_subject_list = subject_list[
                    int(subject_per_fold) : int(fold_index * subject_per_fold)
                ]
            elif fold_index > n_fold - n_valid_fold:
                val_subject_list = (
                    subject_list[int(fold_index * subject_per_fold) :]
                    + subject_list[
                        : int((n_valid_fold - n_fold + fold_index) * subject_per_fold)
                    ]
                )
                test_subject_list = subject_list[
                    int((n_valid_fold - n_fold + fold_index) * subject_per_fold) : int(
                        (n_valid_fold - n_fold + fold_index + 1) * subject_per_fold
                    )
                ]
                train_subject_list = subject_list[
                    int(
                        (n_valid_fold - n_fold + fold_index + 1) * subject_per_fold
                    ) : int(fold_index * subject_per_fold)
                ]
            assert (
                not set(train_subject_list)
                & set(val_subject_list)
                & set(test_subject_list)
            ), "shouldn't have overlap subject"
            assert set(
                train_subject_list + val_subject_list + test_subject_list
            ) == set(subject_list),'should contain every subject'
            n_fold_subject_split_list.append(
                [train_subject_list, val_subject_list, test_subject_list]
            )

    result = []
    for index in range(len(n_fold_subject_split_list)):
        train_metadata_list = [
            i
            for i in metadata_list
            if i["subject"] in n_fold_subject_split_list[index][0]
        ]
        val_metadata_list = [
            i
            for i in metadata_list
            if i["subject"] in n_fold_subject_split_list[index][1]
        ]
        test_metadata_list = [
            i
            for i in metadata_list
            if i["subject"] in n_fold_subject_split_list[index][2]
        ]
        result.append([train_metadata_list, val_metadata_list, test_metadata_list])

    return result


if __name__ == "__main__":
    n_fold = 5
    n_valid_fold = 1
    for dataset in sorted(os.listdir(EVALUATE_METADATA_PATH)):
        dataset_path = os.path.join(EVALUATE_METADATA_PATH, dataset)
        for method in sorted(os.listdir(dataset_path)):
            method_path = os.path.join(dataset_path, method)
            info_path = os.path.join(method_path, "info.json")
            if not os.path.exists(info_path):
                continue
            # 虽然多线程预处理无法确定执行顺序，但是可以在这里强行排序，确保一致
            print(info_path)
            metadata_list = sorted(json.load(open(info_path, "r")),key=lambda x:x['path'])
            n_fold_split_result = n_subject_fold_split(
                metadata_list, n_fold=n_fold, n_valid_fold=n_valid_fold
            )

            fold_path = os.path.join(method_path, f"{n_fold}_fold")
            os.makedirs(fold_path, exist_ok=True)
            for i in range(len(n_fold_split_result)):
                train_metadata_list = n_fold_split_result[i][0]
                val_metadata_list = n_fold_split_result[i][1]
                test_metadata_list = n_fold_split_result[i][2]
                if os.path.exists(os.path.join(fold_path, f"train_{i}_fold.json")):
                    assert train_metadata_list == json.load(open(os.path.join(fold_path, f"train_{i}_fold.json"))),fold_path # ensure reproducibility
                json.dump(
                    train_metadata_list,
                    open(os.path.join(fold_path, f"train_{i}_fold.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                json.dump(
                    val_metadata_list,
                    open(os.path.join(fold_path, f"val_{i}_fold.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                json.dump(
                    test_metadata_list,
                    open(os.path.join(fold_path, f"test_{i}_fold.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
