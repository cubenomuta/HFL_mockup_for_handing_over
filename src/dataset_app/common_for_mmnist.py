"""
The following code is copied and modified from https://github.com/Xtra-Computing/NIID-Bench
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

DATA_ROOT = os.environ["DATA_ROOT"]

def create_mmnist_iid(
    labels: np.ndarray,
    num_parties: int,
    classes: List[int] = None,
    list_labels_idxes: Dict[int, List[int]] = None,
    test: bool = False,
):
    # if labels.shape[0] % num_parties:
    #     raise ValueError("Imbalanced classes are not allowed")

    # FMNISTの1クラス分に合わせる
    if test == False: # train data
        if (list_labels_idxes is None) and (classes is None): # フォグ
            samples_per_party = int(6000 / num_parties)
        # else:
        #     samples_per_party = (len(list_labels_idxes[0]) / num_parties)
    else: # test data
        if (list_labels_idxes is None) and (classes is None): # フォグ
            samples_per_party = int(1000 / num_parties)
        # else:
        #     samples_per_party = (len(list_labels_idxes[0]) / num_parties)

    if classes is None and list_labels_idxes is None:
        classes = list(np.unique(labels))
        list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
        # train data
        if test == False: 
            samples_per_party = int(6000 / num_parties)
        # test data
        else:
            samples_per_party = int(1000 / num_parties)
    elif classes is None or list_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_labels_idxes = list_labels_idxes
        for key, _ in list_labels_idxes.items():
            samples_per_party = (len(list_labels_idxes[key]) / num_parties)

    net_dataidx_map = {i: [] for i in range(num_parties)}
    label_indexes = {k: 0 for k in classes}  # 各ラベルの現在位置を追跡
    id = 0

    for k in classes:   
        while len(net_dataidx_map[num_parties-1]) < samples_per_party * (k+1): # 一番最後のpartyがsamples_per_partyに達するまで
            label_idx = list_labels_idxes[k][label_indexes[k]]
            net_dataidx_map[id % num_parties].append(label_idx)
            label_indexes[k] += 1  # 次の位置に移動
            id += 1
            if (label_indexes[k] >= len(list_labels_idxes[k])): # インデックスが後ろまで行ったら前に戻す
                label_indexes[k] = 0
    record_net_data_stats(labels, net_dataidx_map)
    # net_dataidx_mapの型
    return net_dataidx_map


def create_mmnist_noniid(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    num_parties: int,
    num_classes: int,
    classes: List[int] = None,
    list_train_labels_idxes: Dict[int, List[int]] = None,
    list_test_labels_idxes: Dict[int, List[int]] = None,
):
    # if train_labels.shape[0] % (num_parties * num_classes):
    #     print(f"train_labels.shape[0]: {train_labels.shape[0]}, train_labels.shape[0] % (num_parties * num_classes): {train_labels.shape[0] % (num_parties * num_classes)}")
    #     raise ValueError("Imbalanced classes are not allowed")

    if ( # フォグ
        classes is None
        and list_train_labels_idxes is None
        and list_test_labels_idxes is None
    ):
        classes = list(np.unique(train_labels))
        list_train_labels_idxes = {
            k: np.where(train_labels == k)[0].tolist() for k in classes
        }
        list_test_labels_idxes = {
            k: np.where(test_labels == k)[0].tolist() for k in classes
        }
        # train_samples_per_class = int(
        #     train_labels.shape[0] / (num_parties * num_classes)
        # )
        train_samples_per_class = int(
            6000 / num_parties
        )
        # test_samples_per_class = int(test_labels.shape[0] / (num_parties * num_classes))
        test_samples_per_class = int(
            1000 / num_parties
        )
    elif (
        classes is None
        or list_train_labels_idxes is None
        or list_test_labels_idxes is None
    ):
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_train_labels_idxes = list_train_labels_idxes
        num_train = 0
        for val in list_train_labels_idxes.values():
            num_train += len(val)
        list_test_labels_idxes = list_test_labels_idxes
        num_test = 0
        for val in list_test_labels_idxes.values():
            num_test += len(val)
        # train_samples_per_class = int(num_train / (num_parties * num_classes))
        # test_samples_per_class = int(num_test / (num_parties * num_classes))
        train_samples_per_class = int(num_train / (num_parties * 10)) # MMNISTの11クラスだとエラーになるため
        test_samples_per_class = int(num_test / (num_parties * 10))

    train_json_data = {i: [] for i in range(num_parties)}
    test_json_data = {i: [] for i in range(num_parties)}

    class_ids = list(np.random.permutation(classes))
    for id in range(num_parties):
        for i in range(num_classes):
            cls = class_ids.pop()
            for _ in range(train_samples_per_class):
                train_idx = list_train_labels_idxes[cls].pop()
                train_json_data[id].append(train_idx)
            for _ in range(test_samples_per_class):
                test_idx = list_test_labels_idxes[cls].pop()
                test_json_data[id].append(test_idx)
            if len(class_ids) == 0:
                class_ids = list(np.random.permutation(classes))

    record_net_data_stats(train_labels, train_json_data)
    record_net_data_stats(test_labels, test_json_data)

    return train_json_data, test_json_data


def create_mmnist_noniid_dir(
    labels: np.ndarray,
    num_class: int,
    dirichlet_dist: np.ndarray,
    num_parties: int,
    alpha: float,
    seed: int,
    classes: List[int] = None,
    list_labels_idxes: Dict[int, List[int]] = None,
):

    if labels.shape[0] % num_parties:
        raise ValueError("Imbalanced classes are not allowed")

    if classes is None and list_labels_idxes is None:
        print("creating label_idxes ...")
        classes = list(np.unique(labels))
        list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
        num_samples = [int(labels.shape[0] / num_parties) for _ in range(num_parties)]
    elif classes is None or list_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_labels_idxes = list_labels_idxes
        num_sample = 0
        for val in list_labels_idxes.values():
            num_sample += len(val)
        num_samples = [int(num_sample / num_parties) for _ in range(num_parties)]
    alpha = np.asarray(alpha)
    alpha = np.repeat(alpha, num_class)

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=alpha, size=num_parties
        )
        if dirichlet_dist.size != 0:
            if dirichlet_dist.shape != (num_parties, num_class):
                raise ValueError(
                    "The shape of the given dirichlet distribution is no allowed"
                )

    empty_classes = [False if i in classes else True for i in range(num_class)]
    print(empty_classes)

    net_dataidx_map = {i: [] for i in range(num_parties)}
    for id in range(num_parties):
        net_dataidx_map[id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[id].copy(),
            list_label_idxes=list_labels_idxes,
            num_sample=num_samples[id],
            empty_classes=empty_classes,
        )

    record_net_data_stats(labels, net_dataidx_map)
    return net_dataidx_map, dirichlet_dist

def create_consistent_test_data(
    labels: np.ndarray,
    dirichlet_dist: np.ndarray,
    num_parties: int,
    num_test_samples: int = 2000
) -> Dict[int, List[int]]:
    """
    テストデータを各クライアントで一貫した分布に従って生成する関数。
    各クライアントごとに2000サンプルを取得する。
    """
    # クラスごとのインデックスリストを作成
    classes = list(np.unique(labels))
    list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
    print(f"list_labels_idxes: {list_labels_idxes}")

    test_dataidx_map = {i: [] for i in range(num_parties)}
    for id in range(num_parties):
        # 各クライアントに対して2000サンプル取得し、int型に変換して保存
        test_dataidx_map[id] = [int(idx) for idx in sample_with_replacement(
            distribution=dirichlet_dist[id],
            list_label_idxes=list_labels_idxes,
            num_sample=num_test_samples
        )]

    return test_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print(str(net_cls_counts))
    return net_cls_counts

def create_json_data_stats(y_train, net_dataidx_map, save_dir, file_name):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}  # JSONに書き込むためにintに変換
        net_cls_counts[net_i] = tmp
    
    print(str(net_cls_counts))
    
    write_json(net_cls_counts, save_dir, file_name)
    
    return net_cls_counts


def sample_without_replacement(
    distribution: np.ndarray,
    list_label_idxes: Dict[int, List[np.ndarray]],
    num_sample: int,
    empty_classes: List[bool],
) -> Tuple[List[int], List[bool]]:
    distribution = exclude_classes_and_normalize(
        distribution=distribution,
        exclude_dims=empty_classes,
    )
    label_list = []
    for _ in range(num_sample):
        sample_class = np.where(np.random.multinomial(1, distribution) == 1)[0][0]
        label_idx = list_label_idxes[sample_class].pop()
        label_list.append(label_idx)
        if len(list_label_idxes[sample_class]) == 0:
            empty_classes[sample_class] = True
            distribution = exclude_classes_and_normalize(
                distribution=distribution,
                exclude_dims=empty_classes,
            )
    np.random.shuffle(label_list)
    return label_list, empty_classes

def sample_with_replacement(
    distribution: np.ndarray,
    list_label_idxes: Dict[int, List[int]],
    num_sample: int
) -> List[int]:
    """
    指定された分布に従ってデータをサンプルする関数。
    重複を許容してデータを取得する。
    """
    label_list = []
    for _ in range(num_sample):
        sample_class = np.where(np.random.multinomial(1, distribution) == 1)[0][0]
        label_idx = np.random.choice(list_label_idxes[sample_class])  # 重複可能なサンプリング
        label_list.append(label_idx)
    np.random.shuffle(label_list)
    return label_list


def exclude_classes_and_normalize(
    distribution: np.ndarray, exclude_dims: List[bool], eps: float = 1e-5
) -> np.ndarray:
    if np.any(distribution < 0) or (not np.isclose(np.sum(distribution), 1.0)):
        raise ValueError("distribution must sum to 1 and have only positive values.")

    if distribution.size != len(exclude_dims):
        raise ValueError(
            """Length of distribution must be equal
            to the length `exclude_dims`."""
        )
    if eps < 0:
        raise ValueError("""The value of `eps` must be positive and small.""")

    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows
    return distribution


# def record_net_data_stats(y_train, net_dataidx_map):
#     net_cls_counts = {}
#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp
#     print(str(net_cls_counts))
#     return net_cls_counts


def write_json(json_data: Dict[str, List[np.ndarray]], save_dir: str, file_name: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_path = Path(save_dir) / f"{file_name}.json"
    print("writing {}".format(file_name))
    with open(file_path, "w") as outfile:
        json.dump(json_data, outfile)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    dataset = "FashionMNIST"
    X_train, y_train, X_test, y_test = load_cifar10()
    set_seed(1234)
    train_json = create_iid(
        labels=y_train,
        num_parties=1000,
    )
    test_json = create_iid(labels=y_test, num_parties=1000)
    save_dir = "./data/CIFAR10/partitions/iid"
    write_json(train_json, save_dir=save_dir, file_name="train")
    write_json(test_json, save_dir=save_dir, file_name="test")
