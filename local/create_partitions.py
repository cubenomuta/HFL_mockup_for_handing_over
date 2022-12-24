import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from dataset_app.common import (
    create_iid,
    create_noniid,
    create_noniid_dir,
    load_numpy_dataset,
    write_json,
)
from flwr.common import NDArray

parser = argparse.ArgumentParser("Create dataset partitions for fogs and clients")
parser.add_argument("--num_fogs", type=int, required=True, help="The number of fogs")
parser.add_argument(
    "--num_clients", type=int, required=True, help="The number of clients per fog"
)
parser.add_argument(
    "--fog_partitions", type=str, required=True, help="dataset partitions"
)
parser.add_argument(
    "--client_partitions", type=str, required=True, help="dataset partitions"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["FashionMNIST", "MNIST", "CIFAR10", "CIFAR100"],
    help="dataset name",
)
parser.add_argument("--save_dir", type=str, required=True, help="save directory")
parser.add_argument("--seed", type=int, required=True, help="random seed")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def partitioning(
    train_labels: NDArray,
    test_labels: NDArray,
    num_parties: int,
    partitions: str,
    seed: int,
    classes: Optional[List[int]] = None,
    list_train_labels_idxes: Optional[Dict[int, List[int]]] = None,
    list_test_labels_idxes: Optional[Dict[int, List[int]]] = None,
):
    if partitions == "iid":
        train_json_data = create_iid(
            labels=train_labels,
            num_parties=num_parties,
            classes=classes,
            list_labels_idxes=list_train_labels_idxes,
        )
        test_json_data = create_iid(
            labels=test_labels,
            num_parties=num_parties,
            classes=classes,
            list_labels_idxes=list_test_labels_idxes,
        )
    elif partitions >= "noniid-label1" and partitions <= "noniid-label9":
        train_json_data, test_json_data = create_noniid(
            train_labels=train_labels,
            test_labels=test_labels,
            num_parties=num_parties,
            num_classes=int(partitions[-1]),
            classes=classes,
            list_train_labels_idxes=list_train_labels_idxes,
            list_test_labels_idxes=list_test_labels_idxes,
        )
    elif partitions[:10] == "noniid-dir":
        train_json_data, dirichlet_dist = create_noniid_dir(
            labels=train_labels,
            num_class=10,
            dirichlet_dist=None,
            num_parties=num_parties,
            alpha=float(partitions[10:]),
            seed=seed,
            classes=classes,
            list_labels_idxes=list_train_labels_idxes,
        )
        test_json_data, dirichlet_dist = create_noniid_dir(
            labels=test_labels,
            num_class=10,
            dirichlet_dist=None,
            num_parties=num_parties,
            alpha=float(partitions[10:]),
            seed=seed,
            classes=classes,
            list_labels_idxes=list_test_labels_idxes,
        )
    return train_json_data, test_json_data


def main(args):
    print(args)
    set_seed(args.seed)

    params_config = vars(args)

    # FL configuration
    num_fogs = args.num_fogs
    num_clients = args.num_clients
    dataset = args.dataset
    fog_partitions = args.fog_partitions
    client_partitions = args.client_partitions
    seed = args.seed

    _, y_train, _, y_test = load_numpy_dataset(dataset_name=dataset)

    fog_train_json_data, fog_test_json_data = partitioning(
        train_labels=y_train,
        test_labels=y_test,
        num_parties=num_fogs,
        partitions=fog_partitions,
        seed=seed,
    )
    save_dir = Path(args.save_dir) / "fog"
    write_json(fog_train_json_data, save_dir=save_dir, file_name="train_data")
    write_json(fog_test_json_data, save_dir=save_dir, file_name="test_data")

    client_train_json_data = {}

    for fid in range(num_fogs):
        print(f"creating partitioning for clients connected with fid {fid}")
        client_train_idxs = fog_train_json_data[fid]
        client_test_idxs = fog_test_json_data[fid]
        classes = list(np.unique(y_train[client_train_idxs]))
        list_train_labels_idxes = {k: [] for k in classes}
        for idx in client_train_idxs:
            list_train_labels_idxes[y_train[idx]].append(idx)
        list_test_labels_idxes = {k: [] for k in classes}
        for idx in client_test_idxs:
            list_test_labels_idxes[y_test[idx]].append(idx)
        train_json_data, test_json_data = partitioning(
            train_labels=y_train,
            test_labels=y_test,
            num_parties=num_clients,
            partitions=client_partitions,
            seed=seed,
            classes=classes,
            list_train_labels_idxes=list_train_labels_idxes,
            list_test_labels_idxes=list_test_labels_idxes,
        )
        updatedkey_train_json_data = {
            cid + fid * 100: val for cid, val in train_json_data.items()
        }
        client_train_json_data.update(updatedkey_train_json_data)
    print(client_train_json_data.keys())
    save_dir = Path(args.save_dir) / "client"
    write_json(client_train_json_data, save_dir=save_dir, file_name="train_data")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
