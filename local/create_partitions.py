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
    record_net_data_stats,
    create_json_data_stats
)
from flwr.common import NDArray
from dataset_app.kmeans_clustering import run_clustering_process

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
    "--client_shuffle_ratio", type=float, required=True, help="shuffle ratio"
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
    if partitions == "iid" or partitions == "part-noniid":
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
    client_shuffle_ratio = args.client_shuffle_ratio
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
    client_test_json_data = {}

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
        updatedkey_test_json_data = {
            cid + fid * 100: val for cid, val in test_json_data.items()
        }
        client_train_json_data.update(updatedkey_train_json_data)
        client_test_json_data.update(updatedkey_test_json_data)
    print(client_train_json_data.keys())
    save_dir = Path(args.save_dir) / "client"
    write_json(client_train_json_data, save_dir=save_dir, file_name="train_data")
    write_json(client_test_json_data, save_dir=save_dir, file_name="test_data")
    create_json_data_stats(y_train, client_train_json_data, save_dir=save_dir, file_name=f"client_train_data_stats")


    if (client_partitions == "part-noniid"):

        clients_per_fog = [list(range(fid * num_clients, (fid + 1) * num_clients)) for fid in range(num_fogs)]
        print(f"clients_per_fog: {clients_per_fog}")

        shuffled_train_clients = [] # 抜き出されたクライアントを格納
        shuffled_train_clients_per_fog = [] # シャッフル後のクライアントを格納(レコードにフォグごとのクライアントリストが存在する)
        # シャッフルするクライアントの割合を決める
        num_to_shuffle = num_clients * client_shuffle_ratio
        num_to_shuffle = int(num_to_shuffle)
        for fid, clients in enumerate(clients_per_fog):
            extracted_clients = []
            extracted_clients = (random.sample(clients, num_to_shuffle))
            shuffled_train_clients.extend(extracted_clients)
            """ フォグ0だけ置き換える場合"""
            # if fid == 0:
            #     for client in extracted_clients:
            #         clients.remove(client) # フォグ0のクライアントを削除
            """ フォグ半分を置き換える場合 """
            # if fid < num_fogs // 2:
            for client in extracted_clients:
                clients.remove(client) # 半分のフォグのクライアントを削除
            shuffled_train_clients_per_fog.append(clients)

        random.shuffle(shuffled_train_clients)
        
        """ フォグ0だけ置き換える場合"""
        # for fid in range(num_fogs):
        #     if fid == 0:
        #         for i in range(num_to_shuffle):
        #             shuffled_train_clients_per_fog[fid].append(shuffled_train_clients.pop())
        """ フォグ半分を置き換える場合 """
        for fid in range(num_fogs):
            # if fid < num_fogs // 2:
            for i in range(num_to_shuffle):
                shuffled_train_clients_per_fog[fid].append(shuffled_train_clients.pop())

        flattened_train_cid_list = [item for sublist in shuffled_train_clients_per_fog for item in sublist] 

        shuffledkey_client_train_json_data = {}
        shuffledkey_client_test_json_data = {}
        for index, new_key in enumerate(flattened_train_cid_list):
            # 新しいファイルでは0から順にindexを振る
            # 元のファイルのクライアントのインデックスを新しいインデックスに変換
            shuffledkey_client_train_json_data[index] = client_train_json_data[new_key]
            shuffledkey_client_test_json_data[index] = client_test_json_data[new_key]

        fog_updatedkey_train_json_data = {}
        fog_updatedkey_test_json_data = {}
        index_list = []
        start = 0
        end = num_fogs * num_clients
        step = num_clients
        for i in range(start, end, step):
            sub_list = list(range(i, i + step))
            index_list.append(sub_list)

        print(f"index list: {index_list}")

        # train data
        for index, cid_list in enumerate(index_list):
            combined_values = [shuffledkey_client_train_json_data[cid] for cid in cid_list]
            flattened_combined_values = [item for sublist in combined_values for item in sublist]
            fog_updatedkey_train_json_data[index] = flattened_combined_values

        # test data
        for index, cid_list in enumerate(index_list):
            combined_values = [shuffledkey_client_test_json_data[cid] for cid in cid_list]
            flattened_combined_values = [item for sublist in combined_values for item in sublist]
            fog_updatedkey_test_json_data[index] = flattened_combined_values

        print("訓練データの統計情報")
        print(f"fog record net data stats")
        record_net_data_stats(y_train, fog_train_json_data)
        print(f"fog record net data stats after update")
        record_net_data_stats(y_train, fog_updatedkey_train_json_data)
        print(f"client record net data stats")
        record_net_data_stats(y_train, client_train_json_data)
        print(f"client record net data stats after update")
        record_net_data_stats(y_train, shuffledkey_client_train_json_data)

        print("テストデータの統計情報")
        print(f"fog record net data stats")
        record_net_data_stats(y_test, fog_test_json_data)
        print(f"fog record net data stats after update")
        record_net_data_stats(y_test, fog_updatedkey_test_json_data)
        print(f"client record net data stats")
        record_net_data_stats(y_test, client_test_json_data)
        print(f"client record net data stats after update")
        record_net_data_stats(y_test, shuffledkey_client_test_json_data)

        save_dir = Path(args.save_dir) / "fog"
        print(f"create test json file at {save_dir}")
        write_json(fog_updatedkey_train_json_data, save_dir=save_dir, file_name="train_data")
        write_json(fog_updatedkey_test_json_data, save_dir=save_dir, file_name="test_data")
        save_dir = Path(args.save_dir) / "client"
        write_json(shuffledkey_client_train_json_data, save_dir=save_dir, file_name=f"train_data")
        write_json(shuffledkey_client_test_json_data, save_dir=save_dir, file_name=f"test_data")
        create_json_data_stats(y_train, shuffledkey_client_train_json_data, save_dir=save_dir, file_name=f"client_train_data_stats")

    shuffledkey_client_train_json_data = {}
    shuffledkey_client_test_json_data = {}
    shuffledkey_client_train_json_data.update(client_train_json_data)
    shuffledkey_client_test_json_data.update(client_test_json_data)

    print("フォグ訓練データの統計情報")
    print(f"fog record net data stats")
    record_net_data_stats(y_train, fog_train_json_data)
    # print(f"fog record net data stats after update")
    # record_net_data_stats(y_train, fog_updatedkey_train_json_data)
    print(f"client record net data stats")
    record_net_data_stats(y_train, client_train_json_data)
    # print(f"client record net data stats after update")
    # record_net_data_stats(y_train, shuffledkey_client_train_json_data)

    # implement clustering
    save_dir = Path(args.save_dir) / "client"
    input_file_path = save_dir / "client_train_data_stats.json"
    output_file_path = save_dir / "clustered_client_list.json"
    run_clustering_process(input_file=input_file_path, output_file=output_file_path, cluster_count=5)

    # make cluster train_data.json
    with open(output_file_path, 'r') as f:
        clustered_client_list = json.load(f)
    
    cluster_train_json_data = {}
    cluster_test_json_data = {}

    for fid, clusters in clustered_client_list.items():
        cluster_train_json_data[fid] = {}
        cluster_test_json_data[fid] = {}
        for clsid, cid_list in clusters.items():
            cluster_train_json_data[fid][clsid] = []
            cluster_test_json_data[fid][clsid] = []
            for cid in cid_list:
                cluster_train_json_data[fid][clsid].extend(shuffledkey_client_train_json_data[cid])
                cluster_test_json_data[fid][clsid].extend(shuffledkey_client_test_json_data[cid])

    save_dir = Path(args.save_dir) / "cluster"
    write_json(cluster_train_json_data, save_dir=save_dir, file_name="train_data")
    write_json(cluster_test_json_data, save_dir=save_dir, file_name="test_data")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
