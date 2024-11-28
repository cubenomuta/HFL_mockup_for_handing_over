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
)
from dataset_app.common_for_nih_cxr import (
    create_cxr_iid,
    create_cxr_noniid,
    create_cxr_noniid_dir
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
    choices=["FashionMNIST", "MNIST", "CIFAR10", "CIFAR100", "NIH_CXR"],
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
    # for NIH_CXR
    if args.dataset == "NIH_CXR":
        print("args.dataset is NIH_CXR")
        if partitions == "iid" or partitions == "part-noniid":
            print("create_cxr_iid is called")
            train_json_data = create_cxr_iid(
                labels=train_labels,
                num_parties=num_parties,
                classes=classes,
                list_labels_idxes=list_train_labels_idxes,
                test=False,
            )
            test_json_data = create_cxr_iid(
                labels=test_labels,
                num_parties=num_parties,
                classes=classes,
                list_labels_idxes=list_test_labels_idxes,
                test=True,
            )
        elif partitions >= "noniid-label1" and partitions <= "noniid-label9":
            train_json_data, test_json_data = create_cxr_noniid(
                train_labels=train_labels,
                test_labels=test_labels,
                num_parties=num_parties,
                num_classes=int(partitions[-1]),
                classes=classes,
                list_train_labels_idxes=list_train_labels_idxes,
                list_test_labels_idxes=list_test_labels_idxes,
            )
        elif partitions[:10] == "noniid-dir":
            train_json_data, dirichlet_dist = create_cxr_noniid_dir(
                labels=train_labels,
                num_class=10,
                dirichlet_dist=None,
                num_parties=num_parties,
                alpha=float(partitions[10:]),
                seed=seed,
                classes=classes,
                list_labels_idxes=list_train_labels_idxes,
            )
            # test_json_data = create_consistent_test_data(
            #     labels=test_labels,
            #     dirichlet_dist=dirichlet_dist,
            #     num_parties=num_parties,
            # )
        return train_json_data, test_json_data

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
    print(f"num_fogs: {num_fogs}")
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
    savesave_dir_dir = Path(args.save_dir) / "fog"
    # write_json(fog_train_json_data, save_dir=save_dir, file_name="train_data")
    # write_json(fog_test_json_data, save_dir=save_dir, file_name="test_data")

    client_train_json_data = {}
    client_test_json_data = {}

    for fid in range(num_fogs):
        print(f"creating partitioning for clients connected with fid {fid}")
        client_train_idxs = fog_train_json_data[fid]
        client_test_idxs = fog_test_json_data[fid]
        # Fogに割り当てられたインデックスのラベルユニーク値を取得
        classes = list(np.unique(y_train[client_train_idxs]))
        # {ラベル: インデックス}の辞書を作成
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
        # ここの100は1つのfogに接続するclientの数
        # cidはフォグごとに作られてしまうため、各フォグのクライアントのcidがかぶらないようにキーを更新
        updatedkey_train_json_data = {
            cid + fid * 100: val for cid, val in train_json_data.items()
        }
        updatedkey_test_json_data = {
            cid + fid * 100: val for cid, val in test_json_data.items()
        }
        # updatedkey_train_json_data = {
        #     cid + fid * num_clients: val for cid, val in train_json_data.items()
        # }
        client_train_json_data.update(updatedkey_train_json_data)
        client_test_json_data.update(updatedkey_test_json_data)
    print(f"client_train_json_data: {client_train_json_data.keys()}")
    print(f"client_test_json_data: {client_test_json_data.keys()}")
    save_dir = Path(args.save_dir) / "client"
    write_json(client_train_json_data, save_dir=save_dir, file_name="train_data")
    write_json(client_test_json_data, save_dir=save_dir, file_name="test_data")
    # print(f"{save_dir} create fog json file")
    save_dir = Path(args.save_dir) / "fog"
    write_json(fog_train_json_data, save_dir=save_dir, file_name="train_data")
    write_json(fog_test_json_data, save_dir=save_dir, file_name="test_data")

    if (client_partitions == "part-noniid"):
        
        # クライアントのシャッフルとデータ再配置
        clients_per_fog = [list(range(fid * num_clients, (fid + 1) * num_clients)) for fid in range(num_fogs)]
        # 各フォグに割り当てられているクライアントのリスト　ok
        print(f"clients_per_fog: {clients_per_fog}")

        shuffled_train_clients = []
        shuffled_train_clients_per_fog = []
        num_to_shuffle = 60
        for clients in clients_per_fog:
            # 5分の1のクライアントを抽出してshuffled_clientsに追加
            extracted_clients = []
            extracted_clients = (random.sample(clients, num_to_shuffle))
            shuffled_train_clients.extend(extracted_clients)
            for client in extracted_clients:
                clients.remove(client)
            # ([フォグ0の4/5のクライアント], [フォグ1の4/5のクライアント], ...)
            shuffled_train_clients_per_fog.append(clients)

        # 抽出されたクライアントリストを再度シャッフル
        random.shuffle(shuffled_train_clients)
        print(f"shuffled_train_clients: {shuffled_train_clients}") # ok
        print(f"shuffled_train_clients_per_fog: {shuffled_train_clients_per_fog}") #  ok

        for fid in range(num_fogs):
            for i in range(num_to_shuffle):
                shuffled_train_clients_per_fog[fid].append(shuffled_train_clients.pop())
        print(f"shuffled_train_clients_per_fog after append: {shuffled_train_clients_per_fog}") # ok
        print(f"shuffled_train_clients: {shuffled_train_clients}") # 追記
        
        # shuffled_clients_per_fogは[[フォグ1のcid], [フォグ2のcid], ...]のリスト　をフラットにする
        flattened_train_cid_list = [item for sublist in shuffled_train_clients_per_fog for item in sublist] 
        print(f"flattened_train_cid_list: {flattened_train_cid_list}") # ok

        shuffledkey_client_train_json_data = {} # clientのtrain_jsonを更新するための辞書
        for index, new_key in enumerate(flattened_train_cid_list):
            shuffledkey_client_train_json_data[index] = client_train_json_data[new_key]

        # 更新後のクライアントのトレーニングデータの格納
        # print(f"shuffledkey_client_train_json_data: {shuffledkey_client_train_json_data}")
        # client_train_json_data.update(shuffledkey_client_train_json_data)
        # save_dir = Path(args.save_dir) / "client"
        # write_json(shuffledkey_client_train_json_data, save_dir=save_dir, file_name=f"train_data_2")
        
        # フォグのjsonデータを更新
        fog_updatedkey_train_json_data = {}
        # メインのリストを初期化
        index_list = []
        start = 0
        end = num_fogs * num_clients
        step = num_clients
        # サブリストを作成し、メインのリストに追加
        for i in range(start, end, step):
            sub_list = list(range(i, i + step))
            index_list.append(sub_list)

        print(f"index list: {index_list}")

        for index, cid_list in enumerate(index_list):
            print(f"index: {index}, cid_list: {cid_list}")
            # clientのインデックスをまとめたリストを作成
            combined_values = [shuffledkey_client_train_json_data[cid] for cid in cid_list]
            flattened_combined_values = [item for sublist in combined_values for item in sublist]
            print(f"index: {index}")
            print(f"cid_list: {cid_list}")
            print(f"flattened combined_values: {flattened_combined_values}")
            # fidをキーとして、クライアントのインデックスをまとめたリストを格納
            fog_updatedkey_train_json_data[index] = flattened_combined_values

        # fog_train_json_data.update(fog_updatedkey_train_json_data)
        print(f"fog record net data stats")
        record_net_data_stats(y_train, fog_train_json_data)
        print(f"fog record net data stats updatekey")
        record_net_data_stats(y_train, fog_updatedkey_train_json_data)
        print(f"client record net data stats")
        record_net_data_stats(y_train, client_train_json_data)
        print(f"client record net data stats updatekey")
        record_net_data_stats(y_train, shuffledkey_client_train_json_data)
        save_dir = Path(args.save_dir) / "fog"
        write_json(fog_updatedkey_train_json_data, save_dir=save_dir, file_name="train_data")
        save_dir = Path(args.save_dir) / "client"
        write_json(shuffledkey_client_train_json_data, save_dir=save_dir, file_name="train_data")

        """ テストデータ """
        shuffled_test_clients = []
        shuffled_test_clients_per_fog = []
        num_to_shuffle = num_clients // 5
        for clients in clients_per_fog:
            # 5分の1のクライアントを抽出してshuffled_clientsに追加
            extracted_clients = []
            extracted_clients = (random.sample(clients, num_to_shuffle))
            shuffled_test_clients.extend(extracted_clients)
            for client in extracted_clients:
                clients.remove(client)
            # ([フォグ0の4/5のクライアント], [フォグ1の4/5のクライアント], ...)
            shuffled_test_clients_per_fog.append(clients)

        # 抽出されたクライアントリストを再度シャッフル
        random.shuffle(shuffled_test_clients)
        print(f"shuffled_test_clients: {shuffled_test_clients}") # ok
        print(f"shuffled_test_clients_per_fog: {shuffled_test_clients_per_fog}") #  ok

        for fid in range(num_fogs):
            for i in range(num_to_shuffle):
                shuffled_test_clients_per_fog[fid].append(shuffled_test_clients.pop())
        print(f"shuffled_test_clients_per_fog after append: {shuffled_test_clients_per_fog}") # ok
        print(f"shuffled_test_clients: {shuffled_test_clients}") # 追記
        
        # shuffled_clients_per_fogは[[フォグ1のcid], [フォグ2のcid], ...]のリスト　をフラットにする
        flattened_test_cid_list = [item for sublist in shuffled_test_clients_per_fog for item in sublist] 
        print(f"flattened_test_cid_list: {flattened_test_cid_list}") # ok

        shuffledkey_client_test_json_data = {} # clientのtest_jsonを更新するための辞書
        for index, new_key in enumerate(flattened_test_cid_list):
            shuffledkey_client_test_json_data[index] = client_test_json_data[new_key]

        # 更新後のクライアントのトレーニングデータの格納
        # print(f"shuffledkey_client_test_json_data: {shuffledkey_client_test_json_data}")
        # client_test_json_data.update(shuffledkey_client_test_json_data)
        
        # フォグのjsonデータを更新
        fog_updatedkey_test_json_data = {}
        # メインのリストを初期化
        index_list = []
        start = 0
        end = num_fogs * num_clients
        step = num_clients
        # サブリストを作成し、メインのリストに追加
        for i in range(start, end, step):
            sub_list = list(range(i, i + step))
            index_list.append(sub_list)

        print(f"index list: {index_list}")

        for index, cid_list in enumerate(index_list):
            print(f"index: {index}, cid_list: {cid_list}")
            # clientのインデックスをまとめたリストを作成
            combined_values = [shuffledkey_client_test_json_data[cid] for cid in cid_list]
            flattened_combined_values = [item for sublist in combined_values for item in sublist]
            print(f"index: {index}")
            print(f"cid_list: {cid_list}")
            print(f"flattened combined_values: {flattened_combined_values}")
            # fidをキーとして、クライアントのインデックスをまとめたリストを格納
            fog_updatedkey_test_json_data[index] = flattened_combined_values

        # fog_test_json_data.update(fog_updatedkey_test_json_data)
        print(f"fog test record net data stats")
        record_net_data_stats(y_test, fog_test_json_data)
        print(f"fog test nrecord net data stats updatekey")
        record_net_data_stats(y_test, fog_updatedkey_test_json_data)
        save_dir = Path(args.save_dir) / "fog"
        write_json(fog_updatedkey_test_json_data, save_dir=save_dir, file_name="test_data")
        save_dir = Path(args.save_dir) / "client"
        write_json(shuffledkey_client_test_json_data, save_dir=save_dir, file_name=f"test_data")

        print(f"create json file at {save_dir}")
        print("Fog train json（更新前）")
        record_net_data_stats(y_train, fog_train_json_data)
        print("Fog train json（更新後）")
        record_net_data_stats(y_test, fog_updatedkey_test_json_data)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
