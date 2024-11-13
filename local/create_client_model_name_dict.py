import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

# 必要なインポート
import numpy as np
import torch

parser = argparse.ArgumentParser("Flower hierarchical federated learning simulation")

parser.add_argument(
    "--num_fogs",
    type=int,
    required=False,
    default=4,
    help="FL config: number of fogs",
)
parser.add_argument(
    "--num_clients",
    type=int,
    required=False,
    default=4,
    help="FL config: number of clients",
)
parser.add_argument(
    "--client_models",
    type=str,
    required=True,
    help="FL config: client-side model name",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Directory path for saving results",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)

# シードの設定
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

# クライアントにモデルを割り当てる関数
def assign_models_to_clients(num_fogs: int, num_clients: int, models_list: List[str]) -> Dict[str, str]:
    client_model_assignment = {}
    clients_per_fog = num_clients * num_fogs
    num_models = len(models_list)

    for fog_id in range(num_fogs):
        # 各フォグのクライアントIDの範囲を定義
        client_ids = list(range(fog_id * num_clients, (fog_id + 1) * num_clients))
        
        # 各フォグ内のモデル割り当てをシャッフルしながら決定
        models_for_clients = (models_list * (num_clients // num_models)) + models_list[:num_clients % num_models]
        random.shuffle(models_for_clients)

        # 各クライアントにモデルを割り当て、フォグ内でのモデル数をカウント
        model_counts = {model: 0 for model in models_list}
        for cid, model_name in zip(client_ids, models_for_clients):
            client_model_assignment[str(cid)] = model_name
            model_counts[model_name] += 1  # 各モデルの割り当て数をカウント

        # 各フォグのモデル割り当て結果を出力
        print(f"Fog {fog_id + 1}: Model allocation counts {model_counts}")

    return client_model_assignment

# メイン関数
def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    # モデルのリストを作成
    models_list = args.client_models.split(",")

    # クライアントごとのモデル割り当てを生成
    client_model_assignment = assign_models_to_clients(args.num_fogs, args.num_clients, models_list)

    # 保存ディレクトリが存在しない場合は作成
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # JSONファイルとして保存
    json_path = os.path.join(args.save_dir, "client_models_name_dict.json")
    with open(json_path, "w") as json_file:
        json.dump(client_model_assignment, json_file, indent=4)

    print(f"Client model assignment saved to {json_path}")

if __name__ == "__main__":
    main()