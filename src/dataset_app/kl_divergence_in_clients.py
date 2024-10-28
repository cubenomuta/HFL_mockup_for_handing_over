import json
import numpy as np
from sklearn.cluster import KMeans

import json
from pathlib import Path
import numpy as np
from scipy.stats import entropy
import pandas as pd

# save_dir="./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"
# save_dir = "./data/FashionMNIST/partitions/noniid-label2_part-noniid_0.2/client"

def load_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json_file(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        def convert(o):
            if isinstance(o, np.int32):
                return int(o)
            raise TypeError

        json.dump(data, f, default=convert) 


def save_kl_divergence(save_dir: str):
    # 入力ファイルと出力ディレクトリのパス
    file_dir = Path(save_dir) / "client_train_data_stats.json"
    client_stats = load_json_file(file_dir)

    # クラスIDの範囲
    class_ids = range(10)
    fog_size = 100

    # クライアントごとの確率分布を100個単位で処理
    for fog_start in range(0, len(client_stats), fog_size):
        fog_clients = list(client_stats.keys())[fog_start:fog_start + fog_size]
        client_distributions = {}

        for client_id in fog_clients:
            client_data = client_stats[client_id]
            total_samples = sum(client_data.values())
            client_distribution = np.array([client_data.get(str(class_id), 0) / total_samples for class_id in class_ids])
            client_distribution += 1e-10  # ゼロの確率を避けるために小さな値を加える
            client_distributions[client_id] = client_distribution

        # KLダイバージェンス行列の作成
        clients = list(client_distributions.keys())
        kl_matrix = np.zeros((len(clients), len(clients)))

        for i, client_id1 in enumerate(clients):
            for j, client_id2 in enumerate(clients):
                if i != j:
                    kl_divergence = (entropy(client_distributions[client_id1], client_distributions[client_id2]) + 
                                     entropy(client_distributions[client_id2], client_distributions[client_id1])) / 2
                    kl_matrix[i, j] = kl_divergence

        # DataFrameとして保存
        kl_df = pd.DataFrame(kl_matrix, index=clients, columns=clients)
        fog_id = fog_start // fog_size
        output_file = Path(save_dir) / f"kl_divergence_matrix_fog_{fog_id}.csv"
        kl_df.to_csv(output_file)
        print(f"フォグ {fog_id} のKLダイバージェンス行列を保存しました: {output_file}")
    

# if __name__ == "__main__":
#     main()