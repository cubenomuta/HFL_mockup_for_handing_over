import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path
import matplotlib.pyplot as plt

def load_json_file(file_path: str) -> dict:
    """JSONファイルを読み込む関数"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json_file(data: dict, file_path: str) -> None:
    """クラスタリング結果をJSONファイルとして保存する関数"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def determine_optimal_parameters(X, fid, save_dir, eps_values, min_samples_values):
    """DBSCANの最適なパラメータを決定し、評価指標をプロットする関数"""
    print("run determine_optimal_parameters")
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    # グリッド探索で各評価指標を計算
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            clusters = dbscan.fit_predict(X)
            
            # クラスタ数の確認（ノイズ除く）
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if num_clusters < 2:
                continue  # クラスタが2未満の場合スコアを計算しない
            
            # 評価指標を計算してリストに追加
            if not silhouette_scores:
                print(f"Warning: No valid clusters found for fid {fid}. Skipping plot.")
                return  # スコアがなければ関数を終了

            silhouette_scores.append((eps, min_samples, silhouette_score(X, clusters)))
            calinski_harabasz_scores.append((eps, min_samples, calinski_harabasz_score(X, clusters)))
            davies_bouldin_scores.append((eps, min_samples, davies_bouldin_score(X, clusters)))

    # プロットするためのヘルパー関数
    def plot_score(scores, title, ylabel, filename):
        eps_vals, min_sample_vals, score_vals = zip(*scores)
        plt.figure()
        plt.scatter(eps_vals, min_sample_vals, c=score_vals, cmap='viridis', marker='o')
        plt.colorbar(label=ylabel)
        plt.xlabel('Eps')
        plt.ylabel('Min Samples')
        plt.title(title)
        print(f"評価指標のプロットを '{save_dir / filename}' に保存しました。")
        plt.savefig(save_dir / filename)
        plt.close()

    # 各評価指標のプロット
    plot_score(silhouette_scores, 'Silhouette Score for Optimal Parameters', 'Silhouette Score', f'silhouette_score_plot_{fid}.svg')
    plot_score(calinski_harabasz_scores, 'Calinski-Harabasz Score for Optimal Parameters', 'Calinski-Harabasz Score', f'calinski_harabasz_score_plot_{fid}.svg')
    plot_score(davies_bouldin_scores, 'Davies-Bouldin Score for Optimal Parameters', 'Davies-Bouldin Score', f'davies_bouldin_score_plot_{fid}.svg')
    print(f"save_dirに保存しました: {save_dir}")

def dbscan_clustering(data: dict, save_dir, num_fogs: int, num_clients: int, num_classes: int, eps_values, min_samples_values) -> dict:
    """各フォグサーバ範囲でDBSCANクラスタリングを実行し、評価指標をプロット"""
    step = num_clients
    clustered_data = {}

    for i in range(0, num_fogs * num_clients, step):
        print(f"\nフォグサーバ範囲 {i}~{i + step - 1}")
        fid = str(i // step)  # フォグIDを文字列に変換
        
        # 現在のフォグサーバ範囲のデータを取得
        fog_data = {k: v for k, v in data.items() if i <= int(k) < i + step}
        
        # データのラベルとその数を配列に変換
        X = []
        for k, v in fog_data.items():
            row = [0] * num_classes  # クラス数に合わせて初期化
            for label, count in v.items():
                row[int(label)] = int(count)  # クラス番号のインデックスにカウントを設定
            X.append(row)
        
        # 配列に変換
        X = np.array(X, dtype=float)
        
        # パラメータの評価とプロット
        determine_optimal_parameters(X, fid, save_dir, eps_values, min_samples_values)

    return clustered_data

def run_dbscan_clustering(save_dir: str, output_file: str, num_fogs: int, num_clients: int, num_classes: int): 
    """DBSCANクラスタリングプロセスを実行し、評価指標をプロットする関数"""

    eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]  # 距離のしきい値
    min_samples_values = [3, 5, 7, 10]  # 最小サンプル数

    input_file = Path(save_dir) / "client_train_data_stats.json"
    data = load_json_file(input_file)
    print("元のデータ:", list(data.items())[:5])  # 最初の5件を表示
    
    # クラスタリングを実行
    clustered_data = dbscan_clustering(data, save_dir, num_fogs, num_clients, num_classes, eps_values, min_samples_values)
    
    # クラスタリング結果を保存
    save_json_file(clustered_data, output_file)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")

def main():
    save_dir = "./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"
    input_file_path = Path(save_dir) / "client_train_data_stats.json"
    output_file_path = Path(save_dir) / "clustered_client_list.json"
    
    # DBSCANクラスタリングのパラメータ範囲

    
    run_dbscan_clustering(save_dir=save_dir, output_file=output_file_path, num_fogs=1, num_clients=100, num_classes=10, eps_values=eps_values, min_samples_values=min_samples_values)

if __name__ == "__main__":
    main()
