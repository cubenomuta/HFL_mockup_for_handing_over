from pathlib import Path
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import numpy as np

# save_dir = "./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"
# save_dir = "./data/FashionMNIST/partitions/noniid-label2_part-noniid_0.2/client"
# output_json_file = Path(save_dir) / "all_fogs_clustered_clients.json"

def find_best_threshold(kl_df, fog_id, save_dir):
    # 距離行列を縮約形式に変換
    condensed_distance_matrix = squareform(kl_df.values)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    
    # グリッド探索の範囲
    threshold_values = np.linspace(0.5, 10, num=20)
    silhouette_scores = []
    # 閾値を変えてクラスタリングし、最もシルエットスコアが高い閾値を選ぶ
    highest_score = float('-inf')
    best_threshold = None

    for threshold in threshold_values:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # クラスタ数が2以上のときのみシルエットスコアを計算
        if len(set(labels)) > 2:
            score = silhouette_score(kl_df, labels, metric="precomputed")
            silhouette_scores.append(score)
            
            if score > highest_score:  # シルエットスコアが最大のものを選ぶ
                highest_score = score
                best_threshold = threshold
        else:
            silhouette_scores.append(None)  # クラスタが1つの場合、スコアは無効

    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, silhouette_scores, marker='o', color='b')
    plt.xlabel("Threshold")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score vs. Threshold_{fog_id}")
    plt.grid(True)

    file_name = f"silhouette_score_vs_threshold_{fog_id}.svg"

    # SVG形式で保存
    output_svg_file = Path(save_dir) / file_name
    plt.savefig(output_svg_file, format='svg')
    print(f"シルエットスコアのグラフをSVGファイルとして保存しました: {output_svg_file}")

    print(f"最適なしきい値: {best_threshold}, 最高シルエットスコア: {highest_score}")
    return best_threshold

def cluster_clients(fog_id, save_dir):
    # フォグごとのKLダイバージェンス行列を読み込む
    input_file = Path(save_dir) / f"kl_divergence_matrix_fog_{fog_id}.csv"
    kl_df = pd.read_csv(input_file, index_col=0)

    # 最適な閾値を取得
    best_threshold = find_best_threshold(kl_df, fog_id, save_dir)
    
    # 距離行列から階層クラスタリングを実行
    condensed_distance_matrix = squareform(kl_df.values)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    
    # デンドログラムの作成（オプション）
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=kl_df.index)
    plt.title(f"Fog {fog_id} - Client Clustering Dendrogram")
    plt.xlabel("Client ID")
    plt.ylabel("Distance")
    # plt.show()

    # 最適なしきい値でクラスタの割り当て
    clusters = fcluster(linkage_matrix, best_threshold, criterion='distance')
    
    # クラスタリング結果を整理
    fog_results = {}
    for client_id, cluster_id in zip(kl_df.index, clusters):
        if cluster_id not in fog_results:
            fog_results[cluster_id] = []
        fog_results[cluster_id].append(client_id)

    return {str(fog_id): {str(cluster_id): clients for cluster_id, clients in fog_results.items()}}

def run_hierarchical_clustering(save_dir: str, output_file: str):
    # 複数のフォグのクラスタリング結果をまとめてJSONファイルに保存
    clustered_data = {}
    for i in range(5):
        fog_result = cluster_clients(fog_id=i, save_dir=save_dir)
        clustered_data.update(fog_result)

    with open(output_file, 'w') as f:
        json.dump(clustered_data, f, indent=4)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")