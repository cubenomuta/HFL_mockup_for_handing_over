from pathlib import Path
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np


def find_best_threshold(kl_df, fog_id, save_dir):
    """最適なクラスタリング閾値を見つけ、評価指標を個別にプロットして保存"""
    # 距離行列を縮約形式に変換
    condensed_distance_matrix = squareform(kl_df.values)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    
    # # 閾値でグリッド探索
    # threshold_values = np.linspace(0.5, 10, num=20)
    # highest_silhouette = float('-inf')
    # best_threshold = None
    # for threshold in threshold_values:
    #     labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
    #     # クラスタ数が2以上のときのみスコアを計算
    #     if len(set(labels)) > 2:
    #         silhouette = silhouette_score(kl_df, labels, metric="precomputed")
    #         calinski_harabasz = calinski_harabasz_score(kl_df, labels)
    #         davies_bouldin = davies_bouldin_score(kl_df, labels)
            
    #         silhouette_scores.append(silhouette)
    #         calinski_harabasz_scores.append(calinski_harabasz)
    #         davies_bouldin_scores.append(davies_bouldin)

    #         if silhouette > highest_silhouette:
    #             highest_silhouette = silhouette
    #             best_threshold = threshold
    #     else:
    #         silhouette_scores.append(None)
    #         calinski_harabasz_scores.append(None)
    #         davies_bouldin_scores.append(None)

    # クラスタ数でグリッド探索
    max_clusters = 10
    k_values = range(2, max_clusters + 1)
    highest_silhouette = float('-inf')
    best_k = None

    for k in k_values:
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        
        # クラスタ数が2以上のときのみスコアを計算
        if len(set(labels)) > 1:
            silhouette = silhouette_score(kl_df, labels, metric="precomputed")
            calinski_harabasz = calinski_harabasz_score(kl_df, labels)
            davies_bouldin = davies_bouldin_score(kl_df, labels)
            
            silhouette_scores.append(silhouette)
            calinski_harabasz_scores.append(calinski_harabasz)
            davies_bouldin_scores.append(davies_bouldin)

            if silhouette > highest_silhouette:
                highest_silhouette = silhouette
                best_k = k
        else:
            silhouette_scores.append(None)
            calinski_harabasz_scores.append(None)
            davies_bouldin_scores.append(None)

    # シルエットスコアのプロット
    plt.figure(figsize=(10, 6))
    # plt.plot(threshold_values, silhouette_scores, marker='o', color='b')
    plt.plot(k_values, silhouette_scores, marker='o', color='b')
    # plt.xlabel("Threshold")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score vs. Threshold for Fog ID {fog_id}")
    plt.grid(True)
    silhouette_file = f"silhouette_score_vs_threshold_{fog_id}.svg"
    plt.savefig(Path(save_dir) / silhouette_file, format='svg')
    plt.close()
    print(f"シルエットスコアのグラフを保存しました: {silhouette_file}")

    # Calinski-Harabaszスコアのプロット
    plt.figure(figsize=(10, 6))
    # plt.plot(threshold_values, calinski_harabasz_scores, marker='o', color='g')
    plt.plot(k_values, calinski_harabasz_scores, marker='o', color='g')
    # plt.xlabel("Threshold")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel("Calinski-Harabasz Score")
    plt.title(f"Calinski-Harabasz Score vs. Threshold for Fog ID {fog_id}")
    plt.grid(True)
    calinski_file = f"calinski_harabasz_score_vs_threshold_{fog_id}.svg"
    plt.savefig(Path(save_dir) / calinski_file, format='svg')
    plt.close()
    print(f"Calinski-Harabaszスコアのグラフを保存しました: {calinski_file}")

    # Davies-Bouldinスコアのプロット
    plt.figure(figsize=(10, 6))
    # plt.plot(threshold_values, davies_bouldin_scores, marker='o', color='r')
    plt.plot(k_values, davies_bouldin_scores, marker='o', color='r')
    # plt.xlabel("Threshold")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel("Davies-Bouldin Score")
    plt.title(f"Davies-Bouldin Score vs. Threshold for Fog ID {fog_id}")
    plt.grid(True)
    davies_file = f"davies_bouldin_score_vs_threshold_{fog_id}.svg"
    plt.savefig(Path(save_dir) / davies_file, format='svg')
    plt.close()
    print(f"Davies-Bouldinスコアのグラフを保存しました: {davies_file}")

    # print(f"最適なしきい値: {best_k}, 最高シルエットスコア: {highest_silhouette}")
    # return best_threshold
    print(f"最適なクラスタ数: {best_k}, 最高シルエットスコア: {highest_silhouette}")
    return best_k

def cluster_clients(fog_id, save_dir):
    """各フォグごとに最適な閾値を用いてクラスタリングを実行"""
    input_file = Path(save_dir) / f"kl_divergence_matrix_fog_{fog_id}.csv"
    kl_df = pd.read_csv(input_file, index_col=0)

    best_threshold = find_best_threshold(kl_df, fog_id, save_dir)
    
    condensed_distance_matrix = squareform(kl_df.values)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=kl_df.index)
    plt.title(f"Fog {fog_id} - Client Clustering Dendrogram")
    plt.xlabel("Client ID")
    plt.ylabel("Distance")
    plt.savefig(Path(save_dir) / f"dendrogram_{fog_id}.svg")

    clusters = fcluster(linkage_matrix, best_threshold, criterion='distance')
    
    fog_results = {}
    for client_id, cluster_id in zip(kl_df.index, clusters):
        if cluster_id not in fog_results:
            fog_results[cluster_id] = []
        fog_results[cluster_id].append(client_id)

    return {str(fog_id): {str(cluster_id): clients for cluster_id, clients in fog_results.items()}}

def run_hierarchical_clustering(save_dir: str, output_file: str):
    """全フォグのクラスタリング結果をまとめてJSONに保存"""
    clustered_data = {}
    for i in range(5):
        fog_result = cluster_clients(fog_id=i, save_dir=save_dir)
        clustered_data.update(fog_result)

    with open(output_file, 'w') as f:
        json.dump(clustered_data, f, indent=4)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")