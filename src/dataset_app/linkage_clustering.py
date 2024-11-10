import json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
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

def determine_optimal_clusters(X, fid, save_dir, max_clusters=10):
    """シルエット法、Calinski-Harabaszスコア、Davies-Bouldinスコアを使って最適なクラスタ数を決定"""
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    k_values = range(2, max_clusters + 1)
    
    # 階層型クラスタリングのリンク生成
    linkage_matrix = linkage(X, method='average')
    
    for k in k_values:
        # 階層型クラスタリングの結果を指定クラスタ数に分割
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        
        # クラスタ数が2以上のときのみスコアを計算
        if len(set(clusters)) > 1:
            silhouette_scores.append(silhouette_score(X, clusters, metric="euclidean"))
            calinski_harabasz_scores.append(calinski_harabasz_score(X, clusters))
            davies_bouldin_scores.append(davies_bouldin_score(X, clusters))
        else:
            # クラスタ数が1の場合、スコアをNoneで埋める
            silhouette_scores.append(None)
            calinski_harabasz_scores.append(None)
            davies_bouldin_scores.append(None)
    
    # シルエットスコアプロット
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('linkage() Silhouette Score for Optimal k')
    plt.savefig(save_dir / f'linkage_silhouette_score_plot_{fid}.svg')
    print(f"シルエットスコアをプロットしました{silhouette_scores}")

    # Calinski-Harabaszスコアプロット
    plt.figure()
    plt.plot(k_values, calinski_harabasz_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('linkage() Calinski-Harabasz Score for Optimal k')
    plt.savefig(save_dir / f'linkage_calinski_harabasz_score_plot_{fid}.svg')
    print(f"Calinski-Harabaszスコアをプロットしました{calinski_harabasz_scores}")
    
    # Davies-Bouldinスコアプロット
    plt.figure()
    plt.plot(k_values, davies_bouldin_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('linkage() Davies-Bouldin Score for Optimal k')
    plt.savefig(save_dir / f'linkage_davies_bouldin_score_plot_{fid}.svg')
    print(f"Davies-Bouldinスコアをプロットしました{davies_bouldin_scores}")
    
    # 最適クラスタ数をシルエットスコアの最大値から決定
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"最適なクラスタ数（シルエット法）: {best_k}")

    return best_k

def hierarchical_clustering(data: dict, save_dir, num_fogs: int, num_clients: int, num_classes: int, step: int = 100) -> dict:
    """各フォグサーバ範囲で最適なクラスタ数を決定し、階層型クラスタリングを実行"""
    step = num_clients
    clustered_data = {}

    for i in range(0, num_fogs * num_clients, step):
        print(f"\nフォグサーバ範囲 {i}~{i + step - 1}")
        # フォグID（fid）に基づいてデータをまとめる
        fid = str(i // step)  # フォグIDを文字列に変換
        
        # 現在のフォグサーバ範囲のデータを取得
        fog_data = {k: v for k, v in data.items() if i <= int(k) < i + step}
        print(f"対象データ（キーと値）: {list(fog_data.items())[:5]}")  # 最初の5件を表示
        
        # データのラベルとその数を配列に変換
        X = []
        keys = []
        for k, v in fog_data.items():
            keys.append(k)
            row = [0] * num_classes  # クラス数に合わせて初期化
            for label, count in v.items():
                row[int(label)] = int(count)  # クラス番号のインデックスにカウントを設定
            X.append(row)
        
        # NumPy配列に変換
        X = np.array([np.array(row) for row in X], dtype=float)

        # 最適なクラスタ数を決定
        optimal_clusters = determine_optimal_clusters(X, fid, save_dir)
        
        # 階層型クラスタリングのリンク生成
        linkage_matrix = linkage(X, method='average')
        
        # 最適なクラスタ数でクラスタリング
        clusters = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')

        # 同じクラスタID（clsid）のクライアントをまとめる
        clustered_data[fid] = {}
        for k, cluster in zip(keys, clusters):
            cluster = int(cluster)  # クラスタIDをint型に変換
            if cluster not in clustered_data[fid]:
                clustered_data[fid][cluster] = []  # クラスタIDごとにリストを作成
            clustered_data[fid][cluster].append(int(k))  # クライアントIDをint型で追加

    # クラスタIDを昇順に並び替えた辞書を返す
    sorted_clustered_data = {
        fid: {clsid: sorted(cids) for clsid, cids in sorted(clusters.items())}
        for fid, clusters in clustered_data.items()
    }

    return sorted_clustered_data

def run_linkage_clustering(save_dir: str, output_file: str, num_fogs: int, num_clients: int, num_classes: int): 
    """クラスタリングプロセスを実行する関数"""

    input_file = Path(save_dir) / "client_train_data_stats.json"
    data = load_json_file(input_file)
    print("元のデータ:", list(data.items())[:5])  # 最初の5件を表示
    
    # クラスタリングを実行
    clustered_data = hierarchical_clustering(data, save_dir, num_fogs, num_clients, num_classes)
    
    # クラスタリング結果を保存
    save_json_file(clustered_data, output_file)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")

def main():
    save_dir = "./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"

    input_file_path = Path(save_dir) / "client_train_data_stats.json"
    output_file_path = Path(save_dir) / "clustered_client_list.json"
    run_linkage_clustering(save_dir=save_dir, output_file=output_file_path, num_fogs=1, num_clients=100, num_classes=10)

if __name__ == "__main__":
    main()
