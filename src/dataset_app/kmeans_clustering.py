import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import matplotlib.pyplot as plt
from kneed import KneeLocator

# save_dir = "./data/FashionMNIST/partitions/noniid-label2_part-noniid_0.2/client"
# save_dir = "./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"

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
    """エルボー法とシルエット法を使って最適なクラスタ数を決定"""
    sse = []
    silhouette_scores = []
    k_values = range(2, max_clusters + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        clusters = kmeans.fit_predict(X)
        sse.append(kmeans.inertia_)  # SSE
        silhouette_scores.append(silhouette_score(X, clusters))
    
    # エルボー法プロット
    plt.figure()
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(save_dir / f'elbow_method_plot_{fid}.svg')
    
    # シルエットスコアプロット
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.savefig(save_dir / f'silhouette_score_plot_{fid}.svg')
    
    # 最適クラスタ数をシルエットスコアの最大値から決定
    # optimal_k = k_values[np.argmax(silhouette_scores)]
    # print(f"最適なクラスタ数（シルエット法）: {optimal_k}")

    # エルボーポイントの特定
    kneedle = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    elbow_k = kneedle.elbow

    # エルボー法で選択されたポイントの前後のクラスタ数を候補にする
    candidate_k = [elbow_k - 1, elbow_k, elbow_k + 1]
    candidate_k = [k for k in candidate_k if k in k_values]

    # 候補の中でシルエットスコアが最大のものを選択(index -2でアクセスできる)
    best_k = max(candidate_k, key=lambda k: silhouette_scores[k - 2])
    print(f"最終的に選択されたクラスタ数: {best_k}")

    return best_k

def kmeans_clustering(data: dict, save_dir, num_fogs: int, num_clients: int, step: int = 100) -> dict:
    """各フォグサーバ範囲で最適なクラスタ数を決定し、クラスタリングを実行"""
    step = num_clients
    clustered_data = {}

    for i in range(0, num_fogs*num_clients, step):
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
            row = []
            for label, count in v.items():
                row.extend([int(label)] * int(count))  # 数値型に変換
            X.append(row)
        
        # NumPy配列に変換
        X = np.array([np.array(row) for row in X], dtype=float)

        # 最適なクラスタ数を決定
        optimal_clusters = determine_optimal_clusters(X, fid, save_dir)
        
        # KMeansによるクラスタリング
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init='auto')
        clusters = kmeans.fit_predict(X)

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

def run_kmeans_clustering(save_dir: str, output_file: str, num_fogs: int, num_clients: int):
    """クラスタリングプロセスを実行する関数"""

    input_file = Path(save_dir) / "client_train_data_stats.json"
    data = load_json_file(input_file)
    print("元のデータ:", list(data.items())[:5])  # 最初の5件を表示
    
    # クラスタリングを実行
    clustered_data = kmeans_clustering(data, save_dir, num_fogs, num_clients)
    
    # クラスタリング結果を保存
    save_json_file(clustered_data, output_file)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")


def main():
    save_dir = "./data/FashionMNIST/partitions/iid_noniid-dir0.5/client"

    input_file_path = Path(save_dir) / "client_train_data_stats.json"
    output_file_path = Path(save_dir) / "clustered_client_list.json"
    run_kmeans_clustering(input_file=input_file_path, output_file=output_file_path)

if __name__ == "__main__":
    main()