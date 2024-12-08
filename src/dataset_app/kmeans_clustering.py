import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    k_values = range(2, max_clusters + 1)
    json_file="clustering_scores.json"
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        clusters = kmeans.fit_predict(X)
        # 各種スコアの計算
        sse.append(kmeans.inertia_)  # SSE (エルボー法)
        silhouette_scores.append(silhouette_score(X, clusters))  # シルエットスコア
        calinski_harabasz_scores.append(calinski_harabasz_score(X, clusters))  # Calinski-Harabaszスコア
        davies_bouldin_scores.append(davies_bouldin_score(X, clusters))  # Davies-Bouldinスコア

    # JSONデータを既存ファイルから読み込む（なければ空の辞書）
    json_file_path = save_dir / json_file
    if json_file_path.exists():
        with open(json_file_path, "r") as f:
            all_scores = json.load(f)
    else:
        all_scores = {}

    # 現在のfidのスコアデータを保存
    all_scores[fid] = {
        "k_values": list(k_values),
        "sse": sse,
        "silhouette_scores": silhouette_scores,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "davies_bouldin_scores": davies_bouldin_scores
    }

    # 更新したスコアデータをJSONファイルに保存
    with open(json_file_path, "w") as f:
        json.dump(all_scores, f, indent=4)
    print(f"クラスタリングスコアのデータをJSONファイルに保存しました: {json_file_path}")
    
    # エルボー法プロット
    plt.figure()
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(save_dir / f'elbow_method_plot_{fid}.svg')
    print(f"エルボー法のプロットを '{save_dir / f'elbow_method_plot_{fid}.svg'}' に保存しました。")
    
    # シルエットスコアプロット
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.savefig(save_dir / f'silhouette_score_plot_{fid}.svg')
    print(f"シルエットスコアのプロットを '{save_dir / f'silhouette_score_plot_{fid}.svg'}' に保存しました。")

    # Calinski-Harabaszスコアプロット
    plt.figure()
    plt.plot(k_values, calinski_harabasz_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score for Optimal k')
    plt.savefig(save_dir / f'calinski_harabasz_score_plot_{fid}.svg')
    print(f"Calinski-Harabaszスコアのプロットを '{save_dir / f'calinski_harabasz_score_plot_{fid}.svg'}' に保存しました。")
    
    # Davies-Bouldinスコアプロット
    plt.figure()
    plt.plot(k_values, davies_bouldin_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score for Optimal k')
    plt.savefig(save_dir / f'davies_bouldin_score_plot_{fid}.svg')
    print(f"Davies-Bouldinスコアのプロットを '{save_dir / f'davies_bouldin_score_plot_{fid}.svg'}' に保存しました。")
    
    # 最適クラスタ数をシルエットスコアの最大値から決定
    # optimal_k = k_values[np.argmax(silhouette_scores)]
    # print(f"最適なクラスタ数（シルエット法）: {optimal_k}")

    # エルボーポイントの特定
    kneedle = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    elbow_k = kneedle.elbow
    # エルボーポイントが見つからない場合の対処
    if elbow_k is None:
        print("警告: エルボーポイントが見つかりませんでした。シルエットスコアに基づいてクラスタ数を選択します。")
        # シルエットスコアの最大値を持つkを選択
        best_k = k_values[np.argmax(silhouette_scores)]
    else:
        print(f"エルボー法で選択されたクラスタ数: {elbow_k}")
        # エルボーポイントの前後のクラスタ数を候補にする
        candidate_k = [elbow_k - 1, elbow_k, elbow_k + 1]
        candidate_k = [k for k in candidate_k if k in k_values]
        # 候補の中でシルエットスコアが最大のものを選択
        best_k = max(candidate_k, key=lambda k: silhouette_scores[k - 2])
    print(f"最終的に選択されたクラスタ数: {best_k}")

    return best_k

def kmeans_clustering(data: dict, save_dir, num_fogs: int, num_clients: int, num_classes: int, cluster_num: int = None, step: int = 100) -> dict:
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
            row = [0] * num_classes  # クラス数に合わせて初期化
            for label, count in v.items():
                row[int(label)] = int(count)  # クラス番号のインデックスにカウントを設定
            X.append(row)
        
        # NumPy配列に変換
        X = np.array([np.array(row) for row in X], dtype=float)

        optimal_clusters = determine_optimal_clusters(X, fid, save_dir) # json出力
        # 最適なクラスタ数を決定
        if cluster_num:
            optimal_clusters = cluster_num

        print(f"cluster_num={cluster_num, }optimal_clusters={optimal_clusters}")
        
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

def run_kmeans_clustering(save_dir: str, output_file: str, num_fogs: int, num_clients: int, num_classes: int, cluster_num: int = None): 
    """クラスタリングプロセスを実行する関数"""

    input_file = Path(save_dir) / "client_train_data_stats.json"
    data = load_json_file(input_file)
    print("元のデータ:", list(data.items())[:5])  # 最初の5件を表示
    
    # クラスタリングを実行
    clustered_data = kmeans_clustering(data, save_dir, num_fogs, num_clients, num_classes, cluster_num)
    
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