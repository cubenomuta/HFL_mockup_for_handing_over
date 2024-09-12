import json
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


def load_json_file(file_path: str) -> dict:
    """JSONファイルを読み込む関数"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json_file(data: dict, file_path: str) -> None:
    """クラスタリング結果をJSONファイルとして保存する関数"""
    with open(file_path, 'w') as f:
        def convert(o):
            if isinstance(o, np.int32):
                return int(o)
            raise TypeError

        json.dump(data, f, default=convert)


def kmeans_clustering(data: dict, cluster_count: int = 10, step: int = 100) -> dict:
    """クラスタリングを実行し、100個単位でフォグIDごとにクライアントをまとめる関数"""
    clustered_data = {}

    for i in range(0, 1000, step):
        print(f"\nフォグサーバ範囲 {i}~{i + step - 1}")
        
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
                row.extend([label] * count)
            X.append(row)
        
        # KMeansによるクラスタリング
        kmeans = KMeans(n_clusters=cluster_count, random_state=0, n_init='auto')
        clusters = kmeans.fit_predict(X)
        
        # フォグID（fid）に基づいてデータをまとめる
        fid = str(i // step)  # フォグIDを文字列に変換

        # 同じクラスタID（clsid）のクライアントをまとめる
        clustered_data[fid] = {}
        for k, cluster in zip(keys, clusters):
            # クラスタIDを昇順に揃えるために事前にソート
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


def run_clustering_process(input_file: str, output_file: str, cluster_count: int = 10):
    """クラスタリングプロセスを実行する関数"""
    data = load_json_file(input_file)
    print("元のデータ:", list(data.items())[:5])  # 最初の5件を表示
    
    # クラスタリングを実行
    clustered_data = kmeans_clustering(data, cluster_count)
    
    # クラスタリング結果を保存
    save_json_file(clustered_data, output_file)
    print(f"クラスタリングが完了し、結果は '{output_file}' に保存されました。")