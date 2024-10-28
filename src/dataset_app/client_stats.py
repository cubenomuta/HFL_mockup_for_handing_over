import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Create dataset partitions for fogs and clients")
parser.add_argument("--save_dir", type=str, required=True, help="save directory")

def main(args):
    print(args)
    save_dir = Path(args.save_dir) / "client"
    # 入力ファイルパス
    cluster_file_path = save_dir / "clustered_client_list.json"  # {fid: {cluster_id: [cid, cid,...]}}
    distribution_file_path = save_dir / "client_train_data_stats.json"  # {cid: {データ分布}}

    # 出力ファイルパス
    output_file_path = save_dir / "clustered_clients_stats.json"

    # JSONファイルの読み込み
    with open(cluster_file_path, 'r') as f:
        cluster_data = json.load(f)

    with open(distribution_file_path, 'r') as f:
        distribution_data = json.load(f)

    # 新しい構造を作成
    result = {}

    # クラスタデータをもとにして、各フォグID (fid) のクラスターを処理
    for fid, clusters in cluster_data.items():
        result[fid] = {}
        for cluster_id, client_ids in clusters.items():
            result[fid][cluster_id] = {}
            for cid in client_ids:
                # データ分布が存在する場合のみ追加
                if str(cid) in distribution_data:
                    # データ分布を直接追加
                    result[fid][cluster_id][str(cid)] = distribution_data[str(cid)]

    # 結果をJSON形式で保存
    with open(output_file_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"ファイルが保存されました: {output_file_path}")

    # 各クラスターごとにプロットを作成
    for fid, fog_data in result.items():
        for clsid, cluster_data in fog_data.items():
            # 保存ディレクトリを作成
            savefig_dir = save_dir / f"fid_{fid}"
            savefig_dir.mkdir(parents=True, exist_ok=True)

            num_clients = len(cluster_data)
            num_columns = 4
            num_rows = (num_clients + num_columns - 1) // num_columns  # Calculate rows based on total clients and columns

            fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))
            fig.suptitle("Class Distribution per Client", fontsize=16)

            # Plot each client's data in a matrix-like layout
            for idx, (client_id, class_counts) in enumerate(cluster_data.items()):
                ax = axes[idx // num_columns, idx % num_columns] if num_rows > 1 else axes[idx % num_columns]
                ax.bar(class_counts.keys(), class_counts.values(), color="skyblue")
                ax.set_title(f"Client {client_id}")
                ax.set_xlabel("Class ID")
                ax.set_ylabel("Sample Count")
                ax.set_xticks(range(10))
                ax.set_ylim(0, 80)

            # Hide any unused subplots
            if num_rows > 1:
                for j in range(idx + 1, num_rows * num_columns):
                    fig.delaxes(axes.flatten()[j])
            else:
                for j in range(idx + 1, num_columns):
                    fig.delaxes(axes[j])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(savefig_dir / f"clsid_{clsid}_client_data_distribution.svg")
            plt.close(fig)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
