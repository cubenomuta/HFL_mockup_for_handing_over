#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_fogs=5
num_clients=100

# dataset configuration
dataset="CIFAR10"
fog_partitions="iid" # "noniid-label2,1", "iid"
client_partitions_list=("noniid-dir0.1" "noniid-dir0.5") # 実行する client_partitions のリスト
client_shuffle_ratio=0.5
seed=1234

# 配列を定義
cluster_nums=(2 5 10)
cluster_algs=("kmeans" "hierarchical" "linkage")

for client_partitions in "${client_partitions_list[@]}"; do
    for cluster_num in "${cluster_nums[@]}"; do
        for cluster_alg in "${cluster_algs[@]}"; do

            # 保存先ディレクトリのパスを設定
            if [[ "$client_partitions" == *"noniid-dir"* ]]; then
                save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}_${cluster_alg}_cluster_num=${cluster_num}"
            elif [[ "$client_partitions" == *"part-noniid"* ]]; then
                save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}_${cluster_alg}_${client_shuffle_ratio}"
            else
                save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}"
            fi

            # 保存先ディレクトリを作成
            if [ ! -e "${save_dir}" ]; then
                mkdir -p "${save_dir}/logs/"
            fi

            # パーティション作成スクリプトの実行
            python ./local/create_partitions.py \
            --num_fogs ${num_fogs} \
            --num_clients ${num_clients} \
            --fog_partitions ${fog_partitions} \
            --client_partitions ${client_partitions} \
            --client_shuffle_ratio ${client_shuffle_ratio} \
            --dataset ${dataset} \
            --save_dir ${save_dir} \
            --seed ${seed} \
            --cluster_alg ${cluster_alg} \
            --cluster_num ${cluster_num} \
            1> "${save_dir}/logs/standard.log" \
            2> "${save_dir}/logs/flower.log"

            # クライアント統計スクリプトの実行
            python src/dataset_app/client_stats.py \
            --save_dir ${save_dir}
            
            echo "Completed: client_partitions=${client_partitions}, cluster_num=${cluster_num}, cluster_alg=${cluster_alg}"
        done
    done
done
