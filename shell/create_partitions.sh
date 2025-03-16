#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_fogs=5
num_clients=100

# dataset configuration
dataset="FashionMNIST" # FashionMNIST, CIFAR-10
fog_partitions=$1 # "noniid-label2,1", "iid"
client_partitions=$2 # "noniid-label2,1", "iid" "noniid-dir0.1,0.5"
client_shuffle_ratio=0.5
seed=1234
# cluster_alg="kmeans"
# cluster_alg="hierarchical"
cluster_alg="linkage"
# cluster_alg="dbscan"
cluster_num=2


if [[ "$client_partitions" == *"noniid-dir"* ]]; then
    save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}_${cluster_alg}"
elif [[ "$client_partitions" == *"part-noniid"* ]]; then
    save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}_${cluster_alg}_${client_shuffle_ratio}"
else
    save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}"
fi

if [ ! -e "${save_dir}" ]; then
    mkdir -p "${save_dir}/logs/"
fi

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

python src/dataset_app/client_stats.py \
--save_dir ${save_dir}