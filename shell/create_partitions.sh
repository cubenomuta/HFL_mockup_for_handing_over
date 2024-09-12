#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_fogs=10
num_clients=100

# dataset configuration
dataset="FashionMNIST"
fog_partitions=$1 # "noniid-label2,1", "iid"
client_partitions=$2 # "noniid-label2,1", "iid"
client_shuffle_ratio=0.2
seed=1234

save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}"

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
1> "${save_dir}/logs/standard.log" \
2> "${save_dir}/logs/flower.log"