#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

. ./shell/path.sh

# CUDAデバイスの設定
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
    while read line; do
        export CUDA_VISIBLE_DEVICES="$line"
    done < ${CUDA_VISIBLE_DEVICES_FILE}
fi

# fl configuration
strategy="FedAvg"
server_model="tinyCNN"
client_model="tinyCNN"
dataset="FashionMNIST"
targets=("iid_noniid-dir0.1_kmeans_cluster_num=10" "iid_noniid-dir0.5_kmeans_cluster_num=10")
num_rounds=300
num_clients=300
fraction_fit=1

# fit configuration
yaml_path="./conf/${dataset}/${strategy}_${server_model}_${client_model}/fit_config.yaml"
seed=1234

# targetsごとにループ実行
for target in "${targets[@]}"; do
    time=`date '+%Y%m%d%H%M'`
    exp_dir="./simulation/${dataset}/${target}/${strategy}_${server_model}_${client_model}/run_${time}"

    if [ ! -e "${exp_dir}" ]; then
        mkdir -p "${exp_dir}/logs/"
        mkdir -p "${exp_dir}/models/"
        mkdir -p "${exp_dir}/metrics/"
    fi

    ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 8
    sleep 1 

    python ./local/fl_simulation.py \
    --strategy ${strategy} \
    --server_model ${server_model} \
    --client_model ${client_model} \
    --dataset ${dataset} \
    --target ${target} \
    --num_rounds ${num_rounds} \
    --num_clients ${num_clients} \
    --fraction_fit ${fraction_fit} \
    --yaml_path ${yaml_path} \
    --save_dir ${exp_dir} \
    --seed ${seed} \
    2>"${exp_dir}/logs/flower.log" &

    # Wait for all background processes to complete
    wait

    ray stop -f
    rm -rf /tmp/ray
done