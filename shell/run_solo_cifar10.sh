#!/bin/bash

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

. ./shell/path.sh

args=()
for arg in $@; do
    args+=($arg)
done
server_address=${args[1]}

# fl configuration
strategy="Solo"
client_model="tinyCNN"
dataset="CIFAR10"
targets=("iid_noniid-dir0.1_kmeans_cluster_num=10" "iid_noniid-dir0.5_kmeans_cluster_num=10")
save_model=0

# fl configuration
num_rounds=300
num_clients=300

# fit configuration
yaml_path="./conf/${dataset}/${strategy}_${client_model}/fit_config.yaml"
seed=1234

for target in "${targets[@]}"; do
    time=`date '+%Y%m%d%H%M'`
    exp_dir="./simulation/${dataset}/${target}/${strategy}_${client_model}/run_${time}"

    if [ ! -e "${exp_dir}" ]; then
        mkdir -p "${exp_dir}/logs/"
        mkdir -p "${exp_dir}/models/"
        mkdir -p "${exp_dir}/metrics/"
    fi

    ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 8
    sleep 1 

    python ./local/solo.py \
    --client_model ${client_model} \
    --dataset ${dataset} \
    --target ${target} \
    --num_rounds ${num_rounds} \
    --num_clients ${num_clients} \
    --yaml_path ${yaml_path} \
    --save_dir ${exp_dir} \
    --seed ${seed} \
    2>"${exp_dir}/logs/flower.log" &

    # Wait for all background processes to complete
    wait

    ray stop -f
    rm -rf /tmp/ray
done