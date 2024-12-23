#!/bin/bash

. ./shell/path.sh

if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
export CUDA_VISIBLE_DEVICES="$line"
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi

# fl configuration
strategy="F2MKDC"
server_model="tinyCNN"
# client_model="tinyCNN"
client_models="tinyCNN" # csv形式
dataset="CIFAR10"
target="iid_noniid-dir0.1_kmeans_cluster_num=5"
num_rounds=300
num_fogs=3
num_clients=100
fraction_fit=1
yaml_params=("alpha_0.2" "alpha_0.5" "alpha_0.8")

# fit configuration
seed=1234

for yaml_param in "${yaml_params[@]}"; do
    yaml_path="./conf/${dataset}/${strategy}_${server_model}_${client_models}/fit_config_${yaml_param}.yaml"
    echo "Running ${yaml_path} with target ${target}"

    time=`date '+%Y%m%d%H%M'`
    exp_dir="./simulation/${dataset}/f_${num_fogs}_c_${num_clients}_${target}_${yaml_param}/${strategy}_${server_model}_${client_models}/run_${time}"

    if [ ! -e "${exp_dir}" ]; then
        mkdir -p "${exp_dir}/logs/"
        mkdir -p "${exp_dir}/models/"
        mkdir -p "${exp_dir}/metrics/"
    fi
    echo "create dir to ${exp_dir}"

    ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 8
    sleep 1 

    python ./local/hfl_simulation.py \
    --strategy ${strategy} \
    --server_model ${server_model} \
    --client_models ${client_models} \
    --dataset ${dataset} \
    --target ${target} \
    --num_rounds ${num_rounds} \
    --num_clients ${num_clients} \
    --num_fogs ${num_fogs} \
    --fraction_fit ${fraction_fit} \
    --yaml_path ${yaml_path} \
    --save_dir ${exp_dir} \
    --seed ${seed} \
    2>"${exp_dir}/logs/flower.log" &

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
    ray stop -f
    rm -rf /tmp/ray
done
