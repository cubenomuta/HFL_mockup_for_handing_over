#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}

# fl configuration
strategy="FML"
server_model="GNResNet18"
client_model="tinyCNN"
dataset="CIFAR10"
target="iid_iid"
save_model=0

# fl configuration
num_rounds=10
num_clients=10

# fit configuration
yaml_path="./conf/${dataset}/${strategy}_${server_model}_${client_model}/fit_config.yaml"
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${strategy}_${server_model}_${client_model}/${target}/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/server.py --server_address ${server_address} \
--strategy ${strategy} \
--server_model ${server_model} \
--client_model ${client_model} \
--dataset ${dataset} \
--target ${target} \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--yaml_path ${yaml_path} \
--save_dir ${exp_dir} \
--seed ${seed} &
# 2>"${exp_dir}/logs/server_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait