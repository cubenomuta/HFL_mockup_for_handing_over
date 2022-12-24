#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}

# fl configuration
strategy="F2MKD"
server_model="tinyCNN"
client_model="tinyCNN"
dataset="FashionMNIST"
target="iid_iid"
num_rounds=10
num_fogs=2
num_clients=10
fraction_fit=1

# fit configuration
yaml_path="./conf/${dataset}/${strategy}_${server_model}_${client_model}/fit_config.yaml"
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${target}/${strategy}_${server_model}_${client_model}/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/hfl_server.py --server_address ${server_address} \
--strategy ${strategy} \
--server_model ${server_model} \
--client_model ${client_model} \
--dataset ${dataset} \
--target ${target} \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--num_fogs ${num_fogs} \
--fraction_fit ${fraction_fit} \
--yaml_path ${yaml_path} \
--save_dir ${exp_dir} \
--seed ${seed} &
# 2>"${exp_dir}/logs/server_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait