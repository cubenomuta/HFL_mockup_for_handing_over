#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=20.78.91.146:8080
# server_address=172.18.0.2:8080
fog_address=$1
fid=$2

# fl configuration
strategy="FedFog"
server_model="tinyCNN"
client_model="tinyCNN"
dataset="CIFAR10"
target="iid_iid"
num_clients=5
fraction_fit=1

# fl configuration
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${target}/${strategy}_${server_model}_${client_model}/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

if [ $strategy == "F2MKD" ]; then
ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 48
sleep 1
fi

python ./local/fog.py --server_address ${server_address} \
--fog_address ${fog_address} \
--fid ${fid} \
--strategy ${strategy} \
--server_model ${server_model} \
--client_model ${client_model} \
--dataset ${dataset} \
--target ${target} \
--num_clients ${num_clients} \
--fraction_fit ${fraction_fit} \
--seed ${seed} &
# 2>"${exp_dir}/logs/fog${fid}_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
ray stop -f