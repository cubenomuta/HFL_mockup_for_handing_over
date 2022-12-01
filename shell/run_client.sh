#!/bin/bash

. ./shell/path.sh

# args=()
# for arg in $@; do
# args+=($arg)
# done
# server_address=${args[1]}
# cid=${args[3]}

cid=$CID
# if [ ${cid} -gt 4 ]; then
# server_address="133.11.194.40:8082"
# else
# server_address="133.11.194.40:8081"
# fi    
server_address="133.11.194.40:8080"

# fl_configuration
strategy="FedAvg"
server_model="GNResNet18"
client_model="GNResNet18"
dataset="CIFAR10"
target="iid_iid"
num_clients=5

# fit configuration
batch_size=10
local_epochs=1
lr=0.05

seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${strategy}_${model}/"${target}"/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi


python ./local/client.py --server_address ${server_address} \
--cid ${cid} \
--strategy ${strategy} \
--server_model ${server_model} \
--client_model ${client_model} \
--dataset ${dataset} \
--target ${target} \
--seed ${seed} &
# 2>"${exp_dir}/logs/client${cid}_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait