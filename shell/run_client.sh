#!/bin/bash

. ./shell/path.sh

# args=()
# for arg in $@; do
# args+=($arg)
# done
# server_address=${args[1]}
# cid=${args[3]}

cid=$CID
# cid=$1
if [ ${cid} -gt 4 ]; then
# server_address="172.18.0.2:8082"
server_address="133.11.194.40:8082"
else
# server_address="172.18.0.2:8081"
server_address="133.11.194.40:8081"
fi    
server_address="20.78.91.146:8080"

# fl_configuration
strategy="FedFog"
server_model="tinyCNN"
client_model="tinyCNN"
dataset="CIFAR10"
target="iid_iid"

# fit configuration
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${target}/${strategy}_${server_model}_${client_model}/run_${time}"

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