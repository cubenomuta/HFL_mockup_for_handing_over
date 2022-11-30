#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=20.78.91.146:8080
fog_address=$1
fid=$2

dataset="CIFAR10"
target="iid_iid"
model="GNResNet18"
pretrained="IMAGENET1K_V1"


# fl configuration
num_clients=5
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${strategy}_${model}/"${target}"/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi


python ./local/fog.py --server_address ${server_address} \
--fog_address ${fog_address} \
--fid ${fid} \
--num_clients ${num_clients} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--seed ${seed} &
# 2>"${exp_dir}/logs/fog${fid}_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait