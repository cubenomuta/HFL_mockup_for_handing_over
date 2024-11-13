#!/bin/bash

. ./shell/path.sh

if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
export CUDA_VISIBLE_DEVICES="$line"
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi

# fl configuration
strategy="F2MKD"
server_model="tinyCNN"
# client_model="tinyCNN"
client_models="tinyCNN,tinyCNN_1conv" # csv形式
dataset="OrganAMNIST"
target="noniid-label2_part-noniid_0.2"
num_rounds=300
num_fogs=5
num_clients=100
fraction_fit=1

# fit configuration
yaml_path="./conf/${dataset}/${strategy}_${server_model}_tinyCNN/fit_config.yaml" # とりま固定値
seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./simulation/${dataset}/f_${num_fogs}_c${num_clients}_${target}/${strategy}_${server_model}_${client_models}/run_${time}"
data_dir="./data/${dataset}/partitions/${target}/client"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/create_client_model_name_dict.py \
--num_fogs ${num_fogs} \
--num_clients ${num_clients} \
--target ${target} \
--client_models ${client_models} \
--save_dir ${data_dir} \
--seed ${seed} 

ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 5
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