#!/bin/bash
# if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
# while read line
# do
# info=${line}
# done < ${CUDA_VISIBLE_DEVICES_FILE}
# fi
# CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

# for ((i=0; i < 10; i++)); do
#     . ./shell/run_client.sh ${i} &
#     # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_fog.sh --server_address 172.20.0.2:8080 --fog_address 172.20.0.2:808$i --fid ${i} &
# done
# wait
for fp in 'iid'; do
for cp in 'iid' 'noniid-label1' 'noniid-label2'; do
echo ${fp} ${cp}
. ./shell/create_partitions.sh ${fp} ${cp} &
wait
done
done
for fp in 'noniid-label2'; do
for cp in 'iid' 'noniid-label1'; do
echo ${fp} ${cp}
. ./shell/create_partitions.sh ${fp} ${cp} &
wait
done
done
for fp in 'noniid-label1'; do
for cp in 'iid'; do
echo ${fp} ${cp}
. ./shell/create_partitions.sh ${fp} ${cp} &
wait
done
done