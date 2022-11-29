#!/bin/bash
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
info=${line}
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi
CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

for ((i=1; i < 10; i++)); do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_face_verification_client.sh --server_address SERVER_IP:8080 --cid ${i} &
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_fog.sh --server_address 172.20.0.2:8080 --fog_address 172.20.0.2:808$i --fid ${i} &
done
wait