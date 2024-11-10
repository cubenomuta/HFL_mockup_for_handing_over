# dataset configuration
dataset="FashionMNIST"
fog_partitions=$1 # "noniid-label2,1", "iid"
client_partitions=$2 # "noniid-label2,1", "iid"
seed=1234
# cluster_alg="kmeans"
cluster_alg="linkage"

for client_shuffle_ratio in $(seq 0.2 0.1 1.0); do
    save_dir="./data/${dataset}/partitions/${fog_partitions}_${client_partitions}_${client_shuffle_ratio}"

    if [ ! -e "${save_dir}" ]; then
        mkdir -p "${save_dir}/logs/"
    fi

    # 実行するPythonスクリプト
    python ./local/create_partitions.py \
    --num_fogs ${num_fogs} \
    --num_clients ${num_clients} \
    --fog_partitions ${fog_partitions} \
    --client_partitions ${client_partitions} \
    --client_shuffle_ratio ${client_shuffle_ratio} \
    --dataset ${dataset} \
    --save_dir ${save_dir} \
    --seed ${seed} \
    --cluster_alg ${cluster_alg} \
    1> "${save_dir}/logs/standard.log" \
    2> "${save_dir}/logs/flower.log"

    # クライアント統計情報の作成
    python src/dataset_app/client_stats.py \
    --save_dir ${save_dir}
done