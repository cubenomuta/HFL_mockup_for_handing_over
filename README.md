# Federated Learning シミュレーション実行ガイド

本リポジトリでは、FedAvg, FedFog, FML などの既存手法や、F2MKD/F2MKDC（異種モデル対応）の実験を実行するための環境構築およびシミュレーション手順をまとめています。

---

## ブランチ情報

- **FedAvg, FedFog, FML の実行:**  
  ブランチ `first-simulation-and-client-shuffle`

- **F2MKD の実行（異種モデルも許容）:**  
  ブランチ `Feature/#15-F2MKD-resnet-and-mobilenet`

- **F2MKDC の実行（異種モデルも許容）:**  
  ブランチ `Feature/#14-F2MKDC-Heterogeneous-models`

---

## 環境構築

### 1. Pipenv のインストール

```sh
pip install pipenv
```

### 2. ライブラリのインストール
```sh
pipenv install
```

### 3. torchのインストールでエラーが起こる場合**

3.1. Pipfileからtorch, torchvisionを削除

3.2. ライブラリのインストールを実行
```sh
pipenv install
```
3.3. 仮想環境に入る
```sh
pipenv shell
```

3.4. CUDAのバージョンに合わせて下記を実行する
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```

---

## データパーティションの作成
使用ブランチ: Feature/#14-F2MKDC-Heterogeneous-models

### 1. 仮想環境に入る:

```sh
pipenv shell
```

### 2. スクリプトの実行:

```sh
. ./shell/create_partitions.sh {fog_partitions} {client_partitions}
```
- fog_partitions, client_partitions のオプション:
  - iid
  - noniid-labelx (x=1,2,...：xで指定した数のラベルのデータを渡す)
  - noniid-dirx (x=0～1：ディリクレ分布 α=x に基づくデータパーティションのデータを渡す)

### 3. 実行結果の例:

例えば、

- データセット: FashionMNIST
- fog_partitions: iid
- client_partitions: noniid-dir0.1
- クラスタリング手法: kmeans
- クラスタ数: 2
の場合、
ディレクトリ data/FashionMNIST/partitions/iid_noniid-dir0.1_kmeans_cluster_num=2 内にデータが生成されます。

---

# シミュレーションの実行
### 1. 仮想環境に入る

```sh
pipenv shell
```

### 2. 実行:

実行コマンドの引数は run_hfl_simulation.sh をご確認ください。

```sh
. ./shell/run_hfl_simulationo.sh iid_iid 5
```

### 3. 実行ログの確認

実行中・実行後のログは、run_hfl_simulationo.sh 内で指定された以下のパスに出力されます:
（ブランチによって若干の違いあり）

```
simulation/データセット名/f_フォグ数_c_クライアント数/データパーティション名/run_yyyymmddhhmm/logs/flower.log
```

### 4. 実行結果の出力

実行結果は、以下のフォルダ内に生成されます:

```sh
simulation/データセット名/f_フォグ数_c_クライアント数/データパーティション名/run_yyyymmddhhmm/metrics
```
accuracies_centralized.json：グローバルモデルの精度
accuracies_distributed.json：クライアントモデルの精度
losses_centralized.json：グローバルモデルの損失
losses_distributed.json：クライアントモデルの損失

### 5. 実行中断時の注意
実行を中断する場合は、Ctrl+C でプログラムを停止した後、以下のコマンドを実行してプログラム内で作成されたスレッドを停止してください:

```sh
ray stop
```


## 注意事項
使用する手法、モデル、フォグ数、クライアント数などは run_hfl_simulationo.sh 内で指定可能です。詳細はスクリプトを参照してください。

手法によって対応するブランチが異なるため、実行前に必ず対象ブランチを確認してください:

F2MKDC: Feature/#14-F2MKDC-Heterogeneous-models

F2MKD: Feature/#15-F2MKD-resnet-and-mobilenet

FedAvg, FedFog, FML: first-simulation-and-client-shuffle

以上の手順に従って、Federated Learning シミュレーション環境のセットアップと実行を行ってください。

各ステップの詳細な説明や引数の意味などは、各スクリプト内のコメントやドキュメントも合わせてご確認ください。
