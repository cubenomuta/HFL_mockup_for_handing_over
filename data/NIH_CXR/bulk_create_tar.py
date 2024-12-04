import os
import tarfile
from math import ceil

# 画像データのディレクトリ
input_dir = "test_data"
output_dir = "test_tars"
num_parts = 10

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# ディレクトリ内のすべての画像ファイルを取得
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# 各パートに割り当てるファイル数
files_per_part = ceil(len(all_files) / num_parts)

# ファイルを分割して tar.gz に圧縮
for i in range(num_parts):
    part_files = all_files[i * files_per_part:(i + 1) * files_per_part]
    tar_filename = os.path.join(output_dir, f"test_data_part_{i + 1}.tar.gz")
    
    with tarfile.open(tar_filename, "w:gz") as tar:
        for file in part_files:
            tar.add(file, arcname=os.path.basename(file))  # 相対パスで追加
    print(f"Part {i + 1}: {len(part_files)} files compressed into {tar_filename}")

print("All files have been split and compressed!")
