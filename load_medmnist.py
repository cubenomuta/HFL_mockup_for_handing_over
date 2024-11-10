# load_medmnist.py
import medmnist
from medmnist import INFO
from medmnist import OrganAMNIST  # 使用するデータセットのクラスをインポート
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

def load_organmnist_data(data_flag='organamnist', split='train'):
    info = INFO[data_flag]
    data_class = getattr(medmnist, info['python_class'])
    dataset = data_class(split=split, transform=transforms.ToTensor(), download=True)
    return dataset

# データを読み込み
train_dataset = load_organmnist_data(data_flag='organamnist', split='train')

# データサンプルの表示
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    image, label = train_dataset[i]
    axs[i].imshow(image[0], cmap='gray')
    axs[i].set_title(f"Label: {label}")
    axs[i].axis('off')
plt.show()
