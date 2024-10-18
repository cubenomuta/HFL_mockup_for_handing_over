import json
from pathlib import Path
from flwr.common.logger import log
from logging import INFO, DEBUG

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import transforms

# クラスタ用
class FashionMNIST_truncated(Dataset):
    """
    Copied and modified from
    NIID-Bench
    """

    def __init__(
        self,
        root: str,
        fid: str = None,
        clsid: str = None,
        train: bool = True,
        target: str = None,
        attribute: str = None, # cluster or client
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        self.fid = fid
        self.clsid = clsid
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data_root = Path(root)
        self.json_path = None
        if target is not None and attribute is not None:
            self.json_root = (
                Path(root) / "FashionMNIST" / "partitions" / target / attribute
            )
            if self.train:
                self.json_path = self.json_root / "train_data.json"
            # clientはtest_data.jsonを作成していない -> 何で評価している？
            # test用のロードはすべてattribute="fog"で行っている -> クライアントの評価はフォグが持つすべてのデータを使用している
            else:
                self.json_path = self.json_root / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        fmnist_dataobj = FashionMNIST(
            self.data_root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        data = fmnist_dataobj.data
        target = fmnist_dataobj.targets

        if self.json_path is not None:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
            fog_data = json_data[self.fid]
            data_idx = fog_data[self.clsid]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class FashionMNIST_client_truncated(Dataset):
    """
    Copied and modified from
    NIID-Bench
    """

    def __init__(
        self,
        root: str,
        id: str = None,
        train: bool = True,
        target: str = None,
        attribute: str = None,
        transform=None,
        target_transform=None,
        download: bool = False,
        shuffle: bool = False,
    ):
        self.id = id
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.shuffle = shuffle

        self.data_root = Path(root)
        self.json_path = None
        if target is not None and attribute is not None:
            self.json_root = (
                Path(root) / "FashionMNIST" / "partitions" / target / attribute
            )
            if self.train:
                self.json_path = self.json_root / "train_data.json"
            else:
                if self.shuffle:
                    self.json_path = self.json_root / "before_shuffle_test_data.json"
                else:
                    self.json_path = self.json_root / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        fmnist_dataobj = FashionMNIST(
            self.data_root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        data = fmnist_dataobj.data
        target = fmnist_dataobj.targets

        if self.json_path is not None:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
            data_idx = json_data[self.id]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MNIST_truncated(Dataset):
    """
    Copied and modified from
    NIID-Bench
    """

    def __init__(
        self,
        root: str,
        id: str = None,
        train: bool = True,
        target: str = None,
        attribute: str = None,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        self.id = id
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data_root = Path(root)
        self.json_path = None
        if target is not None and attribute is not None:
            self.json_root = Path(root) / "MNIST" / "partitions" / target / attribute
            if self.train:
                self.json_path = self.json_root / "train_data.json"
            else:
                self.json_path = self.json_root / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        mnist_dataobj = MNIST(
            self.data_root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        data = mnist_dataobj.data
        target = np.array(mnist_dataobj.targets)

        if self.json_path is not None:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
            data_idx = json_data[self.id]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_truncated(Dataset):
    """
    Copied and modified from
    NIID-Bench
    """

    def __init__(
        self,
        root: str,
        id: str = None,
        train: bool = True,
        target: str = None,
        attribute: str = None,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.id = id
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data_root = Path(root) / "CIFAR10" / "raw"
        self.json_path = None
        if target is not None and attribute is not None:
            self.json_root = Path(root) / "CIFAR10" / "partitions" / target / attribute
            if self.train:
                self.json_path = self.json_root / "train_data.json"
            else:
                self.json_path = self.json_root / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(
            self.data_root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.json_path is not None:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
            data_idx = json_data[self.id]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(Dataset):
    """
    Copied and modified from
    NIID-Bench
    """

    def __init__(
        self,
        root: str,
        id: str = None,
        train: bool = True,
        target: str = None,
        attribute: str = None,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.id = id
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data_root = Path(root) / "CIFAR100" / "raw"
        self.json_path = None
        if target is not None and attribute is not None:
            self.json_root = (
                Path(root) / "CIFAR100" / "paratitions" / target / attribute
            )
            if self.train:
                self.json_path = self.json_root / "train_data.json"
            else:
                self.json_path = self.json_root / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(
            self.data_root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.json_path is not None:
            with open(self.json_path, "r") as f:
                json_data = json.load(f)
            data_idx = json_data[self.id]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FederatedCelebaVerification(Dataset):
    def __init__(
        self, id: str, target: str = "small", train: bool = True, transform=None
    ) -> None:
        self.id = int(id)
        self.root = Path("./data/celeba")
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        if train:
            self.json_path = self.root / "identities" / target / "train_data.json"
        else:
            self.json_path = self.root / "identities" / target / "test_data.json"
        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

        self.num_samples = self.json_data["num_samples"][self.id]
        self.user_key = self.json_data["users"][self.id]

        self.data = {"x": [], "y": []}
        self.data["x"].extend(self.json_data["user_data"][self.user_key]["x"])
        self.data["y"].extend(self.json_data["user_data"][self.user_key]["y"])
        assert self.data["y"][0] == self.id

    def __getitem__(self, index):
        img_path = self.root / "img_landmarks_align_celeba" / self.data["x"][index]
        img = Image.open(img_path)
        target = self.data["y"][index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.num_samples


class FederatedUsbcamVerification(Dataset):
    def __init__(self, id: str, train: bool = True, transform=None) -> None:
        self.id = int(id)
        self.root = Path("./data/usbcam")
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        if train:
            self.json_path = self.root / "identities" / "train_data.json"
        else:
            self.json_path = self.root / "identities" / "test_data.json"
        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

        self.data = {}
        self.num_samples = self.json_data["num_samples"]
        self.data["x"] = self.json_data["user_data"]["x"]
        self.data["y"] = [self.id for _ in range(self.num_samples)]

    def __getitem__(self, index):
        img_path = self.root / "img_landmarks_align_usbcam" / self.data["x"][index]
        img = Image.open(img_path)
        target = self.data["y"][index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    dataset = CIFAR10_truncated(
        root="./data",
        id="0",
        train=True,
        target="iid_iid",
        attribute="client",
        download=True,
    )
