import os
from pathlib import Path
from typing import Dict, List, Tuple

from dataset_app import (
    CentralizedCelebaAndUsbcamVerification,
    CentralizedCelebaVerification,
    CIFAR10_truncated,
    CIFAR10_cluster_truncated,
    CIFAR100_truncated,
    NIH_CXR_truncated,
    NIH_CXR_cluster_truncated,
    FashionMNIST_client_truncated, #追加
    FashionMNIST_truncated,
    OrganAMNIST_client_truncated,   #追加
    OrganAMNIST_truncated,          #追加
    FederatedCelebaVerification,
    FederatedUsbcamVerification,
    MNIST_truncated,
    NIH_CXR,
)
from medmnist import OrganAMNIST
from flwr.common import Scalar
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import transforms

DATA_ROOT = Path(os.environ["DATA_ROOT"])


def load_centralized_dataset(
    dataset_name: str, train: bool = True, target: str = None, download: bool = False
) -> Dataset:
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        root = DATA_ROOT
        dataset = FashionMNIST(
            root=root, train=train, transform=transform, download=download
        )
    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        root = DATA_ROOT
        dataset = FashionMNIST(
            root=root, train=train, transform=transform, download=download
        )
    elif dataset_name == "OrganAMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        root = DATA_ROOT
        if train:
            dataset = OrganAMNIST(
                root=root, split='train', transform=transform, download=download
            )
        else:
            dataset = OrganAMNIST(
                root=root, split='test', transform=transform, download=download
            )
        target = dataset.labels
        if target.ndim > 1:
            target = target.flatten()
        dataset.labels = target

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        root = DATA_ROOT / "CIFAR10" / "raw"
        dataset = CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070, 0.4865, 0.4409),
                    (0.2673, 0.2564, 0.2762),
                ),
            ]
        )
        root = DATA_ROOT / "CIFAR100" / "raw"
        dataset = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
    elif dataset_name == "NIH_CXR":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )
        root = DATA_ROOT / "NIH_CXR" / "raw"
        dataset = NIH_CXR(
            centralized=True, train=train, transform=transform
        )
    elif dataset_name == "CelebA":
        assert target is not None
        if target == "mix_usbcam":
            dataset = CentralizedCelebaAndUsbcamVerification()
        else:
            dataset = CentralizedCelebaVerification(train=train, target=target)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
    return dataset


def load_federated_dataset( # クラスタ用
    dataset_name: str,
    fid: str = None,
    clsid: str = None,
    train: bool = True,
    target: str = None,
    attribute: str = None,
    download: bool = False,
) -> Dataset:
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = MNIST_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = FashionMNIST_truncated(
            root=DATA_ROOT,
            fid=fid,
            clsid=clsid,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "OrganAMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = OrganAMNIST_truncated(
            root=DATA_ROOT,
            fid=fid,
            clsid=clsid,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        dataset = CIFAR10_cluster_truncated(
            root=DATA_ROOT,
            fid=fid,
            clsid=clsid,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070, 0.4865, 0.4409),
                    (0.2673, 0.2564, 0.2762),
                ),
            ]
        )
        dataset = CIFAR100_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "NIH_CXR":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # 必要に応じて解像度を調整
                transforms.ToTensor()
            ]
        )
        dataset = NIH_CXR_cluster_truncated(
            root=DATA_ROOT,
            fid=fid,
            clsid=clsid,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CelebA":
        assert target is not None
        dataset = FederatedCelebaVerification(id=id, train=train, target=target)
    elif dataset_name == "usbcam":
        dataset = FederatedUsbcamVerification(id=id, train=train)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
    return dataset

def load_federated_client_dataset( # クライアント用
    dataset_name: str,
    id: str = None,
    train: bool = True,
    target: str = None,
    attribute: str = None,
    download: bool = False,
    shuffle: bool = False,
) -> Dataset:
    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = MNIST_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = FashionMNIST_client_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
            shuffle=shuffle,
        )
    elif dataset_name == "OrganAMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )
        dataset = OrganAMNIST_client_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
            shuffle=shuffle,
        )
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        dataset = CIFAR10_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070, 0.4865, 0.4409),
                    (0.2673, 0.2564, 0.2762),
                ),
            ]
        )
        dataset = CIFAR100_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "NIH_CXR":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # 必要に応じて解像度を調整
                transforms.ToTensor()
            ]
        )
        dataset = NIH_CXR_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            attribute=attribute,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CelebA":
        assert target is not None
        dataset = FederatedCelebaVerification(id=id, train=train, target=target)
    elif dataset_name == "usbcam":
        dataset = FederatedUsbcamVerification(id=id, train=train)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
    return dataset

def configure_dataset(dataset_name: str, target: str = None) -> Dict[str, Scalar]:
    if dataset_name == "MNIST":
        input_spec = (1, 28, 28)
        out_dims = 10
    elif dataset_name == "FashionMNIST":
        input_spec = (1, 28, 28)
        out_dims = 10
    elif dataset_name == "OrganAMNIST":
        input_spec = (1, 32, 32)
        out_dims = 11
    elif dataset_name == "CIFAR10":
        input_spec = (3, 32, 32)
        out_dims = 10
    elif dataset_name == "NIH_CXR":
        input_spec = (3, 32, 32)
        out_dims = 20
    elif (dataset_name == "CelebA") or (dataset_name == "usbcam"):
        input_spec = (3, 112, 112)
        if (target == "small") or (target == "mix_usbcam"):
            out_dims = 10
        elif target == "medium":
            out_dims = 100
        elif target == "large":
            out_dims = 1000
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    config = {"input_spec": input_spec, "out_dims": out_dims}
    return config


def split_validation(dataset: Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
    num_samples = dataset.__len__()
    num_train = int(num_samples * split_ratio)
    num_val = num_samples - num_train
    trainset, valset = random_split(dataset, [num_train, num_val])
    return trainset, valset
