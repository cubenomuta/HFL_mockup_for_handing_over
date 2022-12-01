import argparse
import random
import warnings

import numpy as np
import torch
from client_app.base_client import FlowerClient
from client_app.dml_client import FlowerDMLClient
from flwr.client import start_client

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Client")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    default="0.0.0.0:8080",
    help="server ipaddress:post",
)
parser.add_argument(
    "--cid",
    type=str,
    required=True,
    help="Client id for data partitioning.",
)
parser.add_argument(
    "--strategy",
    type=str,
    required=True,
    choices=["FedAvg", "FML"],
    help="FL config: aggregation strategy",
)
parser.add_argument(
    "--server_model",
    type=str,
    required=True,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    help="FL config: server-side model name",
)
parser.add_argument(
    "--client_model",
    type=str,
    required=True,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    help="FL config: client-side model name",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=["CIFAR10", "FashionMNIST", "MNIST", "CelebA", "usbcam"],
    default="CIFAR10",
    help="dataset name for FL training",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    # Parse command line argument `partition`
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "server_model_name": args.server_model,
        "client_model_name": args.client_model,
    }
    if args.strategy == "FML":
        client = FlowerDMLClient(cid=args.cid, config=config)
    elif (
        args.strategy == "FedAvg"
        or args.strategy == "F2MKD"
        or args.strategy == "FedFog"
    ):
        client = FlowerClient(cid=args.cid, config=config)
    start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
