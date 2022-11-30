import argparse
import random
import warnings

import numpy as np
import torch
from flwr.server import SimpleClientManager
from fog_app.app import start_fog
from fog_app.base_fog import FlowerFog
from fog_app.strategy import FedAvg

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
    "--fog_address",
    type=str,
    required=True,
    default="0.0.0.0:8080",
    help="server ipaddress:post",
)
parser.add_argument(
    "--fid", type=str, required=True, help="Client id for data partitioning."
)
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=["CIFAR10", "CelebA", "usbcam"],
    default="CIFAR10",
    help="dataset name for FL training",
)
parser.add_argument(
    "--num_clients", type=int, required=False, default=10, help="Num. of clients"
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    default="tinyCNN",
    help="model name for FL training",
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
        "model_name": args.model,
    }
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
    )
    client_manager = SimpleClientManager
    fog: FlowerFog = FlowerFog(
        fid=args.fid, config=config, client_manager=client_manager, strategy=strategy
    )
    start_fog(
        server_address=args.server_address,
        fog_address=args.fog_address,
        fog=fog,
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
