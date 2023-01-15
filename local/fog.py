import argparse
import random
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from flwr.common import Parameters, Scalar, ndarrays_to_parameters
from flwr.server import SimpleClientManager
from fog_app.app import start_fog
from fog_app.base_fog import FlowerFog
from fog_app.dml_fog import FlowerDMLFog
from fog_app.strategy.f2mkd import F2MKD
from fog_app.strategy.fedavg import FedAvg
from models.base_model import Net
from utils.utils_dataset import configure_dataset
from utils.utils_model import load_model

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
    "--strategy",
    type=str,
    required=True,
    choices=["F2MKD", "FedFog"],
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
    required=True,
    choices=["CIFAR10", "FashionMNIST", "MNIST", "CelebA"],
    help="FL config: dataset name",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--num_clients",
    type=int,
    required=False,
    default=4,
    help="FL config: number of clients",
)
parser.add_argument(
    "--fraction_fit",
    type=float,
    required=False,
    default=1,
    help="FL config: client selection ratio",
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
    fog_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "server_model_name": args.server_model,
        "client_model_name": args.client_model,
        "num_clients": args.num_clients,
    }

    def fit_metrics_aggregation_fn(
        fit_metrics: List[Tuple[int, Dict[str, Any]]],
    ):
        timestamps_aggregated: Dict[str, Scalar] = {}
        for _, metrics in fit_metrics:
            cid = metrics["cid"]
            timestamps_aggregated[cid + "_comp"] = metrics["comp"]
            timestamps_aggregated[cid + "_comm"] = metrics["total"] - metrics["comp"]
        return timestamps_aggregated

    client_manager = SimpleClientManager()
    if args.strategy == "FedFog":
        assert args.server_model == args.client_model
        fog_strategy = FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        fog: FlowerFog = FlowerFog(
            fid=args.fid,
            config=fog_config,
            client_manager=client_manager,
            strategy=fog_strategy,
        )
    elif args.strategy == "F2MKD":
        fog_strategy = F2MKD(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        dataset_config: Dict[str, str] = configure_dataset(
            dataset_name=args.dataset, target=args.target
        )
        client_net: Net = load_model(
            name=args.client_model,
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
        client_init_parameters: Parameters = ndarrays_to_parameters(
            client_net.get_weights()
        )
        fog: FlowerFog = FlowerDMLFog(
            fid=args.fid,
            config=fog_config,
            client_manager=client_manager,
            strategy=fog_strategy,
            client_init_parameters=client_init_parameters,
        )

    start_fog(
        server_address=args.server_address,
        fog_address=args.fog_address,
        fog=fog,
    )


if __name__ == "__main__":
    main()
