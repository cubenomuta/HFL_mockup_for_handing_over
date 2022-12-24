import argparse
import gc
import os
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig, SimpleClientManager, start_server
from flwr.server.strategy import FedAvg
from models.base_model import Net
from models.driver import test
from server_app.custom_server import CustomServer
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Server")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    default="0.0.0.0:8080",
    help="server ipaddress:post",
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
    required=True,
    choices=["CIFAR10", "CelebA"],
    help="FL config: dataset name",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--num_rounds",
    type=int,
    required=False,
    default=5,
    help="FL config: aggregation rounds",
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
    "--yaml_path",
    type=str,
    required=True,
    help="File path to configure hyperparameter",
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=False,
    help="save directory for the obtained results",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    # Parse command line argument `partition`
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    params_config = vars(args)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)

    net: Net = load_model(
        name=args.server_model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())
    server_config = ServerConfig(num_rounds=args.num_rounds)

    assert os.path.exists(args.yaml_path)
    with open(args.yaml_path, "r") as f:
        fit_parameter_config = yaml.safe_load(f)

    def fit_config(server_rnd: int) -> Dict[str, Scalar]:
        return fit_parameter_config

    def eval_config(server_rnd: int) -> Dict[str, Scalar]:
        config = {
            "batch_size": args.batch_size,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(
            dataset_name=dataset, train=False, target=target, download=True
        )
        testloader = DataLoader(testset, batch_size=1000)

        def evaluate(
            server_round: int, weights: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(weights)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            results = test(model, testloader, device=device)
            torch.cuda.empty_cache()
            gc.collect()
            return results["loss"], {"accuracy": results["acc"]}

        return evaluate

    # Create strategy
    if args.strategy == "FedAvg" or args.strategy == "FML":
        strategy = FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=get_eval_fn(net, args.dataset, args.target),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            initial_parameters=init_parameters,
        )
    else:
        raise NotImplementedError(f"Strategy class {args.strategy} is not supported.")
    server = CustomServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
    )

    # Start Flower server for four rounds of federated learning
    start_server(
        server_address=args.server_address,
        server=server,
        config=server_config,
    )
    # Save results
    save_path = Path(args.save_dir) / "config.yaml"
    config = vars(args)
    config.update(fit_parameter_config)
    with open(save_path, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
