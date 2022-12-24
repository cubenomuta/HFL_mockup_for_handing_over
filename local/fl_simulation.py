import argparse
import gc
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from client_app.base_client import FlowerRayClient
from client_app.dml_client import FlowerRayDMLClient
from flwr.client import Client
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig, SimpleClientManager
from flwr.server.strategy import FedAvg
from models.base_model import Net
from models.driver import test
from server_app.logging_server import LoggingServer
from simulation_app.app import start_simulation
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower federated learning simulation")
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
    required=True,
    help="Directory path for saving results",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)

    net: Net = load_model(
        name=args.server_model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())

    client_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "server_model_name": args.server_model,
        "client_model_name": args.client_model,
        "paramters": init_parameters,
    }
    server_config = ServerConfig(num_rounds=args.num_rounds)

    assert os.path.exists(args.yaml_path)
    with open(args.yaml_path, "r") as f:
        fit_parameter_config = yaml.safe_load(f)

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        fit_parameter_config["server_round"] = server_round
        return fit_parameter_config

    def eval_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "batch_size": 1000,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(
            dataset_name=dataset, train=False, target=target
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

    def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Dict[str, Any]]]):
        metrics_aggregated = {
            "accuracy": {metrics["cid"]: metrics["acc"] for _, metrics in eval_metrics},
            "loss": {metrics["cid"]: metrics["loss"] for _, metrics in eval_metrics},
        }
        return metrics_aggregated

    if args.strategy == "FedAvg":
        assert args.server_model == args.client_model
        strategy = FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=get_eval_fn(net, args.dataset, args.target),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=init_parameters,
        )

        def client_fn(cid: str) -> Client:
            return FlowerRayClient(cid, client_config)

    elif args.strategy == "FML":
        strategy = FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=get_eval_fn(net, args.dataset, args.target),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=init_parameters,
        )

        def client_fn(cid: str, parameters: Parameters) -> Client:
            return FlowerRayDMLClient(cid, client_config, parameters)

    else:
        raise NotImplementedError(f"Strategy class {args.strategy} is not supported.")

    server = LoggingServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
    )
    client_resources = {"num_cpus": 1}
    ray_config = {
        "include_dashboard": False,
        "address": "auto",
    }
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        server=server,
        config=server_config,
        ray_init_args=ray_config,
        keep_initialised=True,
    )

    # Dump results
    # loss of global model
    losses_centralized = hist.losses_centralized
    save_path = Path(args.save_dir) / "metrics" / "losses_centralized.json"
    with open(save_path, "w") as f:
        json.dump(losses_centralized, f)

    # accuracy of global model
    accuracies_centralized = sorted(hist.metrics_centralized["accuracy"])
    save_path = Path(args.save_dir) / "metrics" / "accuracies_centralized.json"
    with open(save_path, "w") as f:
        json.dump(accuracies_centralized, f)

    # loss of client models
    losses_distributed = sorted(hist.metrics_distributed["loss"].items())
    save_path = Path(args.save_dir) / "metrics" / "losses_distributed.json"
    with open(save_path, "w") as f:
        json.dump(losses_distributed, f)

    # accuracy of client models
    accuracies_distributed = sorted(hist.metrics_distributed["accuracy"].items())
    save_path = Path(args.save_dir) / "metrics" / "accuracies_distributed.json"
    with open(save_path, "w") as f:
        json.dump(accuracies_distributed, f)

    # configuration of the executed simulation
    params_config = vars(args)
    params_config.update(fit_parameter_config)
    save_path = Path(args.save_dir) / "params_config.yaml"
    with open(save_path, "w") as f:
        yaml.dump(params_config, f)


if __name__ == "__main__":
    main()
