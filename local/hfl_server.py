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
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig
from hfl_server_app.app import start_hfl_server
from hfl_server_app.base_hflserver import HFLServer
from hfl_server_app.fog_manager import SimpleFogManager
from hfl_server_app.strategy.fedavg import FedAvg
from models.base_model import Net
from models.driver import test
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Hierarchical Server")
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
    "--num_rounds",
    type=int,
    required=False,
    default=5,
    help="FL config: aggregation rounds",
)
parser.add_argument(
    "--num_fogs",
    type=int,
    required=False,
    default=4,
    help="FL config: number of fogs",
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


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    # Parse command line argument `partition`
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    # Model configuration
    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    server_net: Net = load_model(
        name=args.server_model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    server_init_parameters: Parameters = ndarrays_to_parameters(
        server_net.get_weights()
    )

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
            "server_round": server_round,
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

    def evaluate_metrics_server_aggregation_fn(
        eval_metrics: List[Tuple[int, Dict[str, Any]]]
    ):
        accuracy_summary = np.array(
            [metrics["accuracy_mean"] for _, metrics in eval_metrics]
        )
        loss_summary = np.array([metrics["loss_mean"] for _, metrics in eval_metrics])
        metrics_aggregated = {
            "accuracy_mean": float(accuracy_summary.mean()),
            "accuracy_std": float(accuracy_summary.std()),
            "loss_mean": float(loss_summary.mean()),
            "loss_std": float(loss_summary.std()),
        }
        return metrics_aggregated

    # Create strategy
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_fogs=args.num_fogs,
        min_evaluate_fogs=args.num_fogs,
        min_available_fogs=args.num_fogs,
        evaluate_fn=get_eval_fn(server_net, args.dataset, args.target),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        initial_parameters=server_init_parameters,
    )
    fog_manager = SimpleFogManager()
    hfl_server = HFLServer(
        fog_manager=fog_manager,
        strategy=strategy,
    )
    # Start Flower server for four rounds of federated learning
    hist = start_hfl_server(
        server_address=args.server_address,
        hfl_server=hfl_server,
        config=server_config,
    )

    # # Dump results
    # # loss of global model
    # losses_centralized = hist.losses_centralized
    # save_path = Path(args.save_dir) / "metrics" / "losses_centralized.json"
    # with open(save_path, "w") as f:
    #     json.dump(losses_centralized, f)

    # # accuracy of global model
    # accuracies_centralized = sorted(hist.metrics_centralized["accuracy"])
    # save_path = Path(args.save_dir) / "metrics" / "accuracies_centralized.json"
    # with open(save_path, "w") as f:
    #     json.dump(accuracies_centralized, f)

    # # loss of client models
    # losses_distributed = sorted(hist.metrics_distributed["loss"].items())
    # save_path = Path(args.save_dir) / "metrics" / "losses_distributed.json"
    # with open(save_path, "w") as f:
    #     json.dump(losses_distributed, f)

    # # accuracy of client models
    # accuracies_distributed = sorted(hist.metrics_distributed["accuracy"].items())
    # save_path = Path(args.save_dir) / "metrics" / "accuracies_distributed.json"
    # with open(save_path, "w") as f:
    #     json.dump(accuracies_distributed, f)

    # # configuration of the executed simulation
    # params_config = vars(args)
    # params_config.update(fit_parameter_config)
    # save_path = Path(args.save_dir) / "params_config.yaml"
    # with open(save_path, "w") as f:
    #     yaml.dump(params_config, f)


if __name__ == "__main__":
    main()
