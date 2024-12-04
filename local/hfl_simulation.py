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
from flwr.client import Client
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig, SimpleClientManager
from fog_app.fog import Fog
from fog_app.ray_fog import RayFlowerDMLFogProxy, RayFlowerFogProxy
from fog_app.strategy.f2mkd import F2MKD
from fog_app.strategy.fedavg import FedAvg as FogFedAvg
from hfl_server_app.base_hflserver import HFLServer
from hfl_server_app.fog_manager import SimpleFogManager
from hfl_server_app.strategy.fedavg import FedAvg as HFLServerFedAvg
from models.base_model import Net
from models.driver import test
from simulation_app.hfl_app import start_simulation
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower hierarchical federated learning simulation")
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
    choices=["CIFAR10", "FashionMNIST", "OrganAMNIST", "MNIST", "Ce4ebA", "NIH_CXR"],
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
    torch.backends.cudnn.benchmark = False


def main():
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

    client_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "server_model_name": args.server_model,
        "client_model_name": args.client_model,
        "paramters": client_init_parameters,
    }
    fog_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "server_model_name": args.server_model,
        "client_model_name": args.client_model,
        "num_clients": args.num_clients,
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
            "server_round": server_round,
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

    def evaluate_metrics_server_aggregation_fn(
        eval_metrics: List[Tuple[int, Dict[str, Any]]]
    ):
        metrics_aggregated = {
            "accuracy": {},
            "loss": {},
        }
        for _, metrics in eval_metrics:
            metrics_aggregated["accuracy"].update(metrics["accuracy"])
            metrics_aggregated["loss"].update(metrics["loss"])
        return metrics_aggregated

    def evaluate_metrics_fog_aggregation_fn(
        eval_metrics: List[Tuple[int, Dict[str, Any]]]
    ):
        metrics_aggregated = {
            "accuracy": {metrics["cid"]: metrics["acc"] for _, metrics in eval_metrics},
            "loss": {metrics["cid"]: metrics["loss"] for _, metrics in eval_metrics},
        }
        return metrics_aggregated

    server_strategy = HFLServerFedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_fogs=args.num_fogs,
        min_evaluate_fogs=args.num_fogs,
        min_available_fogs=args.num_fogs,
        evaluate_fn=get_eval_fn(server_net, args.dataset, args.target),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_server_aggregation_fn,
        initial_parameters=server_init_parameters,
    )

    if args.strategy == "FedFog":
        assert args.server_model == args.client_model
        fog_strategy = FogFedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=get_eval_fn(server_net, args.dataset, args.target),
            evaluate_metrics_aggregation_fn=evaluate_metrics_fog_aggregation_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
        )

        def client_fn(cid: str) -> Client:
            return FlowerRayClient(cid, client_config)

        def fog_fn(fid: str) -> Fog:
            client_manager = SimpleClientManager()
            return RayFlowerFogProxy(
                fid=fid,
                config=fog_config,
                client_manager=client_manager,
                client_fn=client_fn,
                strategy=fog_strategy,
            )

    elif args.strategy == "F2MKD":
        fog_strategy = F2MKD(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=1,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=get_eval_fn(server_net, args.dataset, args.target),
            evaluate_metrics_aggregation_fn=evaluate_metrics_fog_aggregation_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
        )

        def client_fn(cid: str) -> Client:
            return FlowerRayClient(cid, client_config)

        def fog_fn(fid: str) -> Fog:
            client_manager = SimpleClientManager()
            return RayFlowerDMLFogProxy(
                fid=fid,
                config=fog_config,
                client_manager=client_manager,
                client_fn=client_fn,
                strategy=fog_strategy,
                client_init_parameters=client_init_parameters,
            )

    else:
        raise NotImplementedError(f"Strategy class {args.strategy} is not supported.")

    fog_manager = SimpleFogManager()
    hfl_server = HFLServer(
        fog_manager=fog_manager,
        strategy=server_strategy,
    )
    client_resources = {"num_cpus": 2}
    ray_config = {"include_dashboard": False, "address": "auto"}

    hist = start_simulation(
        client_fn=client_fn,
        fog_fn=fog_fn,
        num_fogs=args.num_fogs,
        client_resources=client_resources,
        hfl_server=hfl_server,
        config=server_config,
        ray_init_args=ray_config,
        keep_initialised=True,
    )
    hfl_server.make_time_json(
        save_dir=args.save_dir
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
