import argparse
import gc
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig
from hfl_app.app import start_hfl_server
from hfl_app.strategy.fedavg import FedAvg
from models.base_model import Net
from models.driver import test
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Server")
parser.add_argument("--server_address", type=str, required=True, default="0.0.0.0:8080", help="server ipaddress:post")
parser.add_argument(
    "--dataset", type=str, required=True, choices=["CIFAR10", "CelebA"], help="FL config: dataset name"
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--model", type=str, required=True, choices=["tinyCNN", "ResNet18", "GNResNet18"], help="FL config: model name"
)
parser.add_argument(
    "--pretrained", type=str, required=False, choices=["IMAGENET1K_V1", None], default=None, help="pretraing recipe"
)
parser.add_argument("--num_rounds", type=int, required=False, default=5, help="FL config: aggregation rounds")
parser.add_argument("--num_fogs", type=int, required=False, default=4, help="FL config: number of fogs")
parser.add_argument("--local_epochs", type=int, required=False, default=5, help="Fog fit config: local epochs")
parser.add_argument("--batch_size", type=int, required=False, default=10, help="Fog fit config: batchsize")
parser.add_argument("--lr", type=float, required=False, default=0.01, help="Fog fit config: learning rate")
parser.add_argument("--weight_decay", type=float, required=False, default=0.0, help="weigh_decay")
parser.add_argument("--save_dir", type=str, required=False, help="save directory for the obtained results")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")


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
        name=args.model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
        pretrained=args.pretrained,
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())

    def fit_config(server_rnd: int) -> Dict[str, Scalar]:
        config = {
            "server_round": server_rnd,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        }
        return config

    server_config = ServerConfig(num_rounds=args.num_rounds)

    def eval_config(server_rnd: int) -> Dict[str, Scalar]:
        config = {
            "batch_size": args.batch_size,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(dataset_name=dataset, train=False, target=target, download=True)
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
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_fogs=args.num_fogs,
        min_evaluate_fogs=args.num_fogs,
        min_available_fogs=args.num_fogs,
        evaluate_fn=get_eval_fn(net, args.dataset, args.target),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        initial_parameters=init_parameters,
    )
    # Start Flower server for four rounds of federated learning
    start_hfl_server(server_address=args.server_address, config=server_config, strategy=strategy)
    # Save results
    save_path = Path(args.save_dir) / "config.yaml"
    config = vars(args)
    with open(save_path, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
