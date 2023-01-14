import argparse
import concurrent.futures
import json
import os
import random
from logging import DEBUG, INFO
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import ray
import torch
import yaml
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from models.base_model import Net
from models.driver import test, train
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_federated_dataset
from utils.utils_model import load_model

FitResultsAndFailures = Tuple[
    List[Tuple[str, FitRes]],
    List[Union[Tuple[str, FitRes], BaseException]],
]

parser = argparse.ArgumentParser("Solo training")

parser.add_argument(
    "--client_model",
    type=str,
    required=True,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    help="FL config: client-side model name",
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
    required=True,
    help="Number of clients per each fog server",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["FashionMNIST", "CIFAR10", "CelebA"],
    help="FL config: dataset name",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
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


def fit_clients(
    client_instructions: List[Tuple[str, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, cid, fit_ins, timeout)
            for cid, fit_ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )
    results: List[Tuple[str, FitRes]] = []
    failures: List[Union[Tuple[str, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future,
            results=results,
            failures=failures,
        )
    return results, failures


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[str, FitRes]],
    failures: List[Union[Tuple[str, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a fog
    result: Tuple[str, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, fog returned a result where the status code is not OK
    failures.append(result)


def fit_client(cid: str, ins: FitIns, timeout: Optional[float]):

    future_res = solo_train.remote(cid, ins)
    try:
        res = ray.get(future_res, timeout=timeout)
    except Exception as ex:
        log(DEBUG, ex)
        raise ex

    fit_res = cast(FitRes, res)

    return cid, fit_res


@ray.remote(num_cpus=1)
def solo_train(cid: str, ins: FitIns):
    """Configure the training model"""
    log(INFO, "solo_train() on cid=%s", cid)
    fid = str(int(cid) // 100)

    config = ins.config
    weights: NDArrays = parameters_to_ndarrays(ins.parameters)

    # dataset configuration
    dataset = config["dataset_name"]
    target = config["target_name"]

    trainset = load_federated_dataset(
        dataset_name=dataset,
        id=cid,
        train=True,
        target=target,
        attribute="client",
    )

    testset = load_federated_dataset(
        dataset_name=dataset,
        id=fid,
        train=False,
        target=target,
        attribute="fog",
    )

    # model configuration
    client_model = config["client_model"]
    input_spec = config["input_spec"]
    out_dims = config["out_dims"]

    net: Net = load_model(
        name=client_model,
        input_spec=input_spec,
        out_dims=out_dims,
    )
    net.set_weights(weights)

    # fit configuration
    num_rounds: int = int(config["num_rounds"])
    epochs: int = int(config["local_epochs"])
    batch_size: int = int(config["batch_size"])
    lr: float = float(config["lr"])
    weight_decay: float = float(config["weight_decay"])

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=1000,
        shuffle=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {
        "loss": [],
        "acc": [],
    }
    for current_round in range(num_rounds):
        train(
            net=net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        res = test(
            net=net,
            testloader=testloader,
            device=device,
        )
        results["loss"].append((current_round, res["loss"]))
        results["acc"].append((current_round, res["acc"]))
    return FitRes(
        status=Status(code=Code.OK, message="successful fit"),
        parameters=None,
        num_examples=len(trainset),
        metrics=results,
    )


def main():
    args = parser.parse_args()
    print(args)
    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    set_seed(args.seed)

    net: Net = load_model(
        name=args.client_model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())

    assert os.path.exists(args.yaml_path)
    with open(args.yaml_path, "r") as f:
        fit_parameter_config = yaml.safe_load(f)
    params_config = vars(args)
    params_config.update(fit_parameter_config)

    fit_parameter_config.update(dataset_config)
    fit_parameter_config["num_rounds"] = args.num_rounds
    fit_parameter_config["dataset_name"] = args.dataset
    fit_parameter_config["target_name"] = args.target
    fit_parameter_config["client_model"] = args.client_model

    ray_config = {"include_dashboard": False}
    ray.init(**ray_config)
    log(
        INFO,
        "Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    # Fit configurations
    cids = [str(cid) for cid in range(args.num_clients)]
    fit_ins: FitIns = FitIns(parameters=init_parameters, config=fit_parameter_config)
    client_instructions = [(cid, fit_ins) for cid in cids]

    results, failures = fit_clients(
        client_instructions=client_instructions,
        max_workers=args.num_clients,
        timeout=None,
    )
    log(
        INFO,
        "fit_clients() received %s results and %s failures ",
        len(results),
        len(failures),
    )
    metrics_distributed = {}
    metrics_distributed["loss"] = {
        int(cid): fit_res.metrics["loss"] for cid, fit_res in results
    }
    metrics_distributed["accuracy"] = {
        int(cid): fit_res.metrics["acc"] for cid, fit_res in results
    }
    # loss of client models
    losses_distributed = sorted(metrics_distributed["loss"].items())
    save_path = Path(args.save_dir) / "metrics" / "losses_distributed.json"
    with open(save_path, "w") as f:
        json.dump(losses_distributed, f)

    # accuracy of client models
    accuracies_distributed = sorted(metrics_distributed["accuracy"].items())
    save_path = Path(args.save_dir) / "metrics" / "accuracies_distributed.json"
    with open(save_path, "w") as f:
        json.dump(accuracies_distributed, f)

    # configuration of the executed simulation
    save_path = Path(args.save_dir) / "params_config.yaml"
    with open(save_path, "w") as f:
        yaml.dump(params_config, f)


if __name__ == "__main__":
    main()
    ray.shutdown()
