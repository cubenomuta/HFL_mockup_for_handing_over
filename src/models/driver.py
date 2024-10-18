import sys
from typing import Any, Dict
from logging import DEBUG, INFO
from flwr.common.logger import log

import ray
import torch
import torch.nn as nn
from flwr.common import (
    Code,
    EvaluateRes,
    Parameters,
    Scalar,
    Status,
    parameters_to_ndarrays,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils_dataset import configure_dataset, load_federated_dataset
from utils.utils_model import load_model

from models.base_model import Net

import json
import os
from pathlib import Path
DATA_ROOT = Path(os.environ["DATA_ROOT"])

def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    criterion: torch.nn.modules.Module = nn.CrossEntropyLoss(),
    device: str = "cpu",
    use_tqdm: bool = False,
) -> None:
    net.to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    if use_tqdm:
        for epoch in range(epochs):
            for _, data in tqdm(
                enumerate(trainloader),
                total=len(trainloader),
                file=sys.stdout,
                desc=f"[Epoch: {epoch}/ {epochs}]",
                leave=False,
            ):
                images, labels = data[0].to(device, non_blocking=True), data[1].to(
                    device, non_blocking=True
                )
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    else:
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    # net.to("cpu")


def test(
    net: Net, testloader: DataLoader, steps: int = None, device: str = "cpu"
) -> Dict[str, Scalar]:
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, steps, loss = 0, 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += float(criterion(outputs, labels).item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
    loss /= steps
    acc = correct / total
    return {"loss": loss, "acc": acc}


@ray.remote
def evaluate_parameters(
    parameters: Parameters,
    config: Dict[str, Any],
) -> EvaluateRes:
    # dataset configuration
    testset = load_federated_dataset(
        dataset_name=config["dataset_name"],
        id=config["fid"],
        train=False,
        target=config["target_name"],
        attribute="fog",
    )
    # model configuration
    dataset_config = configure_dataset(
        dataset_name=config["dataset_name"],
        target=config["target_name"],
    )
    net: Net = load_model(
        name=config["client_model_name"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    net.set_weights(parameters_to_ndarrays(parameters))

    # test configuration
    batch_size: int = int(config["batch_size"])
    # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    result = test(net=net, testloader=testloader, device=device)
    result["num_examples"] = len(testset)
    return result

@ray.remote
def evaluate_parameters_by_client_data(
    parameters: Parameters,
    config: Dict[str, Any],
) -> EvaluateRes:
    # dataset configuration
    # log(
    #     INFO,
    #     "evaluate_parameters_by_client_data() on cid=%s",
    #     config["cid"],
    # )
    testset = load_federated_dataset(
        dataset_name=config["dataset_name"],
        id=config["cid"],
        train=False,
        target=config["target_name"],
        attribute="client",
    )
    # model configuration
    dataset_config = configure_dataset(
        dataset_name=config["dataset_name"],
        target=config["target_name"],
    )
    net: Net = load_model(
        name=config["client_model_name"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    net.set_weights(parameters_to_ndarrays(parameters))

    # test configuration
    batch_size: int = int(config["batch_size"])
    # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    result = test(net=net, testloader=testloader, device=device)
    result["num_examples"] = len(testset)
    return result

@ray.remote
def evaluate_parameters_by_before_shuffle_fog_data(
    parameters: Parameters,
    config: Dict[str, Any],
) -> EvaluateRes:
    
    root = DATA_ROOT
    json_path = (
        Path(root) / config["dataset_name"] / "partitions" / config["target_name"] / "client" / "before_shuffle_cid_fid_dict.json"
    )
    
    with open(json_path, 'r') as f:
        cid_fid_dict = json.load(f)
    before_fid = cid_fid_dict[str(config["cid"])]
    after_fid = config["fid"]
    log(
        INFO, 
        "cid: %s, before_fid: %s, after_fid: %s",
        config["cid"],
        before_fid,
        after_fid
    )

    # if before_fid == 0:
    #     print(f"cid {config['cid']}: before_fid: {before_fid}, after_fid: {after_fid}")
    
    testset = load_federated_dataset(
        dataset_name=config["dataset_name"],
        id=str(before_fid),
        train=False,
        target=config["target_name"],
        attribute="fog",
        shuffle=True, # shuffle前のデータを取得
    )
    log(
        INFO,
        "len(testset): %s",
        len(testset)
    )
    # model configuration
    dataset_config = configure_dataset(
        dataset_name=config["dataset_name"],
        target=config["target_name"],
    )
    net: Net = load_model(
        name=config["client_model_name"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    net.set_weights(parameters_to_ndarrays(parameters))
    # test configuration
    batch_size: int = int(config["batch_size"])
    # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = test(net=net, testloader=testloader, device=device)
    result["num_examples"] = len(testset)
    return result