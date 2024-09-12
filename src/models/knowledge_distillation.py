import timeit
from typing import Any, Dict, List

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_federated_dataset
from utils.utils_model import load_model
from logging import DEBUG, INFO
from flwr.common.logger import log

from .base_model import Net


def mutual_train(
    client_net: Net,
    meme_net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    weight_decay: float = 0.0,
    device: torch.device = "cpu",
):
    client_optimizer = torch.optim.SGD(
        client_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    meme_optimizer = torch.optim.SGD(
        meme_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    client_net.to(device)
    meme_net.to(device)
    for _ in range(epochs):
        meme_net.eval()
        client_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                meme_outputs = meme_net(images)
            client_outputs = client_net(images)
            client_optimizer.zero_grad()
            loss = loss_kd(client_outputs, labels, meme_outputs, alpha)
            loss.backward()
            client_optimizer.step()
        client_net.eval()
        meme_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                client_outputs = client_net(images)
            meme_outputs = meme_net(images)
            meme_optimizer.zero_grad()
            loss = loss_kd(meme_outputs, labels, client_outputs, beta)
            loss.backward()
            meme_optimizer.step()


def loss_kd(outputs, labels, teacher_outputs, alpha):
    loss = alpha * nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1)
    ) + (1 - alpha) * F.cross_entropy(outputs, labels)
    return loss


def loss_kd_multiple(outputs, labels, teacher_outputs_list, alpha):
    loss = (1 - alpha) * F.cross_entropy(outputs, labels)
    for teacher_outputs in teacher_outputs_list:
        loss += (
            alpha
            * nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1)
            )
            / len(teacher_outputs_list)
        )
    return loss


@ray.remote
def distillation_parameters(
    teacher_parameters: Parameters,
    student_parameters: Parameters,
    config: Dict[str, Any],
) -> Parameters:
    # dataset configuration
    dataset = load_federated_dataset(
        dataset_name=config["dataset_name"],
        fid=config["fid"],
        clsid=config["clsid"],
        train=True,
        target=config["target_name"],
        attribute="cluster",
    )
    # model configuration
    dataset_config = configure_dataset(
        dataset_name=config["dataset_name"],
        target=config["target_name"],
    )
    teacher_net: Net = load_model(
        name=config["teacher_model"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    teacher_net.set_weights(parameters_to_ndarrays(teacher_parameters))
    student_net: Net = load_model(
        name=config["student_model"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    student_net.set_weights(parameters_to_ndarrays(student_parameters))

    # training configuration
    lr: float = float(config["lr"])
    batch_size: int = int(config["batch_size"])
    weight_decay: float = float(config["weight_decay"])
    alpha: float = float(config["alpha"])
    epochs: int = int(config["client_epochs"])

    optimizer = torch.optim.SGD(
        student_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher_net.to(device)
    teacher_net.eval()
    student_net.to(device)
    student_net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_net(images)
            student_outputs = student_net(images)
            optimizer.zero_grad()
            loss = loss_kd(
                outputs=student_outputs,
                labels=labels,
                teacher_outputs=teacher_outputs,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            break  # using only one minibatch

    return ndarrays_to_parameters(student_net.get_weights())


@ray.remote
def distillation_multiple_parameters(
    teacher_parameters_list: List[Parameters],
    student_parameters: Parameters,
    config: Dict[str, Any],
) -> Parameters:
    # log(
    #     INFO,
    #     "distillation_multiple_parameters() on fog fid: %s clsid: %s started",
    #     config["fid"],
    #     config["clsid"],
    # )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dataset configuration
    dataset = load_federated_dataset(
        dataset_name=config["dataset_name"],
        fid=config["fid"],
        clsid=config["clsid"],
        train=True,
        target=config["target_name"],
        attribute="cluster",
    )
    # model configuration
    dataset_config = configure_dataset(
        dataset_name=config["dataset_name"],
        target=config["target_name"],
    )
    teacher_net_list: List[Net] = []
    for teacher_parameters in teacher_parameters_list:
        teacher_net: Net = load_model(
            name=config["teacher_model"],
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
        teacher_net.set_weights(parameters_to_ndarrays(teacher_parameters))
        teacher_net_list.append(teacher_net.to(device).eval())
    student_net: Net = load_model(
        name=config["student_model"],
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    student_net.set_weights(parameters_to_ndarrays(student_parameters))

    # training configuration
    lr: float = float(config["lr"])
    batch_size: int = int(config["batch_size"])
    weight_decay: float = float(config["weight_decay"])
    beta: float = float(config["beta"])
    epochs: int = int(config["global_epochs"])

    optimizer = torch.optim.SGD(
        student_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    student_net.to(device)
    student_net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs_list = [
                    teacher_net(images) for teacher_net in teacher_net_list
                ]

            student_outputs = student_net(images)
            optimizer.zero_grad()
            loss = loss_kd_multiple(
                outputs=student_outputs,
                labels=labels,
                teacher_outputs_list=teacher_outputs_list,
                alpha=beta,
            )
            loss.backward()
            optimizer.step()
            break  # using only one minibatch

    return ndarrays_to_parameters(student_net.get_weights())
