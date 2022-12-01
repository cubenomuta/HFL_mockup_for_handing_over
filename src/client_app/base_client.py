import warnings
from logging import INFO
from typing import Dict

import ray
import torch
from flwr.client import Client, NumPyClient
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    NDArrays,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from models.base_model import Net
from models.driver import test, train
from torch.utils.data import DataLoader
from utils.utils_dataset import (
    configure_dataset,
    load_federated_dataset,
    split_validation,
)
from utils.utils_model import load_model

warnings.filterwarnings("ignore")


class FlowerClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid
        self.fid = str(int(self.cid) // 100)
        self.attribute = "client"

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]
        validation_ratio = 0.8
        dataset = load_federated_dataset(
            dataset_name=self.dataset,
            id=self.cid,
            train=True,
            target=self.target,
            attribute=self.attribute,
        )
        self.trainset, self.valset = split_validation(
            dataset, split_ratio=validation_ratio
        )
        self.testset = load_federated_dataset(
            dataset_name=self.dataset,
            id=self.fid,
            train=False,
            target=self.target,
            attribute="fog",
        )

        # model configuration
        self.server_model = config["server_model_name"]
        self.client_model = config["client_model_name"]
        self.dataset_config = configure_dataset(self.dataset, target=self.target)
        self.net: Net = load_model(
            name=self.client_model,
            input_spec=self.dataset_config["input_spec"],
            out_dims=self.dataset_config["out_dims"],
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.net.get_weights())
        return GetParametersRes(status=Code.OK, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        print(ins.config)
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        valloader = DataLoader(
            self.valset, batch_size=100, shuffle=False, drop_last=False
        )

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
            use_tqdm=True,
        )
        results = test(self.net, valloader, device=self.device)
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())
        log(
            INFO,
            "fit() on client cid=%s: val loss %s / val acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size)
        results = test(self.net, testloader=testloader)
        log(
            INFO,
            "evaluate() on client cid=%s: test loss %s / test acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return EvaluateRes(
            status=Status(Code.OK, message="Success eval"),
            loss=float(results["loss"]),
            num_examples=len(self.testset),
            metrics={"accuracy": results["acc"]},
        )


class FlowerRayClient(FlowerClient):
    def __init__(self, cid: str, config: Dict[str, str]):
        super().__init__(cid, config)

    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())

        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={},
        )
