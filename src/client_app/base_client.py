import timeit
import warnings
from logging import INFO
from os import stat
from typing import Dict

import ray
import json
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
    load_federated_client_dataset,
    split_validation,
)
from utils.utils_model import load_model

warnings.filterwarnings("ignore")


class FlowerClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid
        self.fid = str(int(self.cid) // 100)
        self.clsid = None
        self.attribute = "client"

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # clsid の取得
        # log(INFO, "get clsid at FlowerRayClient cid: %s", cid)
        file_path = f"./data/{self.dataset}/partitions/{self.target}/client/clustered_client_list.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
        cluster_data = data[self.fid]
        for clsid, cids in cluster_data.items():
            if int(self.cid) in cids:
                self.clsid = clsid

        self.trainset = load_federated_client_dataset(
            dataset_name=self.dataset,
            id=self.cid,
            train=True,
            target=self.target,
            attribute=self.attribute,
        )
        # validation_ratio = 0.8
        # self.trainset, self.valset = split_validation(
        #     dataset, split_ratio=validation_ratio
        # )
        self.testset = load_federated_client_dataset(
            dataset_name=self.dataset,
            id=self.cid,
            train=False,
            target=self.target,
            attribute="client",
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
        start_time = timeit.default_timer()
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
        )
        # valloader = DataLoader(
        #     self.valset, batch_size=100, shuffle=False, drop_last=False
        # )

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device="cpu",  # self.device,
            use_tqdm=False,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())
        comp_stamp = timeit.default_timer() - start_time
        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={"comp": comp_stamp},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)
        results = test(self.net, testloader=testloader)
        log(
            INFO,
            "evaluate() on client cid=%s: test loss %s / test acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )
        metrics = {
            "cid": self.cid,
            "acc": results["acc"],
            "loss": results["loss"],
        }

        return EvaluateRes(
            status=Status(Code.OK, message="Success eval"),
            loss=float(results["loss"]),
            num_examples=len(self.testset),
            metrics=metrics,
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
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
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

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrapping EvaluateIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration
        # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        testloader = DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            shuffle=False,
        )

        result = test(self.net, testloader=testloader, device="cpu")
        metrics = {"acc": result["acc"], "cid": str(self.cid)}
        return EvaluateRes(
            status=Status(Code.OK, message="Success evaluate"),
            loss=result["loss"],
            num_examples=len(self.testset),
            metrics=metrics,
        )

    def __del__(self):
        self.trainset = None
        self.testset = None
        self.net = None
