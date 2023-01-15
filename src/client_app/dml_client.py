import timeit
from typing import Any, Dict, Optional

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from models.base_model import Net
from models.driver import test
from models.knowledge_distillation import mutual_train
from torch.utils.data import DataLoader
from utils.utils_model import load_model

from .base_client import FlowerClient


class FlowerDMLClient(FlowerClient):
    def __init__(self, cid: str, config: Dict[str, str]):
        super().__init__(cid, config)
        self.meme: Net = load_model(
            name=self.server_model,
            input_spec=self.dataset_config["input_spec"],
            out_dims=self.dataset_config["out_dims"],
        )

    def fit(self, ins: FitIns) -> FitRes:
        start_time = timeit.default_timer()
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        alpha: float = float(ins.config["alpha"])
        beta: float = float(ins.config["beta"])
        weight_decay: float = float(ins.config["weight_decay"])

        # load model weight to meme model
        self.meme.set_weights(weights=weights)
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
        )

        mutual_train(
            client_net=self.net,
            meme_net=self.meme,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            alpha=alpha,
            beta=beta,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.meme.get_weights())
        comp_time = timeit.default_timer() - start_time
        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={"comp": comp_time},
        )


class FlowerRayDMLClient(FlowerClient):
    def __init__(
        self, cid: str, config: Dict[str, Any], parameters: Optional[Parameters] = None
    ):
        super().__init__(cid, config)
        if parameters is not None:
            weights = parameters_to_ndarrays(parameters)
            self.net.set_weights(weights=weights)
        self.meme: Net = load_model(
            name=self.server_model,
            input_spec=self.dataset_config["input_spec"],
            out_dims=self.dataset_config["out_dims"],
        )

    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        alpha: float = float(ins.config["alpha"])
        beta: float = float(ins.config["beta"])
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.meme.set_weights(weights)

        # dataset configuration train / validation
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
        )

        mutual_train(
            client_net=self.net,
            meme_net=self.meme,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            alpha=alpha,
            beta=beta,
            weight_decay=weight_decay,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.meme.get_weights())
        parameters_dual: Parameters = ndarrays_to_parameters(self.net.get_weights())

        return (
            FitRes(
                status=Status(Code.OK, message="Success fit"),
                parameters=parameters_prime,
                num_examples=len(self.trainset),
                metrics={},
            ),
            parameters_dual,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrapping EvaluateIns
        batch_size: int = int(ins.config["batch_size"])

        # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        testloader = DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            shuffle=False,
        )
        result = test(net=self.net, testloader=testloader, device=self.device)
        metrics = {
            "acc": float(result["acc"]),
            "loss": float(result["loss"]),
            "cid": int(self.cid),
        }
        return EvaluateRes(
            status=Status(Code.OK, message="Success evaluate"),
            loss=float(result["loss"]),
            num_examples=len(self.testset),
            metrics=metrics,
        )
