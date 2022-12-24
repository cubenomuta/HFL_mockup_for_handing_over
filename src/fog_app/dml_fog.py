from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
)
from flwr.common.logger import log
from flwr.server import ClientManager, Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.strategy import Strategy

from .base_fog import FlowerFog
from .ray_fog import distillation_from_clients, distillation_from_server

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
import ray


class FlowerDMLFog(FlowerFog):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        client_init_parameters: Optional[Parameters] = None,
    ):
        super(FlowerDMLFog, self).__init__(
            fid=fid,
            config=config,
            client_manager=client_manager,
            strategy=strategy,
        )
        self.client_init_parameters = client_init_parameters
        self.client_parameters_dict = None
        ray_config = {
            "include_dashboard": False,
            "address": "auto",
        }
        ray.init(**ray_config)

        # self.client_parameters_dict: Dict[str, Parameters] = {
        #     str(cid): client_init_parameters
        #     for cid in self.client_manager().all().keys()
        # }

    def fit(self, ins: FitIns) -> FitRes:
        # Fit configuration
        server_round: int = int(ins.config["server_round"])
        server_parameters: Parameters = ins.parameters

        # Distillation from server model to client models
        if self.client_parameters_dict is None:
            self._client_manager.wait_for(self.strategy.min_fit_clients)
            self.client_parameters_dict: Dict[str, Parameters] = {
                str(cid): self.client_init_parameters
                for cid in self.client_manager().all().keys()
            }
        self.set_max_workers(max_workers=len(self.client_parameters_dict))

        distillation_from_server_config = {
            "teacher_model": self.config["server_model_name"],
            "student_model": self.config["client_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
        }
        distillation_from_server_config.update(ins.config)
        results, failures = distillation_from_server(
            server_parameters=server_parameters,
            client_parameters_dict=self.client_parameters_dict,
            config=distillation_from_server_config,
            max_workers=self.max_workers,
        )
        log(
            INFO,
            "distillation_from_server() on fog fid=%s: received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )
        if len(failures) > 0:
            raise ValueError("distillation is failed.")
        for cid, client_parameters in results:
            self.client_parameters_dict[cid] = client_parameters

        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            client_parameters_dict=self.client_parameters_dict,
            config=ins.config,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "fit() on fog fid=%s: strategy sampled %s clients (out of %s)",
            self.fid,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        self.set_max_workers(max_workers=len(client_instructions))

        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=None,
        )
        log(
            INFO,
            "fit() on fog fid=%s: received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )

        if len(failures) > 0:
            raise ValueError("Insufficient fit results from clients.")
        for client, fit_res in results:
            self.client_parameters_dict[client.cid] = fit_res.parameters

        # Distillation from multiple clients to server.
        distillation_from_clients_config = {
            "teacher_model": self.config["client_model_name"],
            "student_model": self.config["server_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
        }
        distillation_from_clients_config.update(ins.config)
        fog_parameters: Parameters = distillation_from_clients(
            teacher_parameters_list=[
                client_parameters
                for client_parameters in self.client_parameters_dict.values()
            ],
            student_parameters=server_parameters,
            config=distillation_from_clients_config,
        )
        if type(fog_parameters) == Parameters:
            log(
                INFO,
                "distillation_multiple_parameters() on fog fid=%s completed",
                self.fid,
            )

        return FitRes(
            status=Status(Code.OK, message="success fit"),
            parameters=fog_parameters,
            num_examples=len(self.trainset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        server_round: int = int(ins.config["server_round"])
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            client_parameters_dict=self.client_parameters_dict,
            config=ins.config,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=None,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result

        return EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(loss_aggregated),
            num_examples=int(ins.config["batch_size"]),
            metrics=metrics_aggregated,
        )
