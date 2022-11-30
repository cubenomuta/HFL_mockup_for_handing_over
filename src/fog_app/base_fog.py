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
from flwr.server.server import fit_clients
from flwr.server.strategy import Strategy
from grpc import server

from .fog import Fog

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class FlowerFog(Server, Fog):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None
    ):
        super(FlowerFog, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        self.fid = fid
        self.attribute = "fog"
        self.config = config

    def fit(self, ins: FitIns) -> FitRes:
        server_round: int = int(ins.config["server_round"])

        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=None,
            ins=ins,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
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
            DEBUG,
            "fit_round %s: received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_prime, metrics_aggregated = aggregated_result

        return FitRes(
            status=Status(Code.OK, message="success fit"),
            parameters=parameters_prime,
            num_examples=metrics_aggregated["num_examples"],
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        server_round: int = int(ins.config["server_round"])
        res_eval = self.evaluate_round(server_round=server_round)
        loss_aggregated, metrics_aggregated, _ = res_eval
        return EvaluateRes(
            Status(Code.OK),
            loss=float(loss_aggregated),
            num_examples=metrics_aggregated["num_examples"],
            metrics={"accuracy": metrics_aggregated["accuracy"]},
        )
