from typing import Dict, Optional

from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Scalar, Status
from flwr.server import ClientManager, Server
from flwr.server.strategy import Strategy

from .fog import Fog


class FlowerFog(Server):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None
    ):
        super(FlowerFog, self).__init__(client_manager=client_manager, strategy=strategy)
        self.fid = fid
        self.config = config

    def fit(self, ins: FitIns) -> FitRes:
        server_round: int = int(ins.config["server_round"])
        res_fit = self.fit_round(server_round=server_round, timeout=None)
        parameters_prime, metrics_aggregated, _ = res_fit

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
