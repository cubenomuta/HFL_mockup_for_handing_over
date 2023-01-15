from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from models.knowledge_distillation import distillation_multiple_parameters


class F2MKD(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def configure_fit(
        self,
        server_round: int,
        client_parameters_dict: Dict[str, Parameters],
        config: Dict[str, Any] = None,
        client_manager: ClientManager = None,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if config is None:
            config = {}
            if self.on_fit_config_fn(server_round):
                config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        client_instructions = [
            (
                client,
                FitIns(parameters=client_parameters_dict[client.cid], config=config),
            )
            for client in clients
        ]

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[
        Optional[Parameters], Optional[Dict[str, Parameters]], Dict[str, Scalar]
    ]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return metrics_aggregated
        # if not results:
        #     return None, None, {}

        # if not self.accept_failures and failures:
        #     return None, None, {}

        # client_parameters_dict = {
        #     client.cid: fit_res.parameters for client, fit_res in results
        # }

        # parameters_prime = distillation_multiple_parameters(
        #     teacher_parameters_list=[
        #         client_parameters
        #         for client_parameters in client_parameters_dict.values()
        #     ],
        #     student_parameters=server_parameters,
        #     config=config,
        # )
        # return parameters_prime, client_parameters_dict, {}

    def configure_evaluate(
        self,
        server_round: int,
        client_parameters_dict: Dict[str, Parameters],
        config: Dict[str, Any] = None,
        client_manager: ClientManager = None,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if config is None:
            config = {}
            if self.on_evaluate_config_fn(server_round):
                config = self.on_evaluate_config_fn(server_round)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        client_instructions = [
            (
                client,
                EvaluateIns(
                    parameters=client_parameters_dict[client.cid], config=config
                ),
            )
            for client in clients
        ]

        return client_instructions
