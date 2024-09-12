# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.
Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING, DEBUG, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from hfl_server_app.client_cluster_proxy import ClientClusterProxy
from ..fog_manager import FogManager
from ..fog_proxy import FogProxy
from .strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_fogs` lower than `min_fit_fogs` or
`min_evaluate_fogs` can cause the server to fail when there are too few fogs
connected to the server. `min_available_fogs` must be set to a value larger
than or equal to the values of `min_fit_fogs` and `min_evaluate_fogs`.
"""


class FedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_fogs: int = 2,
        min_evaluate_fogs: int = 2,
        min_available_fogs: int = 2,
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
        evaluate_metrics_cluster_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.
        Implementation based on https://arxiv.org/abs/1602.05629
        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of fogs used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of fogs used during validation. Defaults to 1.0.
        min_fit_fogs : int, optional
            Minimum number of fogs used during training. Defaults to 2.
        min_evaluate_fogs : int, optional
            Minimum number of fogs used during validation. Defaults to 2.
        min_available_fogs : int, optional
            Minimum number of total fogs in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if min_fit_fogs > min_available_fogs or min_evaluate_fogs > min_available_fogs:
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_fogs = min_fit_fogs
        self.min_evaluate_fogs = min_evaluate_fogs
        self.min_available_fogs = min_available_fogs
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.evaluate_metrics_cluster_aggregation_fn = evaluate_metrics_cluster_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_fogs(self, num_available_fogs: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        fogs."""
        num_fogs = int(num_available_fogs * self.fraction_fit)
        return max(num_fogs, self.min_fit_fogs), self.min_available_fogs

    def num_evaluation_fogs(self, num_available_fogs: int) -> Tuple[int, int]:
        """Use a fraction of available fogs for evaluation."""
        num_fogs = int(num_available_fogs * self.fraction_evaluate)
        return max(num_fogs, self.min_evaluate_fogs), self.min_available_fogs

    def initialize_parameters(self, fog_manager: FogManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, fog_manager: FogManager
    ) -> List[Tuple[FogProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample fogs
        sample_size, min_num_fogs = self.num_fit_fogs(fog_manager.num_available())
        fogs = fog_manager.sample(num_fogs=sample_size, min_num_fogs=min_num_fogs)

        # Return fog/config pairs
        # fit_ins はすべてのフォグで共通  (= グローバルモデルで更新)
        return [(fog, fit_ins) for fog in fogs]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, fog_manager: FogManager
    ) -> List[Tuple[FogProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample fogs
        sample_size, min_num_fogs = self.num_evaluation_fogs(
            fog_manager.num_available()
        )
        fogs = fog_manager.sample(num_fogs=sample_size, min_num_fogs=min_num_fogs)

        # Return fog/config pairs
        return [(fog, evaluate_ins) for fog in fogs]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientClusterProxy, FitRes]],
        failures: List[Union[Tuple[ClientClusterProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Information about the each cluster samples
        # for cluster, res in results:
            # log(DEBUG, f"Fog:{cluster.fid}, Cluster:{cluster.clsid} received {res.num_examples} examples")
            
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        del weights_results[:]

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        fit_metrics = {}
        if self.fit_metrics_aggregation_fn:
            # fit_metrics = {cluster.fid: res.metrics for cluster, res in results}
            for cluster, res in results:
                if cluster.fid not in fit_metrics:
                    fit_metrics[cluster.fid] = {}
                fit_metrics[cluster.fid][cluster.clsid] = res.metrics

            # 正直やらなくていい気がする、これは重要ではない
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[FogProxy, EvaluateRes]],
        failures: List[Union[Tuple[FogProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        # log(
        #     INFO,
        #     "round %s: loss_aggregated is completed %s",
        #     server_round,
        #     loss_aggregated,
        # )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # log(
        #     INFO,   
        #     "round %s: aggregate_evaluate is completed in aggregate_evaluate %s",
        #     server_round,
        #     metrics_aggregated,
        # )

        return loss_aggregated, metrics_aggregated
    
    def aggregate_cluster_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientClusterProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientClusterProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        # DEBUG OK
        # log(
        #     INFO,
        #     "round %s: start aggregate_cluster_evaluate",
        #     server_round,
        # )

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        # 加重平均による損失の集約
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        log(
            INFO,   
            "round %s: loss_aggregated in aggregate_cluster_evaluate is completed in aggregate_evaluate %s",
            server_round,
            loss_aggregated,
        )

        # Aggregate custom metrics if aggregation fn was provided
        # カスタムメトリクスの集約
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            # 
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_fog_aggregation_fn provided")

        log(
            INFO,   
            "round %s: metrics_aggregated in aggregate_cluster_evaluate is completed in aggregate_evaluate %s",
            server_round,
            metrics_aggregated,
        )

        return loss_aggregated, metrics_aggregated
