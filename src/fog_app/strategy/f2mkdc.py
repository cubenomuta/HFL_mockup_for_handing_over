from logging import WARNING, DEBUG, INFO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    EvaluateRes,
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
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


from hfl_server_app.client_cluster_manager import ClientClusterManager
from hfl_server_app.client_cluster_proxy import ClientClusterProxy
from hfl_server_app.fog_proxy import FogProxy


class F2MKDC(FedAvg):
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
        evaluate_metrics_cluster_aggregation_fn: Optional[MetricsAggregationFn] = None,
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

        self.evaluate_metrics_cluster_aggregation_fn = evaluate_metrics_cluster_aggregation_fn

    def __repr__(self):
        return "F2MKDC"

    # hfl_server_appのfedvg.pyを参考にして作成
    # 返却値をリストにすると並列実行のプログラムが面倒なので、辞書にして返却
    def configure_cluster_fit(
        self,
        server_round: int,
        parameters: Parameters,  # すべてのクラスタモデルをグローバルモデルで更新する
        client_cluster_manager: ClientClusterManager,
    ) -> Dict[str, Tuple[ClientClusterProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters=parameters, config=config)

        # All clusters (サンプリングはなし)
        clusters = client_cluster_manager.all()

        # clsid をキーとする辞書を作成
        cluster_fit_dict = {}
        for cluster in clusters:
            clsid = cluster.clsid
            cluster_fit_dict[clsid] = (cluster, fit_ins)

        return cluster_fit_dict

    def configure_fit(
        self,
        server_round: int,
        client_parameters_dict: Dict[str, Parameters],
        client_models_name_dict: Dict[str, str],
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
                FitIns(
                    parameters=client_parameters_dict[client.cid], 
                    config={**config, "client_model_name": client_models_name_dict[client.cid]}
                ),
            )
            for client in clients
        ]

        return client_instructions
    
    def configure_partial_cient_fit(
        self,
        server_round: int,
        pre_client_instructions: List[Tuple[ClientProxy, FitIns]],
        client_parameters_dict: Dict[str, Parameters],
        client_models_name_dict: Dict[str, str],
        config: Dict[str, Any] = None,
        client_manager: ClientManager = None,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if config is None:
            config = {}
            if self.on_fit_config_fn(server_round):
                config = self.on_fit_config_fn(server_round)

        # Sample clients
        # sample_size, min_num_clients = self.num_fit_clients(
        #     client_manager.num_available()
        # )
        # clients = client_manager.sample(
        #     num_clients=sample_size, min_num_clients=min_num_clients
        # )
        # Return client/config pairs
        client_instructions = [
            (
                client,
                FitIns(
                    parameters=client_parameters_dict[client.cid], 
                    config={**config, "client_model_name": client_models_name_dict[client.cid]}
                ),
            )
            for client, _ in pre_client_instructions
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
        
    def configure_cluster_evaluate(
        self,
        server_round: int,
        cluster_parameters_dict: Parameters, # クラスタモデルのパラメータ
        config: Dict[str, Any],
        client_cluster_manager: ClientClusterManager,
    ) -> Dict[str, Tuple[ClientClusterProxy, EvaluateIns]]:
        if config is None:
            config = {}
            if self.on_evaluate_config_fn(server_round):
                config = self.on_evaluate_config_fn(server_round)

        # All clusters (サンプリングはなし)
        clusters = client_cluster_manager.all()

        cluster_evaluate_dict = {}
        for cluster in clusters:
            clsid = cluster.clsid
            cluster_evaluate_dict[clsid] = (
                cluster, 
                EvaluateIns(
                    parameters=cluster_parameters_dict[cluster.clsid], config=config
                )
            )

        return cluster_evaluate_dict

    def configure_evaluate(
        self,
        server_round: int,
        client_parameters_dict: Dict[str, Parameters],
        client_models_name_dict: Dict[str, str],
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
                    parameters=client_parameters_dict[client.cid], 
                    config={**config, "client_model_name": client_models_name_dict[client.cid]}
                ),
            )
            for client in clients
        ]

        return client_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # DEBUG OK
        # log(
        #     INFO,
        #     "round %s: start aggregate_evaluate",
        #     server_round,
        # )
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

        log(
            INFO,   
            "round %s: aggregate_evaluate is completed metrics_aggregated %s",
            server_round,
            metrics_aggregated,
        )

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
            "round %s: loss_aggregated in aggregate_cluster_evaluate is completed loss_aggregated: %s",
            server_round,
            loss_aggregated,
        )

        # Aggregate custom metrics if aggregation fn was provided
        # カスタムメトリクスの集約
        metrics_aggregated = {}
        if self.evaluate_metrics_cluster_aggregation_fn:
            # 
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            log(
                INFO,
                "round %s: eval_metrics in aggregate_cluster_evaluate %s",
                server_round,
                eval_metrics
            )
            metrics_aggregated = self.evaluate_metrics_cluster_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_cluster_aggregation_fn provided")

        log( # No
            INFO,   
            "round %s : metrics_aggregated in evaluate_metrics_cluster_aggregation_fn is completed, metrics_aggregated: %s",
            server_round,
            metrics_aggregated,
        )

        return loss_aggregated, metrics_aggregated