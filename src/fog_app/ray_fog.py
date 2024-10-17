import concurrent.futures
from logging import DEBUG, INFO, WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import ray
import json
from flwr.client import Client
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
    Status,
)
from flwr.common.logger import log
from flwr.server import ClientManager, Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients
from flwr.server.strategy import Strategy
from hfl_server_app.fog_proxy import FogProxy
from hfl_server_app.client_cluster_proxy import ClientClusterProxy
from hfl_server_app.client_cluster_manager import SimpleClientClusterManager
from models.driver import evaluate_parameters, evaluate_parameters_by_client_data, evaluate_parameters_by_before_shuffle_fog_data
from models.knowledge_distillation import (
    distillation_multiple_parameters,
    distillation_parameters,
)
from simulation_app.ray_transport.ray_client_proxy import RayClientProxy

ClusterFitResultsAndFailures = Tuple[
    List[Tuple[ClientClusterProxy, FitRes]],
    List[Union[Tuple[ClientClusterProxy, FitRes], BaseException]],
]
FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]


class RayFlowerFogProxy(Server, FogProxy):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        client_fn: Callable[[str], Client],
        strategy: Optional[Strategy] = None,
    ):
        super(RayFlowerFogProxy, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        # Fog configuration
        self.fid = fid
        self.attribute = "fog"
        self.config = config

        # client configurations
        self.cids = [
            str(x + int(self.fid) * self.config["num_clients"])
            for x in range(self.config["num_clients"])
        ]
        for cid in self.cids:
            client_proxy = RayClientProxy(
                cid=cid, client_fn=client_fn, resources={"num_cpus": 1}
            )
            self.client_manager().register(client=client_proxy)

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        return GetParametersRes(Status=Code.OK, parameters=self.parameters)

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        raise NotImplementedError("method get_parameters() is not implemented ")

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
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

        # # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_prime, metrics_aggregated = aggregated_result

        return FitRes(
            status=Status(Code.OK, message="success fit"),
            parameters=parameters_prime,
            num_examples=int(ins.config["batch_size"]),
            metrics=metrics_aggregated,
        )

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        # evaluate configuration
        evaluate_config = {
            "client_model_name": self.config["client_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
        }
        evaluate_config.update(ins.config)
        parameters_ref = ray.put(ins.parameters)
        evaluate_config_ref = ray.put(evaluate_config)

        future_evaluate_res = evaluate_parameters.remote(
            parameters_ref, evaluate_config_ref
        )

        try:
            res = ray.get(future_evaluate_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        result = cast(Dict[str, Scalar], res)
        # release ObjectRefs in object_store_memory
        ray.internal.free(parameters_ref)
        ray.internal.free(evaluate_config_ref)
        ray.internal.free(future_evaluate_res)

        # Assing the same model results for belonging clients
        metrics = {
            "accuracy": {int(cid): result["acc"] for cid in self.cids},
            "loss": {int(cid): result["loss"] for cid in self.cids},
        }

        return EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(result["loss"]),
            num_examples=int(result["num_examples"]),
            metrics=metrics,
        )

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        return DisconnectRes(reason="Nothing to do here. (yet)")

class RayFlowewrClusterDMLFogProxy(RayFlowerFogProxy):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        client_cluster_fn: Callable[[str], ClientClusterProxy],
        client_fn: Callable[[str], Client],
        strategy: Optional[Strategy] = None,
        client_init_parameters: Optional[Parameters] = None,
    ):
        super(RayFlowewrClusterDMLFogProxy, self).__init__(
            fid=fid,
            config=config,
            client_manager=client_manager,
            client_fn=client_fn,
            strategy=strategy,
        )
        # クライアントパラメータの初期化はClientClusterProxyで行うため一旦コメントアウト
        self.client_parameters_dict: Dict[str, Parameters] = {
            str(cid): client_init_parameters for cid in self.cids
        }
        #クラスタモデルのパラメータ管理
        self.cluster_parameters_dict: Dict[str, Parameters] = {}

        # clsid: [cid] の辞書
        self.cluster_dict: Dict[str, List[Scalar]] = {}
        # 一旦固定値いれる
        file_path = "./data/FashionMNIST/partitions/noniid-label2_part-noniid/client/clustered_client_list.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
        for clsid, cids in data[self.fid].items():
            self.cluster_dict[clsid] = cids
            # self.cluster_parameters_dict[clsid] = self.parameters

        # ClientClusterManagerの作成
        self.client_cluster_manager = SimpleClientClusterManager()

        self.clsids = [
            str(x)
            for x in range(len(self.cluster_dict))
        ]

        # ClientClusterProxyのインスタンス作成
        for clsid in self.clsids:
            client_cluster_proxy = client_cluster_fn(
                fid=self.fid, 
                clsid=clsid,
                client_manager=self.client_manager,
            )
            self.client_cluster_manager.register(client_cluster_proxy)

    
    def fit(self, 
            ins: FitIns, 
            timeout: Optional[float]
        ) -> ClusterFitResultsAndFailures:

        server_round: int = int(ins.config["server_round"])
        parameters = ins.parameters # グローバルモデル

        #  Dict[str("clsid"), Tuple[ClientClusterProxy, FitIns]]
        clusters_instructions = self.strategy.configure_cluster_fit(
            server_round=server_round,
            parameters=parameters,
            client_cluster_manager=self.client_cluster_manager,
        )

        # List[Tuple[ClientProxy, FitIns]]
        clients_list_instructions = self.strategy.configure_fit(
            server_round=server_round,
            client_parameters_dict=self.client_parameters_dict,
            config=ins.config,
            client_manager=self._client_manager,
        )

        clients_per_cluster_instructions = {clsid: [] for clsid in self.cluster_dict.keys()}

        # clients_list_instructions と cluster_dict を照合
        # クラスタごとにList[Tuple[ClientProxy, FitIns]]を持たせる辞書を作成
        for client, fit_ins in clients_list_instructions:
            for clsid, cids in self.cluster_dict.items():
                if int(client.cid) in cids: # 型注意
                    clients_per_cluster_instructions[clsid].append((client, fit_ins))

        self.set_max_workers(max_workers=len(self.clsids))

        # ray worker killed対策
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     submitted_fs = {
        #         executor.submit(
        #             self.fit_client_cluster, 
        #             clusters_instructions[clsid], # Tuple[ClientClusterProxy, FitIns]
        #             clients_instructions, # List[Tuple[ClientProxy, FitIns]]
        #             timeout
        #         )
        #         for clsid, clients_instructions in clients_per_cluster_instructions.items()
        #     }
        #     finished_fs, _ = concurrent.futures.wait(
        #       fs=submitted_fs,
        #       timeout=None,  # Handled in the respective communication stack
        #     )

        results: List[Tuple[ClientClusterProxy, FitRes]] = []
        failures: List[Union[Tuple[ClientClusterProxy, FitRes], BaseException]] = []

        for clsid, clients_instructions in clients_per_cluster_instructions.items():
            try:
                pre_result: Tuple[ClientClusterProxy, Tuple[FitRes, Dict[str, Parameters]]] = self.fit_client_cluster(
                    clusters_instructions[clsid],  # Tuple[ClientClusterProxy, FitIns]
                    clients_instructions,          # List[Tuple[ClientProxy, FitIns]]
                    timeout
                )

                # 結果からクラスターと結果情報を取得
                cluster, res, client_parameters_dict = pre_result

                result = (cluster, res)

                # クラスターパラメータの更新
                self.cluster_parameters_dict[cluster.clsid] = res.parameters

                # クライアントパラメータの更新
                for cid, client_parameters in client_parameters_dict.items():
                    self.client_parameters_dict[cid] = client_parameters

                # 結果のステータスコードを確認
                if res.status.code == Code.OK:
                    results.append(result)
                else:
                    # ステータスコードがOKでない場合は失敗リストに追加
                    failures.append((cluster, res))

            except Exception as e:
                # 例外発生時は失敗リストに追加
                failures.append(e)

        # for future in finished_fs:
        #     self._handle_finished_future_after_fit(
        #         future=future, results=results, failures=failures
        #     )
        return results, failures
    
    def _handle_finished_future_after_fit(
        self,
        future: concurrent.futures.Future,  # type: ignore
        results: List[Tuple[ClientClusterProxy, FitRes]],
        failures: List[Union[Tuple[ClientClusterProxy, FitRes], BaseException]],
    ) -> None:
        """Convert finished future into either a result or a failure."""
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
            return

        # Successfully received a result from a fog
        pre_result: Tuple[ClientClusterProxy, Tuple[FitRes, Dict[str, Parameters]]] = future.result()
        cluster, res, client_parameters_dict = pre_result

        result = (cluster, res)

        # Update cluster parameters
        self.cluster_parameters_dict[cluster.clsid] = res.parameters

        # Update client parameters
        for cid, client_parameters in client_parameters_dict.items():
            self.client_parameters_dict[cid] = client_parameters

        # Check result status code
        if res.status.code == Code.OK:
            results.append(result)
            return

        # Not successful, fog returned a result where the status code is not OK
        failures.append(result)

    def fit_client_cluster(
        self,
        cluster_instructions: Tuple[ClientClusterProxy, FitIns],
        clients_instructions: List[Tuple[ClientProxy, FitIns]],
        timeout: Optional[float] = None,
    ) -> Tuple[ClientClusterProxy, FitRes, Dict[str, Parameters]]:
        """Refine parameters on a single cluster."""
        cluster, ins = cluster_instructions
        fit_res, client_parameters_dict = cluster.fit(ins=ins, client_instructions=clients_instructions, timeout=timeout)
        return cluster, fit_res, client_parameters_dict
    
    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> Tuple[EvaluateRes, EvaluateRes]:
        # Evaluate configuration
        server_round: int = int(ins.config["server_round"])
        evaluate_config = {
            "client_model_name": self.config["client_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
        }
        # ins.config に含まれているキーと値が evaluate_config に追加
        # clsidを追加したconfigは各クラスタで作成する
        evaluate_config.update(ins.config)

        clusters_instructions = self.strategy.configure_cluster_evaluate(
            server_round=server_round,
            cluster_parameters_dict=self.cluster_parameters_dict,
            config=evaluate_config,
            client_cluster_manager=self.client_cluster_manager,
        )

        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            client_parameters_dict=self.client_parameters_dict,
            config=evaluate_config,
            client_manager=self._client_manager,
        )

        clients_per_cluster_instructions = {clsid: [] for clsid in self.cluster_dict.keys()}
        # client_instructions と cluster_dict を照合
        # クラスタごとにList[Tuple[ClientProxy, FitIns]]を持たせる辞書を作成 -> clients_per_cluster_instructions
        for client, evaluate_ins in client_instructions:
            for clsid, cids in self.cluster_dict.items():
                if int(client.cid) in cids:
                    clients_per_cluster_instructions[clsid].append((client, evaluate_ins))
                    # ここなんかおかしくね　全部のclsidが追加されるからここでやるべきではない
                    # evaluate_cluster_config = {
                    #     "clsid": clsid,
                    # }
                    # evaluate_cluster_config.update(evaluate_config)
                    # EvaluateInsにclsidを追加
                    # evaluate_ins.config = evaluate_cluster_config

        self.set_max_workers(max_workers=len(client_instructions))

        # 並列実行
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     submitted_fs = {
        #         executor.submit(
        #             self.evaluate_client_cluster, 
        #             clusters_instructions[clsid], # Tuple[ClientClusterProxy, FitIns]
        #             client_instructions, # List[Tuple[ClientProxy, FitIns]]
        #             evaluate_config,
        #             self.cluster_parameters_dict[clsid],
        #             timeout
        #         )
        #         for clsid, client_instructions in clients_per_cluster_instructions.items()
        #     }
        #     finished_fs, _ = concurrent.futures.wait(
        #       fs=submitted_fs,
        #       timeout=None,  # Handled in the respective communication stack
        #     )

        # シリアル実行（ray worker killed対策）
        # すべてのクラスタの評価結果を集約
        cluster_results: List[Tuple[ClientClusterProxy, EvaluateRes]] = []
        cluster_failures: List[Union[Tuple[ClientClusterProxy, EvaluateRes], BaseException]] = []
        results: List[Tuple[ClientProxy, EvaluateRes]] = []
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
        # for future in finished_fs:
        #     self._handle_finished_future_after_cluster_evaluate(
        #         future=future, cluster_results=cluster_results, cluster_failures=cluster_failures, results=results, failures=failures
        #     )

        for clsid, client_instructions in clients_per_cluster_instructions.items():
            try:
                # `evaluate_client_cluster`のシングルスレッド実行
                res: Tuple[
                    ClientClusterProxy, 
                    List[Tuple[ClientProxy, EvaluateRes]],
                    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
                    EvaluateRes
                ] = self.evaluate_client_cluster(
                    clusters_instructions[clsid],  # Tuple[ClientClusterProxy, FitIns]
                    client_instructions,           # List[Tuple[ClientProxy, FitIns]]
                    evaluate_config,
                    self.cluster_parameters_dict[clsid],
                    timeout
                )

                # 結果の取得と処理
                cluster, cls_results, cls_failures, cluster_model_evaluateres = res

                # クラスターの評価結果のステータスを確認
                if cluster_model_evaluateres.status.code == Code.OK:
                    cluster_results.append((cluster, cluster_model_evaluateres))
                else:
                    cluster_failures.append((cluster, cluster_model_evaluateres))

                # 各クラスター内のクライアント結果を処理
                for cls_result in cls_results:
                    cluster, cls_res = cls_result
                    if cls_res.status.code == Code.OK:
                        results.append(cls_result)
                    else:
                        failures.append(cls_result)

                # クラスター内の失敗を処理
                for failure in cls_failures:
                    failures.append(failure)
            
            except Exception as e:
                # `evaluate_client_cluster`実行中に例外が発生した場合は失敗リストに追加
                failures.append(e)

        # Aggregate evaluation results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        cluster_aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_cluster_evaluate(server_round, cluster_results, cluster_failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        cluster_loss_aggregated, cluster_metrics_aggregated = cluster_aggregated_result

        # 各クラスタのバッチサイズを算出
        # fog_batch_size = int(ins.config["batch_size"])
        
        return EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(loss_aggregated),
            num_examples=int(ins.config["batch_size"]),
            metrics=metrics_aggregated,
        ), EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(cluster_loss_aggregated),
            num_examples=int(ins.config["batch_size"]), # 一旦
            metrics=cluster_metrics_aggregated,
        )
    
    def evaluate_client_cluster(
        self,
        cluster_instructions: Tuple[ClientClusterProxy, EvaluateIns],
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
        config: Dict[str, Any],
        cluster_parameters: Parameters,
        timeout: Optional[float] = None,
    )-> Tuple[
            ClientClusterProxy, 
            List[Tuple[ClientProxy, EvaluateRes]],
            List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            EvaluateRes
        ]:
        cluster, ins = cluster_instructions
        # results: List[Tuple[ClientProxy, EvaluateRes]]
        # failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
        # cluster_result: Tuple[ClientClusterProxy, EvaluateRes]
        # log( # OK
        #     INFO,
        #     "start evaluate_client_cluster() on fog fid=%s clsid=%s",
        #     self.fid,
        #     cluster.clsid,
        # )
        results, failures, cluster_model_evaluateres= cluster.evaluate(
            client_instructions=client_instructions,
            ins=ins,
            config=config,
            cluster_parameters=cluster_parameters,
            max_workers=self.max_workers,
            timeout=timeout
        )
        return cluster, results, failures, cluster_model_evaluateres
    
    def _handle_finished_future_after_cluster_evaluate(
        self,
        future: concurrent.futures.Future,  # type: ignore
        cluster_results: List[Tuple[ClientClusterProxy, EvaluateRes]],
        cluster_failures: List[Union[Tuple[ClientClusterProxy, EvaluateRes], BaseException]],
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> None:
        
        # Check if there was an exception
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
            return
        
        # Successfully received a result from a client
        res: Tuple[
            ClientClusterProxy, 
            List[Tuple[ClientProxy, EvaluateRes]],
            List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            EvaluateRes
        ] = future.result()

        cluster, cls_results, cls_failures, cluster_model_evaluateres = res

        if cluster_model_evaluateres.status.code == Code.OK:
            cluster_results.append((cluster, cluster_model_evaluateres))
        else:
            cluster_failures.append((cluster, cluster_model_evaluateres))

        # log(
        #     INFO,
        #     "fid %s: the number of cluster_results: %s, metrics_keys: %s",
        #     self.fid,
        #     len(cluster_results),
        #     cluster_model_evaluateres.metrics.keys()
        # )
        for cls_result in cls_results:
            cluster, cls_res = cls_result
            if cls_res.status.code == Code.OK:
                results.append(cls_result)
            else:
                failures.append(cls_result)

        # log(
        #     INFO,
        #     "fid %s: the number of results: %s",
        #     self.fid,
        #     len(results),
        # )
            
        for failure in cls_failures:
            failures.append(failure)

        return

class RayFlowerDMLFogProxy(RayFlowerFogProxy):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        client_fn: Callable[[str], Client],
        strategy: Optional[Strategy] = None,
        client_init_parameters: Optional[Parameters] = None,
    ):
        super(RayFlowerDMLFogProxy, self).__init__(
            fid=fid,
            config=config,
            client_manager=client_manager,
            client_fn=client_fn,
            strategy=strategy,
        )
        self.client_parameters_dict: Dict[str, Parameters] = {
            str(cid): client_init_parameters for cid in self.cids
        }

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        # Fit configuration
        server_round: int = int(ins.config["server_round"])
        server_parameters: Parameters = ins.parameters

        # Distillation from server model to client models
        self.set_max_workers(max_workers=len(self.cids))
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
        # if not client_instructions:
        #     log(INFO, "fit_round %s: no clients selected, cancel", server_round)
        #     return None
        # log(
        #     INFO,
        #     "fit() on fog fid=%s: strategy sampled %s clients (out of %s)",
        #     self.fid,
        #     len(client_instructions),
        #     self._client_manager.num_available(),
        # )
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
            num_examples=int(ins.config["batch_size"]),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        # Evaluate configuration
        server_round: int = int(ins.config["server_round"])
        evaluate_config = {
            "client_model_name": self.config["client_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
        }
        evaluate_config.update(ins.config)

        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            client_parameters_dict=self.client_parameters_dict,
            config=evaluate_config,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "evaluate() on fog fid=%s: strategy sampled %s clients (out of %s)",
            self.fid,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        self.set_max_workers(max_workers=len(client_instructions))

        results, failures = evaluate_clients_parameters(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=None,
        )
        log(
            INFO,
            "evaluate() on fog fid=%s: received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )
        # Aggregate evaluation results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(loss_aggregated),
            num_examples=int(ins.config["batch_size"]),
            metrics=metrics_aggregated,
        )


def distillation_from_server(
    server_parameters: Parameters,
    client_parameters_dict: Dict[str, Parameters],
    config: Dict[str, Any],
    max_workers: Optional[int],
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(
                distillation, cid, server_parameters, client_parameters, config
            )
            for cid, client_parameters in client_parameters_dict.items()
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    # Gather results
    results: List[Tuple[str, Parameters]] = []
    failures: List[Union[Tuple[str, Parameters], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_distillation(
            future=future,
            results=results,
            failures=failures,
        )
    return results, failures


def _handle_finished_future_after_distillation(
    future: concurrent.futures.Future,
    results: List[Tuple[str, Parameters]],
    failures: List[Union[Tuple[str, Parameters], BaseException]],
) -> None:
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully receieved a result from a client
    result: Tuple[str, Parameters] = future.result()
    _, res = result

    # Check result type
    if type(res) == Parameters:
        results.append(result)
        return

    failures.append(result)


def distillation(
    cid: str,
    teacher_parameters: Parameters,
    student_parameters: Parameters,
    config: Dict[str, Any],
):
    teacher_parameters_ref = ray.put(teacher_parameters)
    student_parameters_ref = ray.put(student_parameters)
    config_ref = ray.put(config)
    future_distillation_res = distillation_parameters.remote(
        teacher_parameters_ref,
        student_parameters_ref,
        config_ref,
    )
    try:
        res = ray.get(future_distillation_res, timeout=None)
    except Exception as ex:
        log(DEBUG, ex)
        raise ex
    ray.internal.free(teacher_parameters_ref)
    ray.internal.free(student_parameters_ref)
    ray.internal.free(config_ref)
    ray.internal.free(future_distillation_res)
    return cid, cast(Parameters, res)


def distillation_from_clients(
    teacher_parameters_list: List[Parameters],
    student_parameters: Parameters,
    config: Dict[str, Any],
):
    teacher_parameters_list_ref = ray.put(teacher_parameters_list)
    student_parameters_ref = ray.put(student_parameters)
    config_ref = ray.put(config)
    future_distillation_res = distillation_multiple_parameters.remote(
        teacher_parameters_list_ref,
        student_parameters_ref,
        config_ref,
    )
    try:
        res = ray.get(future_distillation_res, timeout=None)
    except Exception as ex:
        log(DEBUG, ex)
        raise ex
    ray.internal.free(teacher_parameters_list_ref)
    ray.internal.free(student_parameters_ref)
    ray.internal.free(config_ref)
    ray.internal.free(future_distillation_res)
    return cast(Parameters, res)


def evaluate_clients_parameters(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate client parameters currently on all selected clinets"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client_parameters, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client_parameters(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float] = None,
):
    ins.config["cid"] = client.cid
    parameters_ref = ray.put(ins.parameters)
    config_ref = ray.put(ins.config)
    # future_evaluate_res = evaluate_parameters_by_client_data.remote(
    #     parameters_ref,
    #     config_ref,
    # )
    future_evaluate_res = evaluate_parameters_by_before_shuffle_fog_data.remote(
        parameters_ref,
        config_ref,
    )
    try:
        res = ray.get(future_evaluate_res, timeout=timeout)
    except Exception as ex:
        log(DEBUG, ex)
        raise ex
    result = cast(Dict[str, Scalar], res)
    metrics = {
        "cid": int(client.cid),
        "acc": result["acc"],
        "loss": result["loss"],
    }
    evaluate_res = EvaluateRes(
        status=Status(Code.OK, message="Success evaluate_paremeter()"),
        loss=result["loss"],
        num_examples=result["num_examples"],
        metrics=metrics,
    )
    ray.internal.free(parameters_ref)
    ray.internal.free(config_ref)
    ray.internal.free(future_evaluate_res)
    return (client, evaluate_res)


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)