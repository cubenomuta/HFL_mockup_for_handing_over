import timeit
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import concurrent.futures
from logging import DEBUG, INFO
from flwr.common.logger import log
import ray

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    NDArrays,
    Parameters,
    Status,
    Scalar,
    ReconnectIns,
    DisconnectRes,
)
from models.driver import test
from models.knowledge_distillation import mutual_train
from utils.utils_model import load_model

from flwr.server.client_proxy import ClientProxy
from flwr.server import ClientManager, Server
from hfl_server_app.client_cluster_proxy import ClientClusterProxy
from flwr.server.strategy import Strategy
from flwr.server.server import fit_clients
from models.driver import evaluate_parameters, evaluate_parameters_by_client_data, evaluate_parameters_by_before_shuffle_fog_data
from models.knowledge_distillation import (
    distillation_multiple_parameters,
    distillation_parameters,
)
import torch
from flwr.common import (
    Code,
    EvaluateRes,
    Parameters,
    Scalar,
    Status,
    parameters_to_ndarrays,
)
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_federated_dataset, load_federated_client_dataset
from utils.utils_model import load_model
from models.base_model import Net


EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]


class FlowerRayClientClusterProxy(Server, ClientClusterProxy):
    def __init__(
        self,
        *,
        fid: str,
        clsid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ):
        super(FlowerRayClientClusterProxy, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        # Cluster configuration
        self.fid = fid
        self.clsid = clsid
        self.attribute = "cluster"
        self.config = config
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        return GetParametersRes(Status=Code.OK, parameters=self.parameters)

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        raise NotImplementedError("method get_parameters() is not implemented ")

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers
    
    def fit(
            self, 
            ins: FitIns,
            client_instructions: List[Tuple[ClientProxy, FitIns]],
            timeout: Optional[float]
        ) -> Tuple[FitRes, Dict[str, Parameters]]:
        # Fit configuration
        server_round: int = int(ins.config["server_round"])
        cluster_parameters: Parameters = ins.parameters
        client_parameters_dict = {
            client.cid: fit_ins.parameters
            for client, fit_ins in client_instructions
        }

        # DEBUG OK
        # log(
        #     INFO,
        #     "ClientCluster.fit() on fog fid=%s, clsid=%s: strategy sampled %s clients (out of %s)",
        #     self.fid,
        #     self.clsid,
        #     len(client_instructions),
        #     self._client_manager.num_available(),
        # )
        # Distillation from server model to client models
        self.set_max_workers(max_workers=len(client_instructions))
        distillation_from_server_config = {
            "teacher_model": self.config["server_model_name"],
            "student_model": self.config["client_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
            "clsid": self.clsid,
        }
        distillation_from_server_config.update(ins.config) # これが実行できてない
        results, failures = distillation_from_server(
            server_parameters=cluster_parameters,
            client_parameters_dict=client_parameters_dict,
            config=distillation_from_server_config,
            max_workers=self.max_workers,
        )
        # DEBUG OK
        # log(
        #     INFO,
        #     "distillation_from_server() on fog fid=%s, clsid=%s: received %s results and %s failures",
        #     self.fid,
        #     self.clsid,
        #     len(results),
        #     len(failures),
        # )
        if len(failures) > 0:
            raise ValueError("distillation is failed.")
        # 更新
        for cid, client_parameters in results:
            client_parameters_dict[cid] = client_parameters

        # FigInsのclient_parametersの更新
        # クライアントサンプリングはクラスタではなくフォグの方で行いたかったが仕方なく
        client_instructions = self.strategy.configure_partial_cient_fit( # ここのクライアントが全量じゃないからおかしい挙動を起こしている -> 修正済み
            server_round=server_round,
            pre_client_instructions=client_instructions,
            client_parameters_dict=client_parameters_dict,
            config=ins.config,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "fit() on fog fid=%s, clsid=%s: strategy sampled %s clients (out of %s)",
            self.fid,
            self.clsid,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        self.set_max_workers(max_workers=len(client_instructions))

        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=None,
        )
        # log( # OK
        #     INFO,
        #     "fit_clients() on fog fid=%s, clsid=%s: received %s results and %s failures",
        #     self.fid,
        #     self.clsid,
        #     len(results),
        #     len(failures),
        # )

        if len(failures) > 0:
            raise ValueError("Insufficient fit results from clients.")
        for client, fit_res in results:
            client_parameters_dict[client.cid] = fit_res.parameters

        # Distillation from multiple clients to server.
        distillation_from_clients_config = {
            "teacher_model": self.config["client_model_name"],
            "student_model": self.config["server_model_name"],
            "dataset_name": self.config["dataset_name"],
            "target_name": self.config["target_name"],
            "fid": self.fid,
            "clsid": self.clsid,
        }
        distillation_from_clients_config.update(ins.config)

        # なぜかはわからんが、cluster_parametersと出しわける
        client_cluster_parameters: Parameters = distillation_from_clients(
            teacher_parameters_list=[
                client_parameters
                for client_parameters in client_parameters_dict.values()
            ],
            student_parameters=cluster_parameters,
            config=distillation_from_clients_config,
        )
        # if type(client_cluster_parameters) == Parameters:
            # OK
            # log(
            #     INFO,
            #     "distillation_multiple_parameters() on fog fid=%s clsid=%s completed",
            #     self.fid,
            #     self.clsid,
            # )
            # log(
            #     INFO,
            #     "batch_size: %s, num_client :%s, num_examples: %s",
            #     ins.config["batch_size"],
            #     len(client_instructions),
            #     ins.config["batch_size"] * len(client_instructions),
            # )

        return FitRes(
            status=Status(Code.OK, message="success fit"),
            parameters=client_cluster_parameters,
            num_examples=int(ins.config["batch_size"] * len(client_instructions)), # batch_sizeは全クラスタ60
            metrics={},
        ), client_parameters_dict

    def evaluate(
        self,
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
        ins: EvaluateIns,
        config: Dict[str, Any],
        cluster_parameters: Parameters,
        max_workers: Optional[int],
        timeout: Optional[float],
    ) -> Tuple[EvaluateResultsAndFailures, EvaluateRes]:
        
        # DEBUG OK
        # log(
        #     INFO,
        #     "ClientCluster.evaluate() on fog fid=%s clsid=%s started",
        #     self.fid,
        #     self.clsid,
        # )

        # クラスタ評価用のconfigを更新
        evaluate_cluser_config= {
            "clsid": self.clsid
        }
        evaluate_cluser_config.update(config)

        # クライアント評価用のconfigを更新
        for client, ins in client_instructions:
            ins.config.update(evaluate_cluser_config)

        # クラスタモデルの評価
        cluster_model_evaluateres = self.evaluate_cluster_parameters(
            config=evaluate_cluser_config,
            cluster_parameters=cluster_parameters,
        )
        # DEBUG OK
        # log(
        #     INFO,
        #     "evaluate_cluster_parameters() on ClusterProxy.evaluate fid=%s clsid=%s completed, res.metrics=%s",
        #     self.fid,
        #     self.clsid,
        #     cluster_model_evaluateres.metrics,
        # )

        client_partition = self.target.split('_')[1]

        # クライアントモデルの評価
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            submitted_fs = {
                executor.submit(evaluate_client_parameters, client_proxy, ins, client_partition, timeout)
                for client_proxy, ins in client_instructions
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # DEBUG OK?
        # log(
        #     INFO,
        #     "evaluate client in cluster() on ClusterProxy.evaluate fid=%s clsid=%s completed",
        #     self.fid,
        #     self.clsid,
        # )

        # Gather results
        results: List[Tuple[ClientProxy, EvaluateRes]] = []
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_evaluate(
                future=future, results=results, failures=failures
            )
        # DEBUG OK
        # log( 
        #     INFO,
        #     "evaluate_client() fid: %s, clsid: %s received %s results and %s failures(out of %s)",
        #     self.fid,
        #     self.clsid,
        #     len(results),
        #     len(failures),
        #     len(client_instructions)
        # )
        return results, failures, cluster_model_evaluateres
    
    def evaluate_cluster_parameters(
        self,
        config: Dict[str, Scalar],
        cluster_parameters: Parameters,
    ):
        testset =  load_federated_client_dataset(
            dataset_name=config["dataset_name"],
            id=config["fid"],
            train=False,
            target=config["target_name"],
            attribute="fog",
        )
        # log( # OK
        #     INFO,
        #     "testset: %s",
        #     len(testset),
        # )
        # model configuration
        dataset_config = configure_dataset(
            dataset_name=config["dataset_name"],
            target=config["target_name"],
        )
        net: Net = load_model(
            name=config["client_model_name"],
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
        net.set_weights(parameters_to_ndarrays(cluster_parameters))
        # DEBUG OK
        # log(INFO,
        #     "net: %s. model_name: %s, iput_spec: %s, out_dims: %s",
        #     net,
        #     config["client_model_name"],
        #     dataset_config["input_spec"],
        #     dataset_config["out_dims"],
        # )

        # test configuration
        # 1000
        # batch_size: int = int(config["batch_size"])
        # クライアントが1のクラスタに合わせて10 or len(testset)
        batch_size: int = 10
        # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        testloader = DataLoader(
            dataset=testset,
            batch_size=batch_size,
            shuffle=False,
        )
        # DEBUG OK
        # log(
        #     INFO,
        #     "testloader: %s",
        #     testloader,
        # )
        log
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DEBUG No
        # log(
        #     INFO,
        #     "Before test() fid=%s clsid=%s testset=%s, batch_size=%s, device=%s",
        #     self.fid,
        #     self.clsid,
        #     len(testset),
        #     batch_size,
        #     device,
        # )

        # return {"loss": loss, "acc": acc}
        res = test(net=net, testloader=testloader, device=device)
        # DEBUG No
        # log(
        #     INFO,
        #     "After test() on ClusterProxy.evaluate fid=%s clsid=%s completed",
        # )
        result = cast(Dict[str, Scalar], res)
        metrics = {
            "acc": float(result["acc"]),
            "loss": float(result["loss"]),
            "fid": int(self.fid),
            "clsid": int(self.clsid),
        }
        # DEBUG No
        # log(
        #     INFO,
        #     "result of metrics fid=%s clsid=%s: accuracy=%s, loss=%s",
        #     self.fid,
        #     self.clsid,
        #     metrics["acc"],
        #     metrics["loss"],
        # )
        return EvaluateRes(
            status=Status(Code.OK, message="Success evaluate"),
            loss=float(result["loss"]),
            num_examples=len(testset),
            metrics=metrics,
        )
    
    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        return DisconnectRes(reason="Nothing to do here. (yet)")


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
    # DEBUG
    # log(
    #     INFO,
    #     "distillation_from_clients() on fog fid=%s clsid=%s started",
    #     config["fid"],
    #     config["clsid"],
    # )
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

# この関数をClientClusterProxy.evaluate()に移動している

# def evaluate_clients_parameters(
#     client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
#     max_workers: Optional[int],
#     timeout: Optional[float],
# ) -> EvaluateResultsAndFailures:
#     """Evaluate client parameters currently on all selected clinets"""
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         submitted_fs = {
#             executor.submit(evaluate_client_parameters, client_proxy, ins, timeout)
#             for client_proxy, ins in client_instructions
#         }
#         finished_fs, _ = concurrent.futures.wait(
#             fs=submitted_fs,
#             timeout=None,  # Handled in the respective communication stack
#         )

#     # Gather results
#     results: List[Tuple[ClientProxy, EvaluateRes]] = []
#     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
#     for future in finished_fs:
#         _handle_finished_future_after_evaluate(
#             future=future, results=results, failures=failures
#         )
#     return results, failures


def evaluate_client_parameters(
    client: ClientProxy,
    ins: EvaluateIns,
    client_partition: str,
    timeout: Optional[float] = None,
):
    ins.config["cid"] = client.cid
    parameters_ref = ray.put(ins.parameters)
    config_ref = ray.put(ins.config)

    # targetに応じてテストに使用するデータを出しわける
    if client_partition == "iid": # フォグのtestデータで評価
        # log(
        #     INFO,
        #     "evaluate_parameters.remote() is called",
        # )
        future_evaluate_res = evaluate_parameters.remote(
            parameters_ref,
            config_ref,
        )
    elif "part-noniid" in client_partition: # シャッフル前のフォグのtestデータで評価
        # log(
        #     INFO,
        #     "evaluate_parameters_by_before_shuffle_fog_data.remote() is called",
        # )
        future_evaluate_res = evaluate_parameters_by_before_shuffle_fog_data.remote(
            parameters_ref,
            config_ref,
        )
    else: # クライアントのtestデータで評価
        # log(
        #     INFO,
        #     "evaluate_parameters_by_client_data.remote() is called",
        # )
        future_evaluate_res = evaluate_parameters_by_client_data.remote(
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
    # DEBUG OK
    # log(
    #     INFO,
    #     "metrics result of evaluate_client_parameters: cid=%s, acc=%s, loss=%s",
    #     metrics["cid"],
    #     metrics["acc"],
    #     metrics["loss"],
    # )
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