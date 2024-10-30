import concurrent.futures
from logging import DEBUG, INFO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import ray
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
from models.driver import evaluate_parameters, evaluate_parameters_by_client_data, evaluate_parameters_by_before_shuffle_fog_data
from models.knowledge_distillation import (
    distillation_multiple_parameters,
    distillation_parameters,
    distillation_multiple_parameters_by_consensus,
    distillation_multiple_parameters_with_extra_term,
)
from simulation_app.ray_transport.ray_client_proxy import RayClientProxy

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
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # client configurations
        self.cids = [
            str(x + int(self.fid) * self.config["num_clients"])
            for x in range(self.config["num_clients"])
        ]
        # log(
        #     INFO,
        #     "Fog fid=%s: self.cids %s",
        #     self.fid,
        #     self.cids
        # )
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
        # parameters_ref = ray.put(ins.parameters)
        # evaluate_config_ref = ray.put(evaluate_config)

        # future_evaluate_res = evaluate_parameters.remote(
        #     parameters_ref, evaluate_config_ref
        # )

        # try:
        #     res = ray.get(future_evaluate_res, timeout=timeout)
        # except Exception as ex:
        #     log(DEBUG, ex)
        #     raise ex
        # result = cast(Dict[str, Scalar], res)
        # # release ObjectRefs in object_store_memory
        # ray.internal.free(parameters_ref)
        # ray.internal.free(evaluate_config_ref)
        # ray.internal.free(future_evaluate_res)

        # # Assing the same model results for belonging clients
        # metrics = {
        #     "accuracy": {int(cid): result["acc"] for cid in self.cids},
        #     "loss": {int(cid): result["loss"] for cid in self.cids},
        # }

        server_round: int = int(ins.config["server_round"])

        client_instructions = self.strategy.configure_client_evaluate(
            server_round=server_round,
            client_parameters=ins.parameters,
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

        client_partition = self.target.split('_')[1]

        results, failures = evaluate_clients_parameters(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            client_partition=client_partition,
            timeout=None,
        )
        log(
            INFO,
            "Fed Fog evaluate() on fog fid=%s: received %s results and %s failures",
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

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        return DisconnectRes(reason="Nothing to do here. (yet)")


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

        client_partition = self.target.split('_')[1]

        results, failures = evaluate_clients_parameters(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            client_partition=client_partition,
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
    if config["kd_from_clients"] == "normal": 
        future_distillation_res = distillation_multiple_parameters.remote(
            teacher_parameters_list_ref,
            student_parameters_ref,
            config_ref,
        )
    elif config["kd_from_clients"] == "by_consensus":
        future_distillation_res = distillation_multiple_parameters_by_consensus.remote(
            teacher_parameters_list_ref,
            student_parameters_ref,
            config_ref,
        )
    elif config["kd_from_clients"] == "with_extra_term":
        future_distillation_res = distillation_multiple_parameters_with_extra_term.remote(
            teacher_parameters_list_ref,
            student_parameters_ref,
            config_ref,
        )
    else:
        raise ValueError("Invalid knowledge distillation method.")
    
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
    client_partition: str,
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate client parameters currently on all selected clinets"""
    # for client_proxy, ins in client_instructions:
    #     log( # 連番だった
    #         INFO,
    #         "evaluate_clients_parameters() on client_proxy.cid=%s ins.config['cid']=%s",
    #         client_proxy.cid,
    #         ins.config["cid"] # エラー出る
    #     )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client_parameters, client_proxy, ins, client_partition, timeout)
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
    client_partition: str,
    timeout: Optional[float] = None,
):
    ins.config["cid"] = client.cid
    parameters_ref = ray.put(ins.parameters)
    config_ref = ray.put(ins.config)
    if client_partition == "iid": # フォグのtestデータで評価
        log(
            INFO,
            "evaluate_parameters.remote() is called",
        )
        future_evaluate_res = evaluate_parameters.remote(
            parameters_ref,
            config_ref,
        )
    elif "part-noniid" in client_partition: # シャッフル前のフォグのtestデータで評価
        log(
            INFO,
            "evaluate_parameters_by_before_shuffle_fog_data.remote() is called",
        )
        future_evaluate_res = evaluate_parameters_by_before_shuffle_fog_data.remote(
            parameters_ref,
            config_ref,
        )
    else: # クライアントのtestデータで評価
        log(
            INFO,
            "evaluate_parameters_by_client_data.remote() is called",
        )
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
