import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from pathlib import Path
import json
import os
import time

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.server import History
from server_app.custom_history import CustomHistory

from .fog_manager import FogManager
from .fog_proxy import FogProxy
from .strategy.fedavg import FedAvg
from .strategy.strategy import Strategy

FitResultsAndFailures = Tuple[
    List[Tuple[FogProxy, FitRes]],
    List[Union[Tuple[FogProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[FogProxy, EvaluateRes]],
    List[Union[Tuple[FogProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[FogProxy, DisconnectRes]],
    List[Union[Tuple[FogProxy, DisconnectRes], BaseException]],
]


class HFLServer:
    """Flower server."""

    def __init__(
        self, *, fog_manager: FogManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._fog_manager: FogManager = fog_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.fogs_time_result: Dict[str, List[float]] = {}
        self.clients_time_result: Dict[str, List[float]] = {}
        self.server_time_result: List[float] = []

    def make_time_json(self, save_dir: str) -> None:
        """"Make Fog & Client Time Json"""
        if save_dir:
            time_dir = Path(save_dir) / "time"
            time_dir.mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / "time" / "fog_train_time.json"
            with open(save_path, "w") as f:
                json.dump(self.fogs_time_result, f)
            save_path = Path(save_dir) / "time" / "client_train_time.json"
            with open(save_path, "w") as f:
                json.dump(self.clients_time_result, f)
            save_path = Path(save_dir) / "time" / "server_train_time.json"
            with open(save_path, "w") as f:
                json.dump(self.server_time_result, f)  
        return

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def fog_manager(self) -> FogManager:
        """Return FogManager."""
        return self._fog_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = CustomHistory()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available fogs
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of fogs."""

        # Get fogs and their respective instructions from strategy
        fog_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            fog_manager=self._fog_manager,
        )
        if not fog_instructions:
            log(INFO, "evaluate_round %s: no fogs selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s fogs (out of %s)",
            server_round,
            len(fog_instructions),
            self._fog_manager.num_available(),
        )
        self.set_max_workers(len(fog_instructions))

        # Collect `evaluate` results from all fogs participating in this round
        results, failures = evaluate_fogs(
            fog_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
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
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get fogs and their respective instructions from strategy
        fog_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            fog_manager=self._fog_manager,
        )

        if not fog_instructions:
            log(INFO, "fit_round %s: no fogs selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s fogs (out of %s)",
            server_round,
            len(fog_instructions),
            self._fog_manager.num_available(),
        )
        self.set_max_workers(len(fog_instructions))

        # Collect `fit` results from all fogs participating in this round
        results, failures, fog_time_results, client_time_results = fit_fogs(
            fog_instructions=fog_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        for fid, fog_time_result in fog_time_results:
            if fid not in self.fogs_time_result:
                self.fogs_time_result[fid] = []
            self.fogs_time_result[fid].append(fog_time_result)

        for cid, client_time_result in client_time_results:
            if cid not in self.clients_time_result:
                self.clients_time_result[cid] = []
            self.clients_time_result[cid].append(client_time_result)

        start_time = time.perf_counter()
        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        end_time = time.perf_counter()
        server_time = end_time - start_time
        self.server_time_result.append(server_time)

        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, ()

    def disconnect_all_fogs(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all fogs."""
        all_fogs = self._fog_manager.all()
        fogs = [all_fogs[k] for k in all_fogs.keys()]
        instruction = ReconnectIns(seconds=None)
        fog_instructions = [(fog_proxy, instruction) for fog_proxy in fogs]
        _ = reconnect_fogs(
            fog_instructions=fog_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available fogs."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            fog_manager=self._fog_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the fogs
        log(INFO, "Requesting initial parameters from one random fog")
        random_fog = self._fog_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_fog.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random fog")
        return get_parameters_res.parameters


def reconnect_fogs(
    fog_instructions: List[Tuple[FogProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct fogs to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_fog, fog_proxy, ins, timeout)
            for fog_proxy, ins in fog_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[FogProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[FogProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_fog(
    fog: FogProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[FogProxy, DisconnectRes]:
    """Instruct fog to disconnect and (optionally) reconnect later."""
    disconnect = fog.reconnect(
        reconnect,
        timeout=timeout,
    )
    return fog, disconnect


def fit_fogs(
    fog_instructions: List[Tuple[FogProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected fogs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_fog, fog_proxy, ins, timeout)
            for fog_proxy, ins in fog_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[FogProxy, FitRes]] = []
    failures: List[Union[Tuple[FogProxy, FitRes], BaseException]] = []
    client_time_results: List[Tuple[str, float]] = []
    fog_time_results: List[Tuple[str, float]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures, client_time_results=client_time_results, fog_time_results=fog_time_results
        )
    return results, failures, fog_time_results, client_time_results


def fit_fog(
    fog: FogProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[FogProxy, FitRes]:
    """Refine parameters on a single fog."""
    fit_res = fog.fit(ins, timeout=timeout)
    return fog, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[FogProxy, FitRes]],
    failures: List[Union[Tuple[FogProxy, FitRes], BaseException]],
    client_time_results: List[Tuple[str, float]],
    fog_time_results: List[Tuple[str, float]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a fog
    # result: Tuple[FogProxy, FitRes] = future.result()]
    # _, res = result
    pre_result: Tuple[FogProxy, FitRes] = future.result()
    fog_proxy, (res, client_fit_time, fog_comp_time) = pre_result
    result = (fog_proxy, res)

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        if client_fit_time:
            client_time_results.extend((client_fit_time))
        if fog_comp_time:
            fog_time_results.append((fog_proxy.fid, fog_comp_time))
        return

    # Not successful, fog returned a result where the status code is not OK
    failures.append(result)


def evaluate_fogs(
    fog_instructions: List[Tuple[FogProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected fogs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_fog, fog_proxy, ins, timeout)
            for fog_proxy, ins in fog_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[FogProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[FogProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_fog(
    fog: FogProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[FogProxy, EvaluateRes]:
    """Evaluate parameters on a single fog."""
    evaluate_res = fog.evaluate(ins, timeout=timeout)
    return fog, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[FogProxy, EvaluateRes]],
    failures: List[Union[Tuple[FogProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a fog
    result: Tuple[FogProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, fog returned a result where the status code is not OK
    failures.append(result)
