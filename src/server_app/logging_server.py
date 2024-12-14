import concurrent
import os
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
import time
from pathlib import Path
import json
import os

import torch
from flwr.common import Code, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
# from flwr.server.server import fit_clients
from flwr.server.strategy import Strategy
from models.base_model import Net

from .custom_history import CustomHistory

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class LoggingServer(Server):
    """
    Flower server implementation for system performance measurement.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        save_model: bool = False,
        save_dir: str = None,
        net: Net = None,
    ) -> None:
        super(LoggingServer, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        self.save_model = save_model
        self.server_time_result: Dict[str, List[float]] = {}
        self.server_time_result[0] = []
        self.clients_time_result: Dict[str, List[float]] = {}
        if self.save_model:
            assert net is not None
            assert save_dir is not None
            self.net = net
            self.save_dir = save_dir

    def make_time_json(self, save_dir: str) -> None:
        """"Make Fog & Client Time Json"""
        if save_dir:
            time_dir = Path(save_dir) / "time"
            time_dir.mkdir(parents=True, exist_ok=True)

            save_path = Path(save_dir) / "time" / "server_train_time.json"
            with open(save_path, "w") as f:
                json.dump(self.server_time_result, f)
            save_path = Path(save_dir) / "time" / "client_train_time.json"
            with open(save_path, "w") as f:
                json.dump(self.clients_time_result, f)

        return

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = CustomHistory()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1]
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                (
                    parameters_prime,
                    timestamps_cen,
                    _,
                    client_time_results,
                    server_comp_time,
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                timestamps_cen["fit_round"] = timeit.default_timer() - start_time

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
            # Evaluate model on a sample of available clients
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
            
            self.server_time_result[0].append(server_comp_time)
            for cid, result in client_time_results:
                if cid not in self.clients_time_result:
                    self.clients_time_result[cid] = []
                self.clients_time_result[cid].append(result)

        if self.save_model:
            weights = parameters_to_ndarrays(self.parameters)
            self.net.set_weights(weights)
            save_path = os.path.join(self.save_dir, "final_model.pth")
            torch.save(self.net.to("cpu").state_dict(), save_path)
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
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

        # Collect `fit` results from all clients participating in this round
        results, failures, client_time_results = fit_clients(
            client_instructions=client_instructions,
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

        server_comp_start_time = time.perf_counter()
        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        server_comp_end_time = time.perf_counter()
        server_comp_time = server_comp_end_time - server_comp_start_time

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures), client_time_results, server_comp_time

def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    client_time_results:  List[Tuple[str, float]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures, client_time_results=client_time_results,
        )
    return results, failures, client_time_results

def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # client_fittime_dict: Dict[str, List[float]] = {},
    client_time_results: List[Tuple[str, float]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    pre_result: Tuple[ClientProxy, Tuple[FitRes, float]] = future.result()
    client_proxy, (pre_res, client_fit_time) = pre_result
    result: Tuple[ClientProxy, FitRes] = (client_proxy, pre_res)
    client_time_result: Tuple[str, float] = (client_proxy.cid, client_fit_time)

    # Check result status code
    if pre_res.status.code == Code.OK and client_fit_time is not None:
        results.append(result)
        if client_fit_time:
            # log(
            #     INFO,
            #     "cid=%s: client_fit_time=%s",
            #     client_proxy.cid,
            #     client_fit_time
            # )
            # if client_proxy.cid not in client_fittime_dict:
            #     client_fittime_dict[client_proxy.cid] = []
            # client_fittime_dict[client_proxy.cid].append(client_fit_time)
            client_time_results.append(client_time_result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)