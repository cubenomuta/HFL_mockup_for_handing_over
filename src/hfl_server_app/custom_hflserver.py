import concurrent.futures
import os
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import torch
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
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server import History
from models.base_model import Net
from server_app.custom_history import CustomHistory

from .base_hflserver import HFLServer
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


class CustomHFLServer(HFLServer):
    """Flower server."""

    def __init__(
        self,
        *,
        fog_manager: FogManager,
        strategy: Optional[Strategy] = None,
        save_model: bool = False,
        save_dir: str = None,
        net: Net = None,
    ) -> None:
        super(CustomHFLServer, self).__init__(
            fog_manager=fog_manager,
            strategy=strategy,
        )
        self.save_model = save_model
        if self.save_model:
            assert net is not None
            assert save_dir is not None
            self.net = net
            self.save_dir = save_dir

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
            res_fit = self.fit_round(
                server_round=current_round, timeout=timeout, start_time=start_time
            )
            if res_fit:
                (
                    parameters_prime,
                    timestamps_cen,
                    timestamps_fed,
                    _,
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
                timestamps_cen["eval_round"] = timeit.default_timer() - start_time

            history.add_timestamps_centralized(
                server_round=current_round, timestamps=timestamps_cen
            )
            history.add_timestamps_distributed(
                server_round=current_round, timestamps=timestamps_fed
            )
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
        start_time: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        timestamps: Dict[str, Scalar] = {}

        # Get fogs and their respective instructions from strategy
        fog_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            fog_manager=self._fog_manager,
        )

        if not fog_instructions:
            log(INFO, "fit_round %s: no fogs selected, cancel", server_round)
            return None
        timestamps["fog_sampling"] = timeit.default_timer() - start_time
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s fogs (out of %s)",
            server_round,
            len(fog_instructions),
            self._fog_manager.num_available(),
        )
        self.set_max_workers(len(fog_instructions))

        # Collect `fit` results from all fogs participating in this round
        results, failures = fit_fogs(
            fog_instructions=fog_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        timestamps["fitres_received"] = timeit.default_timer() - start_time
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        timestamps["model_aggregation"] = timeit.default_timer() - start_time

        parameters_aggregated, metrics_aggregated = aggregated_result

        return (
            parameters_aggregated,
            timestamps,
            metrics_aggregated,
            (results, failures),
        )


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
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_fog(
    fog: FogProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[FogProxy, FitRes]:
    """Refine parameters on a single fog."""
    start_time = timeit.default_timer()
    fit_res = fog.fit(ins, timeout=timeout)
    total_time = timeit.default_timer() - start_time
    fit_res.metrics["total"] = total_time
    fit_res.metrics["fid"] = fog.fid
    return fog, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[FogProxy, FitRes]],
    failures: List[Union[Tuple[FogProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a fog
    result: Tuple[FogProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, fog returned a result where the status code is not OK
    failures.append(result)
