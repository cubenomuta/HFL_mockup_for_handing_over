import concurrent
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
)
from flwr.common.logger import log
from flwr.server import ClientManager, Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import evaluate_clients
from flwr.server.strategy import Strategy
from utils.utils_dataset import load_federated_dataset

from .fog import Fog

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class FlowerFog(Server, Fog):
    def __init__(
        self,
        *,
        fid: str,
        config: Dict[str, Scalar],
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None
    ):
        super(FlowerFog, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        self.fid = fid
        self.attribute = "fog_exp"
        self.config = config

        # dataset configuration
        self.target = config["target_name"]
        self.dataset = config["dataset_name"]

        self.trainset = load_federated_dataset(
            dataset_name=self.dataset,
            id=self.fid,
            train=True,
            target=self.target,
            attribute=self.attribute,
        )
        self.testset = load_federated_dataset(
            dataset_name=self.dataset,
            id=self.fid,
            train=False,
            target=self.target,
            attribute=self.attribute,
        )

    def fit(self, ins: FitIns) -> FitRes:
        start_time = timeit.default_timer()
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
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        self.set_max_workers(max_workers=len(client_instructions))
        fit_clients_start = timeit.default_timer() - start_time
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=None,
        )
        log(
            DEBUG,
            "fit_round %s: received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        fit_clients_end = timeit.default_timer() - start_time
        fit_clients_time = fit_clients_end - fit_clients_start

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_prime, metrics_aggregated = aggregated_result
        fit_total = timeit.default_timer() - start_time
        metrics_aggregated["comp"] = fit_total - fit_clients_time
        metrics_aggregated["fit_total"] = fit_total
        print(metrics_aggregated)

        return FitRes(
            status=Status(Code.OK, message="success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics=metrics_aggregated,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        server_round: int = int(ins.config["server_round"])

        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=None,
            ins=ins,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=None,
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

        return EvaluateRes(
            Status(Code.OK, message="Success evaluate"),
            loss=float(loss_aggregated),
            num_examples=int(ins.config["batch_size"]),
            metrics=metrics_aggregated,
        )


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
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    start_time = timeit.default_timer()
    fit_res = client.fit(ins, timeout=timeout)
    total_time = timeit.default_timer() - start_time
    fit_res.metrics["total"] = total_time
    fit_res.metrics["cid"] = client.cid
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
