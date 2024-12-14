from logging import DEBUG
from typing import Any, Callable, Dict, Optional, Tuple, cast

from logging import DEBUG, INFO
from flwr.common.logger import log
import ray
from flwr.client import Client, ClientLike, to_client
from flwr.client.client import maybe_call_evaluate, maybe_call_fit
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.common.logger import log

from .ray_client_proxy import RayClientProxy

ClientFn = Callable[[str, Dict[str, Any]], Client]


class RayDMLClientProxy(RayClientProxy):
    def __init__(
        self,
        client_fn: ClientFn,
        cid: str,
        resources: Dict[str, float],
    ):
        super().__init__(client_fn, cid, resources)
        self.parameters = None

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        client_fn_ref = ray.put(self.client_fn)
        cid_ref = ray.put(self.cid)
        ins_ref = ray.put(ins)
        parameters_ref = ray.put(self.parameters)

        future_fit_res = launch_and_fit.options(
            **self.resources,
        ).remote(client_fn_ref, cid_ref, ins_ref, parameters_ref)
        try:
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        fit_res, parameters, client_train_time = cast(Tuple[FitRes, Parameters], res)
        del res
        self.parameters = parameters
        ray.internal.free(client_fn_ref)
        ray.internal.free(cid_ref)
        ray.internal.free(ins_ref)
        ray.internal.free(parameters_ref)
        ray.internal.free(future_fit_res)
        return fit_res, client_train_time

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        client_fn_ref = ray.put(self.client_fn)
        cid_ref = ray.put(self.cid)
        ins_ref = ray.put(ins)
        parameters_ref = ray.put(self.parameters)
        future_evaluate_res = launch_and_evaluate.options(
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, self.parameters)
        try:
            res = ray.get(future_evaluate_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        evaluate_res = cast(EvaluateRes, res)
        del res
        ray.internal.free(client_fn_ref)
        ray.internal.free(cid_ref)
        ray.internal.free(ins_ref)
        ray.internal.free(parameters_ref)
        ray.internal.free(future_evaluate_res)
        return evaluate_res


@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: FitIns, parameters: Parameters
) -> FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid, parameters)
    return maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )


@ray.remote
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: EvaluateIns, parameters: Parameters
) -> EvaluateRes:
    """Execute evaluate remotely"""
    client: Client = _create_client(client_fn, cid, parameters)
    return maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )


def _create_client(client_fn: ClientFn, cid: str, parameters: Parameters) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid, parameters)
    return to_client(client_like=client_like)
