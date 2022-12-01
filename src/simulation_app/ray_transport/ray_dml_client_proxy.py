from logging import DEBUG
from typing import Any, Callable, Dict, Optional, Tuple, cast

import ray
from flwr.client import Client, ClientLike, to_client
from flwr.client.client import maybe_call_fit
from flwr.common import FitIns, FitRes, Parameters
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
        future_fit_res = launch_and_fit.options(
            **self.resources,
        ).remote(self.client_fn, self.cid, ins, self.parameters)
        try:
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        fit_res, parameters = cast(Tuple[FitRes, Parameters], res)
        self.parameters = parameters
        return fit_res


@ray.remote(max_calls=1)
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: FitIns, parameters: Parameters
) -> FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid, parameters)
    return maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )


def _create_client(client_fn: ClientFn, cid: str, parameters: Parameters) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid, parameters)
    return to_client(client_like=client_like)
