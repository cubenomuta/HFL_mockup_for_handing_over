import gc
import sys
from logging import DEBUG
from typing import Callable, Dict, Optional, cast

import ray
from flwr import common
from flwr.client import Client, ClientLike, to_client
from flwr.client.client import maybe_call_fit
from flwr.common.logger import log
from flwr.simulation.ray_transport.ray_client_proxy import (
    RayClientProxy as FlowerRayClientProxy,
)

ClientFn = Callable[[str], ClientLike]


class RayClientProxy(FlowerRayClientProxy):
    def __init__(
        self, client_fn: ClientFn, cid: str, resources: Dict[str, float]
    ) -> None:
        super().__init__(client_fn, cid, resources)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        client_fn_ref = ray.put(self.client_fn)
        cid_ref = ray.put(self.cid)
        ins_ref = ray.put(ins)
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
        ).remote(client_fn_ref, cid_ref, ins_ref)
        try:
            res = ray.get(future_fit_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex
        fit_res = cast(common.FitRes, res)
        # release ObjectRefs in object_store_memory
        ray.internal.free(client_fn_ref)
        ray.internal.free(cid_ref)
        ray.internal.free(ins_ref)
        ray.internal.free(future_fit_res)
        return fit_res


@ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid, fit_ins)
    res = maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )
    del client
    return res


def _create_client(client_fn: ClientFn, cid: str, fit_ins: common.FitIns) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid, fit_ins.config["client_model_name"])
    return to_client(client_like=client_like)
