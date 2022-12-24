"""Flower simulation app."""
from logging import INFO
from typing import Any, Callable, Dict, List, Optional

import ray
from flwr.client import Client
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.history import History
from fog_app.fog import Fog

# from fog_app.avg_fogserver_proxy import AvgFogServerProxy
# from fog_app.mkd_fogserver_proxy import MKDFogServerProxy
from hfl_server_app.app import _fl, _init_defaults
from hfl_server_app.base_hflserver import HFLServer
from hfl_server_app.fog_manager import FogManager, SimpleFogManager
from hfl_server_app.strategy.strategy import Strategy

# from fog_app.ray_fog import RayFlowerFog


def start_simulation(  # pylint: disable=too-many-arguments
    *,
    client_fn: Callable[[str], Client],
    fog_fn: Callable[[str], Fog],
    num_fogs: Optional[int],
    client_resources: Optional[Dict[str, int]] = None,
    hfl_server: Optional[HFLServer] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    fog_manager: Optional[FogManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
) -> History:
    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
        hfl_server=hfl_server,
        config=config,
        strategy=strategy,
        fog_manager=fog_manager,
    )
    log(
        INFO,
        "Starting Flower simulation running: %s",
        config,
    )
    # fogsevers_ids takes precedence
    fids: List[str] = [str(x) for x in range(num_fogs)]

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized
    if ray.is_initialized() and not keep_initialised:
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    log(
        INFO,
        "Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    # Initialize server and server config

    # Register one RayClientProxy object for each client with the ClientManager
    # resources = fog_resources if fog_resources is not None else {}
    for fid in fids:
        fogserver_proxy = fog_fn(
            fid=fid,
        )
        initialized_server.fog_manager().register(fog=fogserver_proxy)

    # Start training
    hist = _fl(
        hfl_server=initialized_server,
        config=initialized_config,
    )
    return hist
