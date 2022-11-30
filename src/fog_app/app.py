import time
from dataclasses import dataclass
from logging import INFO, WARN
from typing import Optional, Tuple

from flwr.client.grpc_client.connection import grpc_connection
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server import ClientManager, SimpleClientManager
from flwr.server.grpc_server.grpc_server import start_grpc_server
from flwr.server.strategy import Strategy

from .base_fog import FlowerFog
from .fog import Fog
from .message_handler.message_handler import handle
from .strategy import FedAvg

DEFAULT_SERVER_ADDRESS = "[::]:8080"


@dataclass
class FogConfig:
    """Flower fog config.
    All attributes have default values which allows users to configure
    just the ones they care about.
    """

    round_timeout: Optional[float] = None


def start_fog(
    *,
    server_address: str,
    fog_address: str,
    fid: Optional[str] = None,
    fog: Optional[Fog] = None,
    config: Optional[FogConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None
) -> None:
    # Initialize fog and fog config
    initialized_fog, initialized_config = _init_defaults(
        fid=fid,
        fog=fog,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )

    # Start gRPC server
    grpc_fogserver = start_grpc_server(
        client_manager=initialized_fog.client_manager(),
        server_address=fog_address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running, SSL is %s",
        "enabled" if certificates is not None else "disabled",
    )

    while True:
        sleep_duration: int = 0
        with grpc_connection(
            server_address,
            max_message_length=grpc_max_message_length,
            root_certificates=root_certificates,
        ) as conn:
            receive, send = conn

            while True:
                server_message = receive()
                fog_message, sleep_duration, keep_going = handle(
                    initialized_fog, server_message
                )
                send(fog_message)
                if not keep_going:
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)

    grpc_fogserver.stop(grace=1)


def _init_defaults(
    fid: str,
    fog: Optional[Fog],
    config: Optional[FogConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Fog, FogConfig]:
    # Create fog instance if none was given
    if fog is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        if config is None:
            config = {}
        fog = FlowerFog(
            fid=fid, config=config, client_manager=client_manager, strategy=strategy
        )
    elif strategy is not None:
        log(WARN, "Both fog and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = FogConfig()

    return fog, config
