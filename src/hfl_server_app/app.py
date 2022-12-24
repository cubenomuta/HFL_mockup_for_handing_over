import time
from logging import INFO, WARN
from typing import Optional, Tuple

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Parameters
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.history import History

from .base_hflserver import HFLServer
from .fog_manager import FogManager, SimpleFogManager
from .grpc_server.grpc_server import start_grpc_server
from .strategy.fedavg import FedAvg
from .strategy.strategy import Strategy

DEFAULT_SERVER_ADDRESS = "[::]:8080"


def start_hfl_server(  # pylint: disable=too-many-arguments
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    hfl_server: Optional[HFLServer] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    fog_manager: Optional[FogManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:
    """Start a Flower server using the gRPC transport layer.
    Arguments
    ---------
        server_address: Optional[str] (default: `"[::]:8080"`). The IPv6
            address of the server.
        server: Optional[flwr.server.Server] (default: None). An implementation
            of the abstract base class `flwr.server.Server`. If no instance is
            provided, then `start_server` will create one.
        config: ServerConfig (default: None).
            Currently supported values are `num_rounds` (int, default: 1) and
            `round_timeout` in seconds (float, default: None).
        strategy: Optional[flwr.server.Strategy] (default: None). An
            implementation of the abstract base class `flwr.server.Strategy`.
            If no strategy is provided, then `start_server` will use
            `flwr.server.strategy.FedAvg`.
        fog_manager: Optional[flwr.server.FogManager] (default: None)
            An implementation of the abstract base class `flwr.server.FogManager`.
            If no implementation is provided, then `start_server` will use
            `flwr.server.fog_manager.SimpleFogManager`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower fogs. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower fogs need to be started with the
            same value (see `flwr.fog.start_fog`), otherwise fogs will
            not know about the increased limit and block larger messages.
        certificates : Tuple[bytes, bytes, bytes] (default: None)
            Tuple containing root certificate, server certificate, and private key to
            start a secure SSL-enabled server. The tuple is expected to have three bytes
            elements in the following order:
                * CA certificate.
                * server certificate.
                * server private key.
    Returns
    -------
        hist: flwr.server.history.History. Object containing metrics from training.
    Examples
    --------
    Starting an insecure server:
    >>> start_server()
    Starting a SSL-enabled server:
    >>> start_server(
    >>>     certificates=(
    >>>         Path("/crts/root.pem").read_bytes(),
    >>>         Path("/crts/localhost.crt").read_bytes(),
    >>>         Path("/crts/localhost.key").read_bytes()
    >>>     )
    >>> )
    """

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
        hfl_server=hfl_server,
        config=config,
        strategy=strategy,
        fog_manager=fog_manager,
    )
    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = start_grpc_server(
        fog_manager=initialized_server.fog_manager(),
        server_address=server_address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    hist = _fl(
        hfl_server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    return hist


def _init_defaults(
    hfl_server: Optional[HFLServer],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    fog_manager: Optional[FogManager],
) -> Tuple[HFLServer, ServerConfig]:
    # Create server instance if none was given
    if hfl_server is None:
        if fog_manager is None:
            fog_manager = SimpleFogManager()
        if strategy is None:
            strategy = FedAvg()
        hfl_server = HFLServer(fog_manager=fog_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return hfl_server, config


def _fl(
    hfl_server: HFLServer,
    config: ServerConfig,
) -> Tuple[History, Parameters]:
    # Fit model
    hist = hfl_server.fit(num_rounds=config.num_rounds, timeout=config.round_timeout)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    # Graceful shutdown
    hfl_server.disconnect_all_fogs(timeout=config.round_timeout)

    return hist


def run_server() -> None:
    """Run Flower server."""
    print("Running Flower server...")
    time.sleep(3)
