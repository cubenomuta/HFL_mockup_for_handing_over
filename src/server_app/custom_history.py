# Flwoer API
from typing import Dict, List, Tuple

from flwr.common.typing import Scalar
from flwr.server.history import History


class CustomHistory(History):
    """History class for training and/or evaluation metrics collection."""

    def __init__(self) -> None:
        super(CustomHistory, self).__init__()
        self.timestamps_centralized: Dict[str, List[Tuple[int, Scalar]]] = {}
        self.timestamps_distributed: Dict[str, Dict[str, List[Tuple[int, Scalar]]]] = {}

    def add_metrics_distributed(
        self, server_round: int, metrics: Dict[str, Dict[str, Scalar]]
    ) -> None:
        """Add metrics entries (from distributed evaluation)."""
        for key in metrics:
            # if not (isinstance(metrics[key], float) or isinstance(metrics[key], int)):
            #     continue  # ignore non-numeric key/value pairs
            if key not in self.metrics_distributed:
                self.metrics_distributed[key] = {}
            for cid in metrics[key]:
                if cid not in self.metrics_distributed[key]:
                    self.metrics_distributed[key][cid] = []
                self.metrics_distributed[key][cid].append(
                    (server_round, metrics[key][cid])
                )

    def add_timestamps_centralized(
        self, server_round: int, timestamps: Dict[str, Scalar]
    ) -> None:
        for key in timestamps:
            if key not in self.timestamps_centralized:
                self.timestamps_centralized[key] = []
            self.timestamps_centralized[key].append((server_round, timestamps[key]))

    def add_timestamps_distributed(
        self, server_round: int, timestamps: Dict[str, Scalar]
    ) -> None:
        for key in timestamps:
            if key not in self.timestamps_distributed:
                self.timestamps_distributed[key] = {}
                self.timestamps_distributed[key]["comm"] = []
                self.timestamps_distributed[key]["comp"] = []
            self.timestamps_distributed[key]["comm"].append(
                (server_round, timestamps[key]["comm"])
            )
            self.timestamps_distributed[key]["comp"].append(
                (server_round, timestamps[key]["comp"])
            )
