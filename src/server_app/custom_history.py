# Flwoer API
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from flwr.common.typing import Scalar
from flwr.server.history import History

from logging import DEBUG, INFO
from flwr.common.logger import log


class CustomHistory(History):
    """History class for training and/or evaluation metrics collection."""

    def __init__(self, save_dir: Optional[str] = None) -> None:
        super(CustomHistory, self).__init__()
        self.save_dir = save_dir
        self.metrics_cluster_model: Dict[str, List[Tuple[int, Scalar]]] = {}
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

        losses_distributed = sorted(self.metrics_distributed["loss"].items())
        save_path = Path(self.save_dir) / "metrics" / "losses_distributed.json"
        with open(save_path, "w") as f:
            json.dump(losses_distributed, f)

        # accuracy of client models
        accuracies_distributed = sorted(self.metrics_distributed["accuracy"].items())
        save_path = Path(self.save_dir) / "metrics" / "accuracies_distributed.json"
        with open(save_path, "w") as f:
            json.dump(accuracies_distributed, f)

    def add_metrics_cluster_models(
        self, server_round: int, metrics: Dict[str, Dict[int, Scalar]]
    ) -> None:
        for key in metrics:
            # log(INFO, "Adding metrics for key: %s", key)
            if key not in self.metrics_cluster_model:
                self.metrics_cluster_model[key] = {}
            for fid in metrics[key]:
                for clsid in metrics[key][fid]:
                    if clsid not in self.metrics_cluster_model[key]:
                        self.metrics_cluster_model[key][clsid] = []
                    self.metrics_cluster_model[key][clsid].append(
                        (server_round, metrics[key][fid][clsid])
                    )

        losses_fog_centralized = sorted(self.metrics_cluster_model["loss"].items())
        save_path = Path(self.save_dir) / "metrics" / "losses_cluster_model.json"
        with open(save_path, "w") as f:
            json.dump(losses_fog_centralized, f)

        accuracies_fog_centralized = sorted(self.metrics_cluster_model["accuracy"].items())
        save_path = Path(self.save_dir) / "metrics" / "accuracies_cluster_model.json"
        with open(save_path, "w") as f:
            json.dump(accuracies_fog_centralized, f)

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
    
    def save_loss_centralized(self):
        losses_centralized = sorted(self.losses_centralized)
        save_path = Path(self.save_dir) / "metrics" / "losses_centralized.json"
        with open(save_path, "w") as f:
            json.dump(losses_centralized, f)

    def save_metrics_centralized(self):
        accuracies_centralized = sorted(self.metrics_centralized["accuracy"])
        save_path = Path(self.save_dir) / "metrics" / "accuracies_centralized.json"
        with open(save_path, "w") as f:
            json.dump(accuracies_centralized, f)
