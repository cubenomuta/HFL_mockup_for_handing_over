# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower FogManager."""


import random
import threading
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log

from .criterion import Criterion
from .fog_proxy import FogProxy
from .client_cluster_proxy import ClientClusterProxy


class ClientClusterManager(ABC):
    """Abstract base class for managing Flower fogs."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available fogs."""

    @abstractmethod
    def register(self, fog: ClientClusterProxy) -> bool:
        """Register Flower ClientClusterProxy instance.
        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, fog: ClientClusterProxy) -> None:
        """Unregister Flower ClientClusterProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, ClientClusterProxy]:
        """Return all available fogs."""

    @abstractmethod
    def wait_for(self, num_fogs: int, timeout: int) -> bool:
        """Wait until at least `num_fogs` are available."""

    @abstractmethod
    def sample(
        self,
        num_fogs: int,
        min_num_fogs: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientClusterProxy]:
        """Sample a number of Flower ClientClusterProxy instances."""


class SimpleClientClusterManager(ClientClusterManager):
    """Provides a pool of available clientclusters."""

    def __init__(self) -> None:
        self.client_clusters: Dict[str, ClientClusterProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.client_clusters)
    
    def select_by_clsid(self, clsid: str) -> Optional[ClientClusterProxy]:
        """Select a client cluster by cid."""
        return self.client_clusters.get(clsid, None)

    def wait_for(self, num_client_clusters: int, timeout: int = 86400) -> bool:
        """Block until at least `num_fogs` are available or until a timeout
        is reached.
        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.fogs) >= num_client_clusters, timeout=timeout
            )

    def num_available(self) -> int:
        """Return the number of available fogs."""
        return len(self)

    def register(self, client_cluster: ClientClusterProxy) -> bool:
        """Register Flower ClientClusterProxy instance.
        Returns:
            bool: Indicating if registration was successful. False if client_clusterProxy is
                already registered or can not be registered for any reason
        """
        if client_cluster.clsid in self.client_clusters:
            return False

        with self._cv:
            self.client_clusters[client_cluster.clsid] = client_cluster
            self._cv.notify_all()

        return True

    def unregister(self, client_cluster: ClientClusterProxy) -> None:
        """Unregister Flower ClientClusterProxy instance.
        This method is idempotent.
        """
        if client_cluster.clsid in self.client_clusters:
            # del self.fogs[fog.clsid]

            with self._cv:
                del self.client_clusters[client_cluster.clsid]
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientClusterProxy]:
        """Return all available clusters."""
        # return self.client_clusters
        return list(self.client_clusters.values())

    def sample(
        self,
        num_client_clusters: int,
        min_num_client_clusters: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientClusterProxy]:
        """Sample a number of Flower ClientClusterProxy instances."""
        # Block until at least num_fogs are connected.
        if min_num_client_clusters is None:
            min_num_client_clusters = num_client_clusters
        self.wait_for(min_num_client_clusters)
        # Sample fogs which meet the criterion
        available_clsids = list(self.client_clusters)
        if criterion is not None:
            available_clsids = [
                clsid for clsid in available_clsids if criterion.select(self.client_clusters[clsid])
            ]

        if num_client_clusters > len(available_clsids):
            log(
                INFO,
                "Sampling failed: number of available client_clusters"
                " (%s) is less than number of requested client_clusters (%s).",
                len(available_clsids),
                num_client_clusters,
            )
            return []

        sampled_clsids = random.sample(available_clsids, num_client_clusters)
        return [self.client_clusters[clsid] for clsid in sampled_clsids]
