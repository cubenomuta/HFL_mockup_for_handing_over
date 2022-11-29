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


class FogManager(ABC):
    """Abstract base class for managing Flower fogs."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available fogs."""

    @abstractmethod
    def register(self, fog: FogProxy) -> bool:
        """Register Flower FogProxy instance.
        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, fog: FogProxy) -> None:
        """Unregister Flower FogProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, FogProxy]:
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
    ) -> List[FogProxy]:
        """Sample a number of Flower FogProxy instances."""


class SimpleFogManager(FogManager):
    """Provides a pool of available fogs."""

    def __init__(self) -> None:
        self.fogs: Dict[str, FogProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.fogs)

    def wait_for(self, num_fogs: int, timeout: int = 86400) -> bool:
        """Block until at least `num_fogs` are available or until a timeout
        is reached.
        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(lambda: len(self.fogs) >= num_fogs, timeout=timeout)

    def num_available(self) -> int:
        """Return the number of available fogs."""
        return len(self)

    def register(self, fog: FogProxy) -> bool:
        """Register Flower FogProxy instance.
        Returns:
            bool: Indicating if registration was successful. False if FogProxy is
                already registered or can not be registered for any reason
        """
        if fog.fid in self.fogs:
            return False

        with self._cv:
            self.fogs[fog.fid] = fog
            self._cv.notify_all()

        return True

    def unregister(self, fog: FogProxy) -> None:
        """Unregister Flower FogProxy instance.
        This method is idempotent.
        """
        if fog.fid in self.fogs:
            # del self.fogs[fog.fid]

            with self._cv:
                del self.fogs[fog.fid]
                self._cv.notify_all()

    def all(self) -> Dict[str, FogProxy]:
        """Return all available fogs."""
        return self.fogs

    def sample(
        self,
        num_fogs: int,
        min_num_fogs: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[FogProxy]:
        """Sample a number of Flower FogProxy instances."""
        # Block until at least num_fogs are connected.
        print(f"sample {min_num_fogs}")
        if min_num_fogs is None:
            min_num_fogs = num_fogs
        self.wait_for(min_num_fogs)
        print(f"fogs {min_num_fogs} are available")
        # Sample fogs which meet the criterion
        available_fids = list(self.fogs)
        if criterion is not None:
            available_fids = [fid for fid in available_fids if criterion.select(self.fogs[fid])]

        if num_fogs > len(available_fids):
            log(
                INFO,
                "Sampling failed: number of available fogs" " (%s) is less than number of requested fogs (%s).",
                len(available_fids),
                num_fogs,
            )
            return []

        sampled_fids = random.sample(available_fids, num_fogs)
        return [self.fogs[fid] for fid in sampled_fids]
