from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar

from ..fog_manager import FogManager
from ..fog_proxy import FogProxy


class Strategy(ABC):
    """Abstract base class for server strategy implementations."""

    @abstractmethod
    def initialize_parameters(self, fog_manager: FogManager) -> Optional[Parameters]:
        """Initialize the (global) model parameters.
        Parameters
        ----------
        fog_manager : FogManager
            The fog manager which holds all currently connected fogs.
        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """

    @abstractmethod
    def configure_fit(
        self, server_round: int, parameters: Parameters, fog_manager: FogManager
    ) -> List[Tuple[FogProxy, FitIns]]:
        """Configure the next round of training.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        fog_manager : FogManager
            The fog manager which holds all currently connected fogs.
        Returns
        -------
        fit_configuration : List[Tuple[FogProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `FogProxy` and the
            `FitIns` for this particular `FogProxy`. If a particular `FogProxy`
            is not included in this list, it means that this `FogProxy`
            will not participate in the next round of federated learning.
        """

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[FogProxy, FitRes]],
        failures: List[Union[Tuple[FogProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[FogProxy, FitRes]]
            Successful updates from the previously selected and configured
            fogs. Each pair of `(FogProxy, FitRes)` constitutes a
            successful update from one of the previously selected fogs. Not
            that not all previously selected fogs are necessarily included in
            this list: a fog might drop out and not submit a result. For each
            fog that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[FogProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for fog
            updates.
        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, fog_manager: FogManager
    ) -> List[Tuple[FogProxy, EvaluateIns]]:
        """Configure the next round of evaluation.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        fog_manager : FogManager
            The fog manager which holds all currently connected fogs.
        Returns
        -------
        evaluate_configuration : List[Tuple[FogProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `FogProxy` and the
            `EvaluateIns` for this particular `FogProxy`. If a particular
            `FogProxy` is not included in this list, it means that this
            `FogProxy` will not participate in the next round of federated
            evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[FogProxy, EvaluateRes]],
        failures: List[Union[Tuple[FogProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[FogProxy, FitRes]]
            Successful updates from the
            previously selected and configured fogs. Each pair of
            `(FogProxy, FitRes` constitutes a successful update from one of the
            previously selected fogs. Not that not all previously selected
            fogs are necessarily included in this list: a fog might drop out
            and not submit a result. For each fog that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[FogProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for fog updates.
        Returns
        -------
        aggregation_result : Optional[float]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average.
        """

    @abstractmethod
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters.
        This function can be used to perform centralized (i.e., server-side) evaluation
        of model parameters.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters: Parameters
            The current (global) model parameters.
        Returns
        -------
        evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """
