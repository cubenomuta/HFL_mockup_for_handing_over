"""Flower fog (abstract base class)."""


from abc import ABC

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
)


class Fog(ABC):
    """Abstract base class for Flower fogs."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return set of fog's properties.
        Parameters
        ----------
        ins : GetPropertiesIns
            The get properties instructions received from the server containing
            a dictionary of configuration values.
        Returns
        -------
        GetPropertiesRes
            The current fog properties.
        """

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters.
        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.
        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided parameters using the locally held dataset.
        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.
        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset.
        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.
        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """


def has_get_properties(fog: Fog) -> bool:
    """Check if Fog implements get_properties."""
    return type(fog).get_properties != Fog.get_properties


def has_get_parameters(fog: Fog) -> bool:
    """Check if Fog implements get_parameters."""
    return type(fog).get_parameters != Fog.get_parameters


def has_fit(fog: Fog) -> bool:
    """Check if Fog implements fit."""
    return type(fog).fit != Fog.fit


def has_evaluate(fog: Fog) -> bool:
    """Check if Fog implements evaluate."""
    return type(fog).evaluate != Fog.evaluate


def maybe_call_get_properties(fog: Fog, get_properties_ins: GetPropertiesIns) -> GetPropertiesRes:
    """Call `get_properties` if the fog overrides it."""

    # Check if fog overrides `get_properties`
    if not has_get_properties(fog=fog):
        # If fog does not override `get_properties`, don't call it
        status = Status(
            code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message="Fog does not implement `get_properties`",
        )
        return GetPropertiesRes(
            status=status,
            properties={},
        )

    # If the fog implements `get_properties`, call it
    return fog.get_properties(get_properties_ins)


def maybe_call_get_parameters(fog: Fog, get_parameters_ins: GetParametersIns) -> GetParametersRes:
    """Call `get_parameters` if the fog overrides it."""

    # Check if fog overrides `get_parameters`
    if not has_get_parameters(fog=fog):
        # If fog does not override `get_parameters`, don't call it
        status = Status(
            code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
            message="Fog does not implement `get_parameters`",
        )
        return GetParametersRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    # If the fog implements `get_parameters`, call it
    return fog.get_parameters(get_parameters_ins)


def maybe_call_fit(fog: Fog, fit_ins: FitIns) -> FitRes:
    """Call `fit` if the fog overrides it."""

    # Check if fog overrides `fit`
    if not has_fit(fog=fog):
        # If fog does not override `fit`, don't call it
        status = Status(
            code=Code.FIT_NOT_IMPLEMENTED,
            message="Fog does not implement `fit`",
        )
        return FitRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    # If the fog implements `fit`, call it
    return fog.fit(fit_ins)


def maybe_call_evaluate(fog: Fog, evaluate_ins: EvaluateIns) -> EvaluateRes:
    """Call `evaluate` if the fog overrides it."""

    # Check if fog overrides `evaluate`
    if not has_evaluate(fog=fog):
        # If fog does not override `evaluate`, don't call it
        status = Status(
            code=Code.EVALUATE_NOT_IMPLEMENTED,
            message="Fog does not implement `evaluate`",
        )
        return EvaluateRes(
            status=status,
            loss=0.0,
            num_examples=0,
            metrics={},
        )

    # If the fog implements `evaluate`, call it
    return fog.evaluate(evaluate_ins)
