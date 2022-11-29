"""Handle server messages by calling appropriate fog methods."""


from typing import Tuple

from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage
from fog_app.fog import Fog, maybe_call_evaluate, maybe_call_fit, maybe_call_get_parameters, maybe_call_get_properties

# pylint: disable=missing-function-docstring


class UnknownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(fog: Fog, server_msg: ServerMessage) -> Tuple[ClientMessage, int, bool]:
    """Handle incoming messages from the server.
    Parameters
    ----------
    fog : Fog
        The Fog instance provided by the user.
    Returns
    -------
    fog_message: ClientMessage
        The message comming from the server, to be processed by the fog.
    sleep_duration : int
        Number of seconds that the fog should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the fog should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    field = server_msg.WhichOneof("msg")
    if field == "reconnect_ins":
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
        return disconnect_msg, sleep_duration, False
    if field == "get_properties_ins":
        return _get_properties(fog, server_msg.get_properties_ins), 0, True
    if field == "get_parameters_ins":
        return _get_parameters(fog, server_msg.get_parameters_ins), 0, True
    if field == "fit_ins":
        return _fit(fog, server_msg.fit_ins), 0, True
    if field == "evaluate_ins":
        return _evaluate(fog, server_msg.evaluate_ins), 0, True
    raise UnknownServerMessage()


def _reconnect(
    reconnect_msg: ServerMessage.ReconnectIns,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending DisconnectRes message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build DisconnectRes message
    disconnect_res = ClientMessage.DisconnectRes(reason=reason)
    return ClientMessage(disconnect_res=disconnect_res), sleep_duration


def _get_properties(fog: Fog, get_properties_msg: ServerMessage.GetPropertiesIns) -> ClientMessage:
    # Deserialize `get_properties` instruction
    get_properties_ins = serde.get_parameters_ins_from_proto(get_properties_msg)

    # Request properties
    get_properties_res = maybe_call_get_properties(fog=fog, get_properties_ins=get_properties_ins)

    # Serialize response
    get_properties_res_proto = serde.get_parameters_res_to_proto(get_properties_res)

    return ClientMessage(get_properties_res=get_properties_res_proto)


def _get_parameters(fog: Fog, get_parameters_msg: ServerMessage.GetParametersIns) -> ClientMessage:
    # Deserialize get_properties instruction
    get_parameters_ins = serde.get_parameters_ins_from_proto(get_parameters_msg)
    # Request parameters
    get_parameters_res = maybe_call_get_parameters(fog=fog, get_parameters_ins=get_parameters_ins)
    # Serialize response
    get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
    return ClientMessage(get_parameters_res=get_parameters_res_proto)


def _fit(fog: Fog, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)
    # Perform fit
    fit_res = maybe_call_fit(fog=fog, fit_ins=fit_ins)
    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(fog: Fog, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)
    # Perform evaluation
    evaluate_res = maybe_call_evaluate(fog=fog, evaluate_ins=evaluate_ins)
    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)
