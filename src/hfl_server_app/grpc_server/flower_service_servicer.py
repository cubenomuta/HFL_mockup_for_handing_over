"""Servicer for Hierarchucal server.
"""
from typing import Callable, Iterator

import grpc
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.proto.transport_pb2_grpc import FlowerServiceServicer
from iterators import TimeoutIterator

from ..fog_manager import FogManager
from .grpc_bridge import GRPCBridge, InsWrapper, ResWrapper
from .grpc_fog_proxy import GrpcFogProxy


def default_bridge_factory() -> GRPCBridge:
    """Return GRPCBridge instance."""
    return GRPCBridge()


def default_grpc_fog_factory(fid: str, bridge: GRPCBridge) -> GrpcFogProxy:
    """Return GrpcFogProxy instance."""
    return GrpcFogProxy(fid=fid, bridge=bridge)


def register_fog(
    fog_manager: FogManager,
    fog: GrpcFogProxy,
    context: grpc.ServicerContext,
) -> bool:
    """Try registering GrpcFogProxy with FogManager."""
    is_success = fog_manager.register(fog)

    if is_success:

        def rpc_termination_callback() -> None:
            fog.bridge.close()
            fog_manager.unregister(fog)

        context.add_callback(rpc_termination_callback)

    return is_success


class FlowerServiceServicer(FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC message stream."""

    def __init__(
        self,
        fog_manager: FogManager,
        grpc_bridge_factory: Callable[[], GRPCBridge] = default_bridge_factory,
        grpc_fog_factory: Callable[
            [str, GRPCBridge], GrpcFogProxy
        ] = default_grpc_fog_factory,
    ) -> None:
        self.fog_manager: FogManager = fog_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.fog_factory = grpc_fog_factory

    def Join(  # pylint: disable=invalid-name
        self,
        request_iterator: Iterator[ClientMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[ServerMessage]:
        """Method will be invoked by each GrpcFogProxy which participates in
        the network.
        Protocol:
            - The first message is sent from the server to the fog
            - Both ServerMessage and ClientMessage are message "wrappers"
                wrapping the actual message
            - The Join method is (pretty much) protocol unaware
        """
        peer = context.peer()
        bridge = self.grpc_bridge_factory()
        fog = self.fog_factory(peer, bridge)
        is_success = register_fog(self.fog_manager, fog, context)

        if is_success:
            # Get iterators
            fog_message_iterator = TimeoutIterator(
                iterator=request_iterator, reset_on_next=True
            )
            ins_wrapper_iterator = bridge.ins_wrapper_iterator()

            # All messages will be pushed to fog bridge directly
            while True:
                try:
                    # Get ins_wrapper from bridge and yield server_message
                    ins_wrapper: InsWrapper = next(ins_wrapper_iterator)
                    yield ins_wrapper.server_message

                    # Set current timeout, might be None
                    if ins_wrapper.timeout is not None:
                        fog_message_iterator.set_timeout(ins_wrapper.timeout)

                    # Wait for fog message
                    fog_message = next(fog_message_iterator)

                    if fog_message is fog_message_iterator.get_sentinel():
                        # Important: calling `context.abort` in gRPC always
                        # raises an exception so that all code after the call to
                        # `context.abort` will not run. If subsequent code should
                        # be executed, the `rpc_termination_callback` can be used
                        # (as shown in the `register_fog` function).
                        details = f"Timeout of {ins_wrapper.timeout}sec was exceeded."
                        context.abort(
                            code=grpc.StatusCode.DEADLINE_EXCEEDED,
                            details=details,
                        )
                        # This return statement is only for the linter so it understands
                        # that fog_message in subsequent lines is not None
                        # It does not understand that `context.abort` will terminate
                        # this execution context by raising an exception.
                        return

                    bridge.set_res_wrapper(
                        res_wrapper=ResWrapper(fog_message=fog_message)
                    )
                except StopIteration:
                    break
