from typing import List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.server import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy


class FedAvg(fl.server.strategy.FedAvg):
    def configure_fit(
        self,
        server_round: int,
        parameters: Optional[Parameters] = None,
        ins: Optional[FitIns] = None,
        client_manager: ClientManager = None,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if parameters is None and ins is None:
            raise ValueError(
                "Insuficient argument. configure fit requires either parameters or ins, or both"
            )
        if parameters is not None:
            config = {}
            if self.on_fit_config_fn(server_round):
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters=parameters, config=config)

        # When ins is given, override fit_ins
        if ins is not None:
            fit_ins = ins

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
