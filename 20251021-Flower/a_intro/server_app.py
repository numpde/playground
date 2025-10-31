# /home/ra/repos/playground/20251021-Flower/a_intro/server_app.py

from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp


def server_fn(context: Context) -> ServerAppComponents:
    return ServerAppComponents(
        config=ServerConfig(num_rounds=3),
        strategy=FedAvg(
            min_fit_clients=2,  # require 2 clients to train
            min_evaluate_clients=2,
            min_available_clients=2,
        ),
    )


app = ServerApp(server_fn=server_fn)
