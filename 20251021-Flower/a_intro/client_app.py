# /home/ra/repos/playground/20251021-Flower/a_intro/client_app.py

import os
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.app import Context
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader, TensorDataset


# --- local data (synthetic linear regression) -------------------------------
def make_data(seed: int, n: int = 1500, d: int = 5) -> Tuple[DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    w_true = rng.normal(size=(d, 1)).astype("float32")
    b_true = rng.normal(size=(1,)).astype("float32")
    y = X @ w_true + b_true + 0.05 * rng.normal(size=(n, 1)).astype("float32")
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    n_train = int(0.9 * n)
    train = torch.utils.data.Subset(ds, range(0, n_train))
    val = torch.utils.data.Subset(ds, range(n_train, n))
    return (
        DataLoader(train, batch_size=64, shuffle=True, drop_last=True),
        DataLoader(val, batch_size=64, shuffle=False),
    )


# --- tiny trainer + param bridge -------------------------------------------
class Trainer:
    def __init__(self, d_in: int = 5, lr: float = 1e-2):
        self.model = nn.Linear(d_in, 1)  # CPU only
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.keys = sorted(self.model.state_dict().keys())  # stable order

    def get_parameters(self) -> List[np.ndarray]:
        sd = self.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in self.keys]

    def set_parameters(self, params: List[np.ndarray]) -> None:
        sd = self.model.state_dict()
        for k, arr in zip(self.keys, params):
            sd[k] = torch.tensor(arr, device=sd[k].device)
        self.model.load_state_dict(sd, strict=False)

    def train_steps(self, loader: DataLoader, steps: int = 100) -> float:
        self.model.train()
        it = iter(loader)
        total = 0.0
        for _ in range(steps):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            self.opt.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(xb), yb)
            loss.backward()
            self.opt.step()
            total += float(loss.detach().cpu())
        return total / max(1, steps)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                loss = self.loss_fn(self.model(xb), yb)
                bs = len(xb)
                total += float(loss.cpu()) * bs
                n += bs
        return total / max(1, n)


# --- Flower NumPyClient -----------------------------------------------------
class TorchFLClient(fl.client.NumPyClient):
    """Implements Flower's 3 hooks for one site."""

    def __init__(self, seed: int):
        (self.train_loader, self.val_loader) = make_data(seed)
        self.trainer = Trainer(d_in=5, lr=1e-2)
        self.local_steps = 100

    def get_parameters(self, config):
        return self.trainer.get_parameters()

    def fit(self, parameters, config):
        self.trainer.set_parameters(parameters)  # load G^k
        train_loss = self.trainer.train_steps(self.train_loader)  # → W_i^{k+1}
        return self.trainer.get_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.trainer.set_parameters(parameters)
        val_loss = self.trainer.evaluate(self.val_loader)
        return float(val_loss), len(self.val_loader.dataset), {"val_loss": float(val_loss)}


# --- ClientApp factory ------------------------------------------------------
def client_fn(context: Context):
    seed = int(os.environ.get("SEED", "0"))  # different seed ⇒ different local data
    return TorchFLClient(seed).to_client()


app = ClientApp(client_fn=client_fn)
