import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor1LDS(nn.Module):
    """
    Input: flattened state [B, state_dim]
    Output: logits [B, NRBG, A]  where A = max_ues + 1 (last action = no-allocation)
    """
    def __init__(self, state_dim: int, n_rbg: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.n_rbg = n_rbg
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_rbg * n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)  # [B, n_rbg*n_actions]
        return z.view(x.shape[0], self.n_rbg, self.n_actions)


class QuantileCritic1LDS(nn.Module):
    """
    Input: flattened state [B, state_dim]
    Output: quantiles [B, NRBG, A, N]
    """
    def __init__(self, state_dim: int, n_rbg: int, n_actions: int, n_quantiles: int = 16, hidden: int = 256):
        super().__init__()
        self.n_rbg = n_rbg
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_rbg * n_actions * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)  # [B, n_rbg*n_actions*n_quantiles]
        return z.view(x.shape[0], self.n_rbg, self.n_actions, self.n_quantiles)
