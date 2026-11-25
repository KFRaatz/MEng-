"""The fully connected networks of AZ (Section 3.6.2, Table 3.4), after Kemmerling's agent.

Separate policy and value networks, each with two hidden layers of 256 units and Mish activations, read the
two features of all ``n * m`` operations as one vector of ``2 * n * m`` values.  The policy network has one
output per job (finished jobs are masked before the softmax) and the value network one output with a tanh.
A pair of networks is specific to one instance size.
"""
from __future__ import annotations

import torch
from torch import nn


def _mish_mlp(in_dim: int, hidden: int, hidden_layers: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, hidden), nn.Mish()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class AZNetworks(nn.Module):
    kind = "ffnn"

    def __init__(self, n: int, m: int, hidden: int = 256, hidden_layers: int = 2, in_features: int = 2):
        super().__init__()
        self.n, self.m = int(n), int(m)
        self.hparams = dict(n=self.n, m=self.m, hidden=hidden, hidden_layers=hidden_layers,
                            in_features=in_features)
        in_dim = in_features * self.n * self.m
        self.policy = _mish_mlp(in_dim, hidden, hidden_layers, self.n)
        self.value = _mish_mlp(in_dim, hidden, hidden_layers, 1)

    def policy_parameters(self) -> list:
        """Parameters of the policy network (the value network is separate)."""
        return list(self.policy.parameters())

    def forward_batch(self, x: torch.Tensor, mpred: torch.Tensor, next_op: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same interface as the GIN; the arcs (``mpred``) and ``next_op`` are not used."""
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self.policy[0].in_features:
            raise ValueError(f"the networks of AZ were built for {self.n}x{self.m} instances")
        logits = self.policy(flat).masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value(flat)).squeeze(-1)
        return logits, value
