"""The GIN of LR: a graph isomorphism network with a policy and a value head (Section 3.6.2, Table 3.4).

The design follows L2D (Zhang et al.):

* encoder: two GIN layers.  In each layer an MLP with two hidden layers of 64 units computes the new
  embedding of an operation from its own embedding and those of its incoming neighbours in the disjunctive
  graph (job predecessor and machine predecessor), followed by batch normalisation and ReLU;
* pooling: ``h_G`` is the mean of the node embeddings;
* policy head: for every candidate operation ``a`` the concatenation ``[h_a; h_G]`` (128 values) passes
  through two hidden layers of 32 units with tanh and gives one logit;
* value head: ``h_G`` (64 values) passes through two hidden layers of 32 units with tanh, and a tanh on the
  single output keeps the value in [-1, 1], the range of the ranked reward.

The encoder shares its weights across the nodes, so the network reads a graph of any size.

Two forward passes compute the same function:

* :meth:`GINPolicyValue.forward_batch` uses the message passing of PyTorch Geometric (``GINConv``) on a batch
  of graphs and is used for training;
* :meth:`GINPolicyValue.forward_single` evaluates one graph by gathering the two predecessors of every node
  directly, which avoids the overhead of message passing in the rollouts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv


def _gin_mlp(in_dim: int, hidden: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, hidden))
    return nn.Sequential(*layers)


def _tanh_head(in_dim: int, hidden: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class GINPolicyValue(nn.Module):
    kind = "gin"

    def __init__(self, hidden: int = 64, gin_layers: int = 2, mlp_hidden_layers: int = 2,
                 head_hidden: int = 32, head_hidden_layers: int = 2, in_features: int = 2):
        super().__init__()
        self.hparams = dict(hidden=hidden, gin_layers=gin_layers, mlp_hidden_layers=mlp_hidden_layers,
                            head_hidden=head_hidden, head_hidden_layers=head_hidden_layers,
                            in_features=in_features)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_features
        for _ in range(gin_layers):
            # eps = 0 and not trained: the embedding of a node and of its neighbours are summed (as in L2D)
            self.convs.append(GINConv(_gin_mlp(d, hidden, mlp_hidden_layers), eps=0.0, train_eps=False))
            self.norms.append(nn.BatchNorm1d(hidden))
            d = hidden
        self.policy_head = _tanh_head(2 * hidden, head_hidden, head_hidden_layers)
        self.value_head = _tanh_head(hidden, head_hidden, head_hidden_layers)

    def policy_parameters(self) -> list:
        """Parameters that the policy loss reaches: the shared encoder and the policy head."""
        return list(self.convs.parameters()) + list(self.norms.parameters()) + list(self.policy_head.parameters())

    # ----------------------------------------------------------------------------- batch (training)
    @staticmethod
    def edge_index(mpred: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """Arcs ``predecessor -> operation`` of a batch of graphs, with node offsets per graph.

        ``mpred`` is ``(B, N)`` with ``N`` marking operations without a machine predecessor.
        """
        batch, n_ops = mpred.shape
        device = mpred.device
        nodes = torch.arange(n_ops, device=device)
        offsets = (torch.arange(batch, device=device) * n_ops).unsqueeze(1)
        has_jp = (nodes % m) != 0                                    # conjunctive arcs within the jobs
        dst_j = (nodes[has_jp].unsqueeze(0) + offsets).reshape(-1)
        src_j = dst_j - 1
        b_idx, v_idx = torch.nonzero(mpred < n_ops, as_tuple=True)   # machine arcs of scheduled operations
        dst_m = v_idx + b_idx * n_ops
        src_m = mpred[b_idx, v_idx] + b_idx * n_ops
        return torch.stack([torch.cat([src_j, src_m]), torch.cat([dst_j, dst_m])])

    def forward_batch(self, x: torch.Tensor, mpred: torch.Tensor, next_op: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits ``(B, n)`` (``-inf`` for finished jobs) and values ``(B,)`` of a batch of states.

        ``x`` is ``(B, N, 2)``, ``mpred`` ``(B, N)``, ``next_op`` ``(B, n)`` and ``mask`` ``(B, n)`` (True
        for unfinished jobs).  All graphs of a batch have the same size.
        """
        batch, n_ops, _ = x.shape
        n = mask.shape[1]
        m = n_ops // n
        edges = self.edge_index(mpred, n, m)
        h = x.reshape(batch * n_ops, -1)
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, edges)))
        h = h.view(batch, n_ops, -1)
        h_graph = h.mean(dim=1)
        cand = torch.arange(n, device=x.device).unsqueeze(0) * m + next_op.clamp(max=m - 1)
        h_cand = torch.gather(h, 1, cand.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        pooled = h_graph.unsqueeze(1).expand(-1, n, -1)
        logits = self.policy_head(torch.cat([h_cand, pooled], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value_head(h_graph)).squeeze(-1)
        return logits, value

    # ---------------------------------------------------------------------------- single (rollouts)
    def encode_single(self, x: torch.Tensor, jpred: torch.Tensor, mpred: torch.Tensor) -> torch.Tensor:
        """Node embeddings of one graph; ``jpred`` and ``mpred`` use ``N`` for "no predecessor"."""
        h = x
        for conv, norm in zip(self.convs, self.norms):
            padded = torch.cat([h, h.new_zeros(1, h.shape[1])])
            h = torch.relu(norm(conv.nn(h + padded[jpred] + padded[mpred])))
        return h

    def forward_single(self, x: torch.Tensor, jpred: torch.Tensor, mpred: torch.Tensor,
                       cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits of the candidate operations ``cand`` and the value of one state (evaluation mode)."""
        h = self.encode_single(x, jpred, mpred)
        h_graph = h.mean(dim=0, keepdim=True)
        h_cand = h[cand]
        logits = self.policy_head(torch.cat([h_cand, h_graph.expand(h_cand.shape[0], -1)], dim=1)).squeeze(-1)
        value = torch.tanh(self.value_head(h_graph)).reshape(())
        return logits, value
