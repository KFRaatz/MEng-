"""Node of the search tree.

The statistics of the edge ``(s, a)`` that leads to a node are kept on the node itself: ``N`` is the visit
count ``N(s, a)``, ``W`` the sum of the backed-up values and ``Q = W / N`` their mean (Section 3.2.1).
States are not stored in the tree: an iteration copies the state of the root once and replays the actions on
its path, which is cheap because the dynamics model is explicit.
"""
from __future__ import annotations

from typing import Optional


class Node:
    __slots__ = ("action", "children", "untried", "N", "W", "P", "P0", "terminal", "expanded")

    def __init__(self, action: int = -1, untried: Optional[list] = None, prior: float = 0.0):
        self.action = action                 # job dispatched on the edge into this node
        self.children: list[Node] = []
        self.untried = untried               # legal actions without a child yet (UCT); None until known
        self.N = 0
        self.W = 0.0
        self.P = prior                       # prior used by PUCT (with Dirichlet noise at the root)
        self.P0 = prior                      # prior of the policy network without noise
        self.terminal = False                # every operation is scheduled
        self.expanded = False                # PUCT: children created with priors

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def child(self, action: int) -> Optional["Node"]:
        for c in self.children:
            if c.action == action:
                return c
        return None

    def most_visited(self) -> "Node":
        """Child with the most visits; ties are broken by the higher mean value, then the lower job."""
        return max(self.children, key=lambda c: (c.N, c.Q, -c.action))

    def subtree_size(self) -> int:
        size, stack = 0, [self]
        while stack:
            node = stack.pop()
            size += 1
            stack.extend(node.children)
        return size
