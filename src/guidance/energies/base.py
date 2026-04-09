from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseEnergy(nn.Module, ABC):
    """Energy E_i(G) on soft graph probabilities; lower is better."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        X_soft: Tensor,
        E_soft: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Return per-batch energies of shape (B,)."""
        ...
