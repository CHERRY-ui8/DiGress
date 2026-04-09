import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.guidance.energies.base import BaseEnergy


class EnergyGuidance(nn.Module):
    """
    Mixture-of-energies objective E = -log sum_i pi_i exp(-E_i) on soft predictions;
    one gradient step on detached softmax probabilities, then project back to the simplex.
    """

    def __init__(
        self,
        energy_fns: List[BaseEnergy],
        weights: List[float],
        lambda_scale: float = 1.0,
        clip_grad_max: float = 1.0,
    ):
        super().__init__()
        if len(energy_fns) != len(weights):
            raise ValueError("energy_fns and weights must have the same length")
        total = sum(weights)
        if total <= 0:
            raise ValueError("sum of weights must be positive")
        self._energy_fns = nn.ModuleList(energy_fns)
        self.weights = [w / total for w in weights]
        self.lambda_scale = lambda_scale
        self.clip_grad_max = clip_grad_max

    @property
    def energy_fns(self):
        return list(self._energy_fns)

    def combined_energy(self, X_soft: torch.Tensor, E_soft: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        log_pi = torch.tensor(
            [math.log(w) for w in self.weights],
            device=X_soft.device,
            dtype=X_soft.dtype,
        )
        terms = []
        for i, e_fn in enumerate(self._energy_fns):
            e_i = e_fn(X_soft, E_soft, node_mask)
            terms.append(log_pi[i] - e_i)
        stack = torch.stack(terms, dim=0)
        E_comb = -torch.logsumexp(stack, dim=0)
        return E_comb.mean()

    def guide(
        self,
        X_soft: torch.Tensor,
        E_soft: torch.Tensor,
        node_mask: torch.Tensor,
        lambda_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns guided soft probabilities (same shape as inputs), detached.
        :param lambda_scale: if None, uses self.lambda_scale (caller may pass time-scaled value).
        """
        lr = self.lambda_scale if lambda_scale is None else float(lambda_scale)
        with torch.enable_grad():
            X_in = X_soft.detach().requires_grad_(True)
            E_in = E_soft.detach().requires_grad_(True)
            energy = self.combined_energy(X_in, E_in, node_mask)
            energy.backward()
            torch.nn.utils.clip_grad_norm_([X_in, E_in], max_norm=self.clip_grad_max)

            X_guided = X_in - lr * X_in.grad
            E_guided = E_in - lr * E_in.grad
            E_guided = (E_guided + E_guided.transpose(1, 2)) * 0.5

            X_guided = F.softmax(X_guided, dim=-1)
            E_guided = F.softmax(E_guided, dim=-1)
            E_guided = E_guided / E_guided.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        return X_guided.detach(), E_guided.detach()
