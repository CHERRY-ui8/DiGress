import torch
import torch.nn.functional as F

from src.guidance.energies.base import BaseEnergy


class CosineReferenceEnergy(BaseEnergy):
    """
    Placeholder energy: 1 - cosine similarity between a masked mean atom-type
    vector and a reference fingerprint (same dimension as atom-type classes).
    """

    def __init__(self, reference_fp: torch.Tensor):
        """
        :param reference_fp: (dx,) or (1, dx) non-negative or arbitrary; normalized internally.
        """
        super().__init__()
        if reference_fp.dim() == 2:
            reference_fp = reference_fp.squeeze(0)
        self.register_buffer("reference_fp", reference_fp)

    def forward(self, X_soft: torch.Tensor, E_soft: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        del E_soft  # unused for this placeholder
        mask = node_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        mol_vec = (X_soft * mask).sum(dim=1) / denom.unsqueeze(-1)
        ref = self.reference_fp.to(device=mol_vec.device, dtype=mol_vec.dtype).unsqueeze(0)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        mol_vec = mol_vec / mol_vec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = F.cosine_similarity(mol_vec, ref, dim=-1)
        return 1.0 - sim
