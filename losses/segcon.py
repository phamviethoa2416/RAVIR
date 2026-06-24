from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegCon(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        num_anchors: int = 256,
        num_positives: int = 4,
        num_negatives: int = 16,
        negative_radius: int = 20,
        reference_pool_size: int = 4096,
        confidence_gated: bool = True,
        confidence_gamma: float = 1.0,
        confidence_detach: bool = True,
        confidence_min_weight: float = 0.05,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.num_anchors = num_anchors
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.negative_radius = negative_radius
        self.reference_pool_size = reference_pool_size
        self.confidence_gated = confidence_gated
        self.confidence_gamma = confidence_gamma
        self.confidence_detach = confidence_detach
        self.confidence_min_weight = confidence_min_weight

    def forward(
        self,
        embedding: torch.Tensor,
        branch_labels: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ):
        if self.confidence_gated and confidence is None:
            raise ValueError("Confidence is required when enable confidence gated")
        if confidence is not None and self.confidence_detach:
            confidence = confidence.detach()

        per_anchor: list[torch.Tensor] = []
        for b in range(embedding.shape[0]):
            v = self.segcon_loss(
                embedding=embedding[b],
                branch=branch_labels[b],
                target=targets[b],
                confidence=confidence[b] if confidence is not None else None,
            )
            if v is not None:
                per_anchor.append(v)

        if not per_anchor:
            return embedding.new_zeros(())

        return torch.cat(per_anchor).mean()

    def segcon_loss(self, embedding, branch, target, confidence):
        D, Hp, Wp = embedding.shape
        H, W = branch.shape
        device = embedding.device

        branch_flat = branch.reshape(-1)
        target_flat = target.reshape(-1)
        confidence_flat = confidence.reshape(-1) if confidence is not None else None

        candidate = (target_flat > 0) & (branch_flat > 0)
        num_candidates = int(candidate.sum().item())
        if num_candidates < 2:
            return None
        candidate_idx = torch.nonzero(candidate, as_tuple=False).squeeze(1)

        if self.confidence_gated and confidence_flat is not None:
            c = confidence_flat[candidate_idx].clamp(0.0, 1.0)
            w = (1.0 - c).pow(self.confidence_gamma) + self.confidence_min_weight
        else:
            w = torch.ones(num_candidates, device=device)
        num_anchors = min(self.num_anchors, num_candidates)
        anchor_gidx = candidate_idx[
            torch.multinomial(w, num_anchors, replacement=False)
        ]

        pool_size = min(self.reference_pool_size, num_candidates)
        extra = candidate_idx[torch.randperm(num_candidates, device=device)[:pool_size]]
        pool_gidx = torch.cat([anchor_gidx, extra])

        anchor_branch = branch_flat[anchor_gidx]
        anchor_class = target_flat[anchor_gidx]
        anchor_y = (anchor_gidx // W).float()
        anchor_x = (anchor_gidx % W).float()

        pool_branch = branch_flat[pool_gidx]
        pool_class = target_flat[pool_gidx]
        pool_y = (pool_gidx // W).float()
        pool_x = (pool_gidx % W).float()

        positive_mask = (pool_branch.unsqueeze(0) == anchor_branch.unsqueeze(1)) & (
            pool_gidx.unsqueeze(0) != anchor_gidx.unsqueeze(1)
        )

        opposite = (3 - anchor_class).unsqueeze(1)
        radius = self.negative_radius * (H / Hp)
        dy = (pool_y.unsqueeze(0) - anchor_y.unsqueeze(1)).abs()
        dx = (pool_x.unsqueeze(0) - anchor_x.unsqueeze(1)).abs()
        negative_mask = (
            (pool_class.unsqueeze(0) == opposite) & (dy <= radius) & (dx <= radius)
        )

        valid = positive_mask.any(1) & negative_mask.any(1)
        if not bool(valid.any()):
            return None

        anchor_gidx = anchor_gidx[valid]
        positive_mask = positive_mask[valid]
        negative_mask = negative_mask[valid]

        positive_idx = torch.multinomial(
            positive_mask.float(), self.num_positives, replacement=True
        )
        negative_idx = torch.multinomial(
            negative_mask.float(), self.num_negatives, replacement=True
        )

        anchor_embedding = self.sample_embedding(embedding, anchor_gidx, H, W)
        pool_embedding = self.sample_embedding(embedding, pool_gidx, H, W)
        positive_embedding = pool_embedding[positive_idx]
        negative_embedding = pool_embedding[negative_idx]

        sim_positive = (
            torch.einsum("ad,apd->ap", anchor_embedding, positive_embedding)
            / self.temperature
        )
        sim_negative = (
            torch.einsum("ad, akd->ak", anchor_embedding, negative_embedding)
            / self.temperature
        )
        log_z = torch.logsumexp(
            torch.cat([sim_positive, sim_negative], dim=1), dim=1, keepdim=True
        )
        return -(sim_positive - log_z).mean(dim=1)

    def sample_embedding(self, embedding, gidx, H, W):
        D = embedding.shape[0]
        y = (gidx // W).float()
        x = (gidx % W).float()
        gy = 2.0 * y / max(H - 1, 1) - 1.0
        gx = 2.0 * x / max(W - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2)
        sample = F.grid_sample(
            embedding.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        sample = sample.view(D, -1).t().contiguous()
        return F.normalize(sample, dim=1, eps=1e-6)
