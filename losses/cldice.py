from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iterations: int = 10):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        opened = soft_open(mask)
        skel = F.relu(mask - opened)

        for _ in range(self.num_iterations):
            mask = soft_erode(mask)
            opened = soft_open(mask)
            delta = F.relu(mask - opened)
            skel = skel + F.relu(delta - skel * delta)

        return skel


class ClDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        num_iterations: int = 10,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iterations=num_iterations)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        with torch.no_grad():
            gt_onehot = (
                F.one_hot(targets.long(), self.num_classes).permute(0, 3, 1, 2).float()
            )

        probs_fg = probs[:, 1:]
        gt_fg = gt_onehot[:, 1:]

        skel_pred = self.soft_skeletonize(probs_fg)
        with torch.no_grad():
            skel_gt = self.soft_skeletonize(gt_fg)

        t_prec = ((skel_pred * gt_fg).sum(dim=(2, 3)) + self.smooth) / (
            skel_pred.sum(dim=(2, 3)) + self.smooth
        )

        t_sens = ((skel_gt * probs_fg).sum(dim=(2, 3)) + self.smooth) / (
            skel_gt.sum(dim=(2, 3)) + self.smooth
        )

        cl_dice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-8)

        return 1.0 - cl_dice.mean()
