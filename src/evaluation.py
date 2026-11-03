"""Sliding-window inference and Dice scoring, shared by training and benchmarking.

Why sliding windows?
--------------------
The network was trained on fixed ``roi_size`` crops (96^3 voxels), but a
whole CT volume after resampling is much larger, e.g. 250 x 250 x 120. Rather
than feeding it in one go (too much memory, and a shape the model never saw),
MONAI's ``sliding_window_inference`` tiles the volume into overlapping
``roi_size`` windows, runs the network on ``sw_batch_size`` windows at a time,
and blends the overlapping predictions with a weighted average.

Because every window has exactly the same shape, any backend that accepts a
fixed ``(sw_batch_size, 1, *roi_size)`` input works here - including a
TensorRT engine compiled for that one shape.
"""

from __future__ import annotations

from typing import Callable

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

Predictor = Callable[[torch.Tensor], torch.Tensor]


def predict_volume(
    predictor: Predictor,
    image: torch.Tensor,
    roi_size,
    sw_batch_size: int,
    overlap: float,
) -> torch.Tensor:
    """Run sliding-window inference over one (or a batch of) full volume(s).

    Returns raw logits with the same spatial size as ``image``.
    """
    return sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        mode="gaussian",  # weight window centres more than their edges when blending
    )


class DiceScorer:
    """Accumulate the spleen Dice score over several volumes.

    Dice = 2 * |prediction ∩ ground truth| / (|prediction| + |ground truth|),
    from 0 (no overlap) to 1 (perfect). Background is excluded, otherwise the
    huge number of correctly-predicted background voxels would inflate it.
    """

    def __init__(self, num_classes: int = 2):
        self.metric = DiceMetric(include_background=False, reduction="mean")
        self.to_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.to_label = AsDiscrete(to_onehot=num_classes)

    def __call__(self, logits: torch.Tensor, label: torch.Tensor) -> float:
        """Score one batch, add it to the running total, and return its Dice."""
        preds = [self.to_pred(x) for x in logits]
        labels = [self.to_label(y) for y in label]
        batch_dice = self.metric(y_pred=preds, y=labels)
        return float(batch_dice.nanmean().item())

    def aggregate(self) -> float:
        """Mean Dice over every volume scored since the last ``reset``."""
        return float(self.metric.aggregate().item())

    def reset(self) -> None:
        self.metric.reset()
