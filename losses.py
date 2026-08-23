"""Compatibility exports for the losses used by the paper-aligned model."""

from contrastive_learning import instance_contrastive_loss, temporal_contrastive_loss
from reconstruction import masked_reconstruction_loss

__all__ = [
    "instance_contrastive_loss",
    "temporal_contrastive_loss",
    "masked_reconstruction_loss",
]
