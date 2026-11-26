"""Standalone YAML-configured domain randomization preprocessing package (2026-11).

All modules in this package have zero imports from BerryPicker core packages or
``ExpRunFlow`` and operate on ``(H, W, 3) uint8`` RGB frames.
"""

from .pipeline import DomainRandomizer, load_recipe
from .transforms import MASK_REQUIRED_OPS, OP_REGISTRY, apply_op

__all__ = [
    "DomainRandomizer",
    "MASK_REQUIRED_OPS",
    "OP_REGISTRY",
    "apply_op",
    "load_recipe",
]
