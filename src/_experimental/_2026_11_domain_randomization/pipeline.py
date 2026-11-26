"""YAML-configured domain randomization pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .transforms import OP_REGISTRY, apply_op, step_requires_mask


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Load and validate a domain randomization YAML recipe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Recipe YAML must be a mapping, got {type(cfg).__name__}")
    return cfg


class DomainRandomizer:
    """Composable, YAML-configured RGB frame domain randomizer.

    Usage::

        dr = DomainRandomizer.from_yaml("recipes/dr_full.yaml")
        out_rgb = dr(frame_rgb, mask=mask, rng=np.random.default_rng(0))
    """

    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.name: str = str(self.config["name"])
        self.seed: int = int(self.config.get("seed", 0))
        self.copies_per_episode: int = int(self.config.get("copies_per_episode", 1))
        if self.copies_per_episode < 1:
            raise ValueError(f"copies_per_episode must be >= 1, got {self.copies_per_episode}")

        raw_pipeline = self.config.get("pipeline", [])
        if not isinstance(raw_pipeline, list):
            raise ValueError("Recipe 'pipeline' must be a list of step mappings")

        self.steps: list[dict[str, Any]] = []
        for idx, step in enumerate(raw_pipeline):
            if not isinstance(step, dict) or "op" not in step:
                raise ValueError(f"Pipeline step {idx} must be a mapping with an 'op' key")
            op_name = step["op"]
            if op_name not in OP_REGISTRY:
                raise KeyError(
                    f"Unknown op '{op_name}' in recipe '{self.name}' step {idx}. "
                    f"Available: {sorted(OP_REGISTRY)}"
                )
            p = float(step.get("p", 1.0))
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Step {idx} ('{op_name}') probability p={p} out of [0, 1]")
            self.steps.append(dict(step))

        self.requires_mask: bool = bool(
            self.config.get("requires_mask", any(step_requires_mask(s) for s in self.steps))
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DomainRandomizer":
        return cls(load_recipe(path))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DomainRandomizer":
        return cls(config)

    def frame_rng(self, episode_idx: int, frame_idx: int, copy_idx: int = 0) -> np.random.Generator:
        """Return a deterministic per-frame RNG derived from recipe seed and indices."""
        seed_seq = np.random.SeedSequence([self.seed, int(copy_idx), int(episode_idx), int(frame_idx)])
        return np.random.default_rng(seed_seq)

    def __call__(
        self,
        frame_rgb: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(self.seed)
        out = frame_rgb.copy()
        for step in self.steps:
            p = float(step.get("p", 1.0))
            if p < 1.0 and rng.random() >= p:
                continue
            params = {k: v for k, v in step.items() if k not in ("op", "p")}
            out = apply_op(step["op"], out, rng, mask=mask, params=params)
        return out

    def as_pil_transform(
        self,
        *,
        mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        seed: int | None = None,
    ) -> Callable[[Any], Any]:
        """Return a PIL Image -> PIL Image transform adapter for torchvision Compose."""
        from PIL import Image

        rng = np.random.default_rng(self.seed if seed is None else seed)

        def _transform(pil_img: Any) -> Any:
            rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            m = mask_fn(rgb) if (mask_fn is not None and self.requires_mask) else None
            out = self(rgb, mask=m, rng=rng)
            return Image.fromarray(out)

        return _transform
