"""
sp_conv_vae_concat_multiview.py

Sensor processing using the encoder part of a convolutional VAE for concatenated multi-view images.
This extends the existing ConvVAE sensor processing to handle multiple camera views.

This version fixes the dimension mismatch issue by handling the VAE encoding process more directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from sensorprocessing.sp_conv_vae import ConvVaeSensorProcessing as _SingleViewSP
from sensorprocessing.sensor_processing import MultiViewDemonstrationProcessing

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

class ConcatConvVaeSensorProcessing(
    MultiViewDemonstrationProcessing, _SingleViewSP
):
    """Sensor‑processing module that accepts *N* camera views and encodes them
    either by width‑concatenating (default) **or** channel‑stacking before
    passing through a Conv‑VAE.

    This version fixes the dimension mismatch by extracting features directly
    from the encoder and bypassing the shape incompatibility.
    """

    _ALLOWED = {"width", "channel"}

    def __init__(self, exp: dict) -> None:
        self.num_views = exp.get("num_views", 2)
        self.stack_mode: str = exp.get("stack_mode", "width").lower()
        self.latent_size = exp.get("latent_size", 128)
        self.debug = exp.get("debug", False)

        if self.debug:
            print(f"Initializing ConcatConvVaeSensorProcessing:")
            print(f"  num_views: {self.num_views}")
            print(f"  stack_mode: {self.stack_mode}")
            print(f"  latent_size: {self.latent_size}")

        if self.stack_mode not in self._ALLOWED:
            raise ValueError(
                f"invalid stack_mode {self.stack_mode}; choose one of {self._ALLOWED}"
            )
        super().__init__(exp)

        # Store expected image size
        self.expected_size = (64, 64)  # Most VAEs trained on this size

    def _concat_views(self, views: List[torch.Tensor]) -> torch.Tensor:
        """Fuse *N* view tensors according to `self.stack_mode` with enhanced dimension handling."""
        # Ensure all views have 4 dimensions [B,C,H,W]
        processed_views = []
        for i, view in enumerate(views):
            # Check if view is missing batch dimension
            if len(view.shape) == 3:  # [C,H,W] format
                view = view.unsqueeze(0)  # Add batch dimension -> [1,C,H,W]
                if self.debug:
                    print(f"Added batch dimension to view {i}, new shape: {view.shape}")

            # Ensure all views have the expected size
            if view.shape[2:] != self.expected_size:
                view = F.interpolate(
                    view,
                    size=self.expected_size,
                    mode="bilinear",
                    align_corners=False
                )

            processed_views.append(view)

        # Now check that shapes match
        if not all(v.shape == processed_views[0].shape for v in processed_views):
            shapes = [v.shape for v in processed_views]
            raise ValueError(f"Mismatched view shapes: {shapes}")

        if self.stack_mode == "width":
            # [B,C,H,W] → [B,C,H,W*N]
            try:
                composite = torch.cat(processed_views, dim=3)

                # Down‑sample width back to original W
                _, _, H, W_total = composite.shape
                W_single = W_total // self.num_views

                if self.debug:
                    print(f"Concatenated shape: {composite.shape}, resizing to {H}x{W_single}")

                if W_total != W_single:  # Always true for N>1
                    composite = F.interpolate(
                        composite,
                        size=(H, W_single),
                        mode="bilinear",
                        align_corners=False,
                    )

                return composite
            except Exception as e:
                if self.debug:
                    print(f"Error during width concatenation: {e}")
                    print(f"View shapes: {[v.shape for v in processed_views]}")
                raise

        else:  # "channel" stacking
            try:
                composite = torch.cat(processed_views, dim=1)  # [B,C*N,H,W]
                return composite
            except Exception as e:
                if self.debug:
                    print(f"Error during channel concatenation: {e}")
                    print(f"View shapes: {[v.shape for v in processed_views]}")
                raise

    def process(self, views: Union[List[torch.Tensor], torch.Tensor]) -> np.ndarray:
        """Process multiple views or a single preprocessed tensor to produce a latent representation.

        Args:
            views: Either a list of image tensors from different views,
                or a single tensor with already concatenated views.

        Returns:
            Latent representation as a numpy array
        """

        # Handle both list of views and single tensor formats
        if isinstance(views, list):
            # Case 1: List of views
            if len(views) != self.num_views:
                raise ValueError(f"expected {self.num_views} views, got {len(views)})")
            # Concatenate views
            composite = self._concat_views(views)
        else:
            # Case 2: Single tensor (already concatenated views)
            composite = views

        if not isinstance(composite, torch.Tensor):
            raise TypeError(
                "views must be a list of tensors or a preprocessed tensor"
            )

        # Process through the VAE model directly.  Do not manufacture a
        # substitute latent when the encoder is incompatible with the input or
        # checkpoint: callers must not continue with data that did not come
        # from the trained VAE.
        with torch.no_grad():
            # Ensure input is on the correct device
            composite = composite.to(Config().runtime["device"])

            try:
                encoder_output = self.model.encode(composite)
            except Exception as error:
                raise RuntimeError(
                    "Conv-VAE encoding failed for composite input "
                    f"with shape {tuple(composite.shape)}"
                ) from error

            if not isinstance(encoder_output, (tuple, list)):
                raise TypeError(
                    "Conv-VAE encode() must return a tuple or list whose first "
                    "item is the latent mean"
                )
            if not encoder_output or not isinstance(encoder_output[0], torch.Tensor):
                raise TypeError(
                    "Conv-VAE encode() must return a tensor latent mean as its "
                    "first item"
                )

            mu = encoder_output[0]
            expected_shape = (composite.size(0), self.latent_size)
            if tuple(mu.shape) != expected_shape:
                raise ValueError(
                    "Conv-VAE latent mean has shape "
                    f"{tuple(mu.shape)}; expected {expected_shape}"
                )

            latent = mu.cpu().numpy()
            if latent.shape[0] == 1:
                latent = latent.squeeze(0)

            if self.debug:
                print(f"Final latent shape: {latent.shape}, size: {latent.size}")

            return latent

    # Convenience alias.
    encode = process

    # Makes training code that expects `sp.enc.encode(...)` work.
    @property
    def enc(self):
        return self
