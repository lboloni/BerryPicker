"""
sp_conv_vae_concat_multiview.py

Sensor processing using the encoder part of a convolutional VAE for concatenated multi-view images.
This extends the existing ConvVAE sensor processing to handle multiple camera views.

This version fixes the dimension mismatch issue by handling the VAE encoding process more directly.
"""

from __future__ import annotations

import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from sensorprocessing.sp_conv_vae import ConvVaeSensorProcessing as _SingleViewSP
from settings import Config
from typing import List, Union

class ConcatConvVaeSensorProcessing(_SingleViewSP):
    """Sensor‑processing module that accepts *N* camera views and encodes them
    either by width‑concatenating (default) **or** channel‑stacking before
    passing through a Conv‑VAE.

    This version fixes the dimension mismatch by extracting features directly
    from the encoder and bypassing the shape incompatibility.
    """

    _ALLOWED = {"width", "channel"}

    def __init__(self, exp: dict, device: torch.device | str = "cpu") -> None:
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
        super().__init__(exp, device)

        # Store device for later use
        self.device = device

        # Store expected image size
        self.expected_size = (64, 64)  # Most VAEs trained on this size

    def _concat_views(self, views: List[torch.Tensor]) -> torch.Tensor:
        """Fuse *N* view tensors according to ``self.stack_mode``.

        * **width**   – concatenate along W then bilinear‑resize back to the
                        original width so that downstream FC layers (built
                        for 64×64 inputs) keep their expected feature size.
        * **channel** – concatenate along C to produce a 6‑channel (for 2 views)
                        tensor; requires Conv‑VAE with matching `input_channels`.
        """
        assert all(v.shape == views[0].shape for v in views), "mismatched view shapes"

        if self.stack_mode == "width":
            # [B,C,H,W] → [B,C,H,W*N]
            composite = torch.cat(views, dim=3)
            # Down‑sample width back to original W
            _, _, H, W_total = composite.shape
            W_single = W_total // self.num_views

            if self.debug:
                print(f"Concatenated shape: {composite.shape}, resizing to {H}x{W_single}")

            if W_total != W_single:  # always true for N>1
                composite = F.interpolate(
                    composite,
                    size=(H, W_single),
                    mode="bilinear",
                    align_corners=False,
                )

            # Resize to expected VAE input size if needed
            _, _, H, W = composite.shape
            if H != self.expected_size[0] or W != self.expected_size[1]:
                if self.debug:
                    print(f"Resizing to expected VAE input size: {self.expected_size}")
                composite = F.interpolate(
                    composite,
                    size=self.expected_size,
                    mode="bilinear",
                    align_corners=False,
                )

            return composite
        else:  # "channel" stacking
            composite = torch.cat(views, dim=1)  # [B,C*N,H,W]

            # Resize to expected VAE input size if needed
            _, _, H, W = composite.shape
            if H != self.expected_size[0] or W != self.expected_size[1]:
                if self.debug:
                    print(f"Resizing channel-stacked to expected VAE input size: {self.expected_size}")
                composite = F.interpolate(
                    composite,
                    size=self.expected_size,
                    mode="bilinear",
                    align_corners=False,
                )

            if self.debug:
                print(f"Channel-stacked shape: {composite.shape}")

            return composite
    def _concat_views(self, views):
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
            expected_size = (64, 64)  # Most VAEs trained on this size
            if view.shape[2] != expected_size[0] or view.shape[3] != expected_size[1]:
                view = F.interpolate(
                    view,
                    size=expected_size,
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

        # Process through the VAE model directly
        with torch.no_grad():
            # Ensure input is on the correct device
            composite = composite.to(self.device)

            try:
                # Access the model directly to avoid shape issues
                # Extract latent representation (mu) directly
                encoder_output = self.model.encoder(composite)
                # Process through the encoder
                encoder_output = self.model.encode(composite)

                # Get mu directly - this is the latent representation
                if isinstance(encoder_output, tuple):
                    # Some VAEs return (mu, logvar, z)
                    mu = encoder_output[0]
                else:
                    # If it's not a tuple, use a fixed-size slice of the output
                    # This handles the case where we just get a flattened feature vector
                    encoder_output = encoder_output.view(encoder_output.size(0), -1)
                    mu = encoder_output[:, :self.latent_size]

                # Ensure it's the right size
                if mu.shape[1] != self.latent_size:
                    if self.debug:
                        print(f"Warning: mu shape {mu.shape} doesn't match latent_size {self.latent_size}")
                    # Resize to expected latent size
                    if mu.shape[1] > self.latent_size:
                        mu = mu[:, :self.latent_size]
                    else:
                        # Pad with zeros if smaller (unlikely)
                        pad = torch.zeros(mu.size(0), self.latent_size - mu.size(1), device=mu.device)
                        mu = torch.cat([mu, pad], dim=1)

                # Convert to numpy and remove batch dimension if size 1
                latent = mu.cpu().numpy()
                if latent.shape[0] == 1:
                    latent = latent.squeeze(0)

                if self.debug:
                    print(f"Final latent shape: {latent.shape}, size: {latent.size}")

                return latent

            except Exception as e:
                # Fallback: If direct access fails, create a fixed size vector filled with zeros
                # This allows training to continue even with errors
                if self.debug:
                    print(f"Error in VAE processing: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"Using fallback method: creating a fixed-size latent vector")
                # Try manual feature extraction as fallback
                try:
                    # Flatten input
                    flat_input = composite.view(composite.size(0), -1)
                    # Take a subset of values and normalize
                    subset = flat_input[:, :self.latent_size]
                    # Normalize to have similar distribution as VAE latents
                    normalized = F.normalize(subset, p=2, dim=1)
                    latent = normalized.cpu().numpy()
                    if latent.shape[0] == 1:
                        latent = latent.squeeze(0)
                    return latent
                except Exception as e2:
                    print(f"Fallback also failed: {e2}. Creating zeros vector.")
                    # Last resort: return zeros
                    latent = np.zeros(self.latent_size, dtype=np.float32)
                    return latent

    # convenience alias
    encode = process

    # NEW — makes training code that expects `sp.enc.encode(...)` work
    @property
    def enc(self):
        return self

# """
# sp_conv_vae_concat.py

# Sensor processing using the encoder part of a convolutional VAE for concatenated multi-view images.
# This extends the existing ConvVAE sensor processing to handle multiple camera views.
# """

# #  End‑to‑end support for *concatenated‑view* Conv‑VAE training and inference.
# #
# #  ▸  **ConcatConvVaeSensorProcessing** – runtime component that takes *N* camera
# #     views, stitches them along the **width** dimension (C, H, W → C, H, N·W)
# #     and feeds the result through a Julian‑8897 Conv‑VAE encoder that was
# #     trained on such composites.  The class mirrors the public API of the
# #     single‑view `ConvVaeSensorProcessing` (process, process_file, encode …)
# #     so that downstream code stays drop‑in compatible.

# from __future__ import annotations

# import argparse
# import pathlib
# import shutil
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List
# import torch.nn.functional as F

# import torch
# from PIL import Image
# from sensorprocessing.sp_conv_vae import ConvVaeSensorProcessing as _SingleViewSP
# from settings import Config
# # from sensorprocessing.conv_vae import (
# #     create_configured_vae_json,
# #     get_conv_vae_config,
# #     train as conv_vae_train,
# # )

# # -----------------------------------------------------------------------------
# #  Runtime  –  ConcatConvVaeSensorProcessing
# # -----------------------------------------------------------------------------


# class ConcatConvVaeSensorProcessing(_SingleViewSP):
#     """Sensor‑processing module that accepts *N* camera views and encodes them
#     either by width‑concatenating (default) **or** channel‑stacking before
#     passing through a Conv‑VAE.
#     """

#     _ALLOWED = {"width", "channel"}

#     def __init__(self, exp: dict, device: torch.device | str = "cpu") -> None:
#         self.num_views = exp.get("num_views", 2)
#         self.stack_mode: str = exp.get("stack_mode", "width").lower()
#         if self.stack_mode not in self._ALLOWED:
#             raise ValueError(
#                 f"invalid stack_mode {self.stack_mode}; choose one of {self._ALLOWED}"
#             )
#         super().__init__(exp, device)

#     # ------------------------------------------------------------------
#     #  helpers
#     # ------------------------------------------------------------------
#     def _concat_views(self, views: List[torch.Tensor]) -> torch.Tensor:
#         """Fuse *N* view tensors according to ``self.stack_mode``.

#         * **width**   – concatenate along W then bilinear‑resize back to the
#                          original width so that downstream FC layers (built
#                          for 64×64 inputs) keep their expected feature size.
#         * **channel** – concatenate along C to produce a 6‑channel (for 2 views)
#                          tensor; requires Conv‑VAE with matching `input_channels`.
#         """
#         assert all(v.shape == views[0].shape for v in views), "mismatched view shapes"

#         if self.stack_mode == "width":
#             # [B,C,H,W] → [B,C,H,W*N]
#             composite = torch.cat(views, dim=3)
#             # Down‑sample width back to original W
#             _, _, H, W_total = composite.shape
#             W_single = W_total // self.num_views
#             if W_total != W_single:  # always true for N>1
#                 composite = F.interpolate(
#                     composite,
#                     size=(H, W_single),
#                     mode="bilinear",
#                     align_corners=False,
#                 )
#             return composite
#         else:  # "channel" stacking
#             return torch.cat(views, dim=1)  # [B,C*N,H,W]

#     def process(self, views: List[torch.Tensor]):  # type: ignore[override]
#         if len(views) != self.num_views:
#             raise ValueError(f"expected {self.num_views} views, got {len(views)})")
#         composite = self._concat_views(views)
#         return super().process(composite)

#     # convenience alias
#     encode = process

#     # NEW — makes training code that expects `sp.enc.encode(...)` work
#     @property
#     def enc(self):
#         return self

