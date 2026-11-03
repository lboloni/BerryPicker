#!/usr/bin/env python3
"""
Process 112 Camera Data for Hemisphere Visualization
Correctly reads MSE values from the vp_vgg19_128_0001 column
"""

import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from hemisphere_heatmap import HemisphereHeatmap
from hemisphere_with_robot import plot_hemisphere_transparent_robot
from hemisphere_with_robot import (
    plot_hemisphere_transparent_robot,
    plot_hemisphere_with_robot_topdown,
    plot_hemisphere_with_robot_combined
)
def make_height_cvpr_2x3_across_models(
        output_root,
        model_columns,
        height_png_name="height_robot_topdown_cvpr.png",
        out_name="height_across_models_cvpr_2x3.png",
        csv_name="height_hemisphere_data.csv",
        titles=("VAE-128","VAE-256","VGG19-128","VGG19-256","ResNet50-128","ResNet50-256"),
    ):
    """
    Build a CVPR-style 2×3 grid using panels saved in each model's output dir.

    Looks for:
        <OUTPUT_DIR>/<model_column>/<height_png_name>  (per-component topdown image)
        <OUTPUT_DIR>/<model_column>/<csv_name>         (per-component hemisphere CSV)

    By changing ``height_png_name`` and ``csv_name`` you can reuse this for:
        height, distance, heading, wrist_angle, wrist_rotation, gripper, ...
    """
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_root = Path(output_root)
    six_images = []
    csv_paths = []

    # Collect the six panel images (+ CSVs for vmin/vmax)
    for mc in model_columns:
        mdir = output_root / mc
        img_path = mdir / height_png_name
        if not img_path.exists():
            raise FileNotFoundError(f"Missing panel for {mc}: {img_path}")
        six_images.append(Image.open(img_path).convert("RGB"))

        csv_path = mdir / csv_name
        if csv_path.exists():
            csv_paths.append(csv_path)

    # Compute vmin/vmax from CSVs if available
    vmin = None
    vmax = None
    if csv_paths:
        import pandas as pd
        all_vals = []
        for p in csv_paths:
            df = pd.read_csv(p)
            if "mse" in df.columns:
                all_vals.extend(df["mse"].values.tolist())
        if all_vals:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))

    # Light crop of white margins per tile
    cropped = []
    for img in six_images:
        arr = np.asarray(img)
        thr = 252
        mask = ~((arr[..., 0] >= thr) & (arr[..., 1] >= thr) & (arr[..., 2] >= thr))
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size and cols.size:
            img = img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
        cropped.append(img)
    six_images = cropped

    # Layout
    nrows, ncols = 3, 2
    w, h = six_images[0].size
    pad = int(0.07 * w)
    title_h = int(0.18 * h)
    bar_h = int(0.22 * h)
    bar_gap = int(0.12 * h)

    canvas_w = ncols * w + (ncols - 1) * pad
    canvas_h = nrows * (h + title_h) + (nrows - 1) * pad + bar_gap + bar_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Paste 2×3
    idx = 0
    positions = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * (w + pad)
            y = r * (h + title_h + pad)
            canvas.paste(six_images[idx], (x, y))
            positions.append((x, y))
            idx += 1

    # Titles
    if titles is not None:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", size=max(12, h // 18))
        except Exception:
            font = ImageFont.load_default()
        for i, (x, y) in enumerate(positions):
            label = titles[i] if i < len(titles) else ""
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.text(
                (x + (w - tw) // 2, y + h + max(2, title_h // 5)),
                label,
                fill="black",
                font=font,
            )

    # Colorbar (visual consistency with 'hot_r')
    fig = plt.figure(figsize=(8, 0.6), dpi=300)
    ax = fig.add_axes([0.08, 0.35, 0.84, 0.3])
    if vmin is None or vmax is None:
        vmin_, vmax_ = 0.0, 1.0
    else:
        vmin_, vmax_ = vmin, vmax
    norm = Normalize(vmin=vmin_, vmax=vmax_)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="hot_r")
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("MSE", fontsize=20)

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        bar_path = tmp.name
    fig.savefig(bar_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)

    bar_img = Image.open(bar_path).convert("RGB")
    os.remove(bar_path)

    # Tight crop and resize colorbar, then paste
    arr = np.asarray(bar_img)
    thr = 252
    mask = ~((arr[..., 0] >= thr) & (arr[..., 1] >= thr) & (arr[..., 2] >= thr))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size and cols.size:
        bar_img = bar_img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    bar_img = bar_img.resize((canvas_w, bar_h), Image.Resampling.LANCZOS)
    canvas.paste(bar_img, (0, canvas_h - bar_h))

    out_path = output_root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    if out_path.suffix.lower() == ".png":
        canvas.save(out_path.with_suffix(".pdf"))
    print(f"\n✓ Saved cross-model CVPR 2×3 to: {out_path}")

def main():
    """Main function to build cross-model 2×3 grids for multiple components."""

    ##############################################################################
    #                   UPDATE THESE PATHS FOR YOUR MACHINE                      #
    ##############################################################################
    BASE_FOLDER = "C:\\Users\\rkhan\\Downloads\\VGG_OK_All_models_data"
    CAMERA_PLACEMENTS_FILE = "C:\\Users\\rkhan\\Downloads\\camera_placements.txt"
    OUTPUT_DIR = "C:\\Users\\rkhan\\Downloads\\hemisphere_output_models_newVGG-last"

    # robot images (unchanged) – kept here in case you extend the script later
    ROBOT_IMAGE_PATH_3D = "C:\\Users\\rkhan\\Downloads\\robot.png"
    ROBOT_IMAGE_PATH_TOP = "C:\\Users\\rkhan\\Downloads\\robot_top.png"

    # all model columns we want to process (order = how they appear in the 2×3 grid)
    MODEL_COLUMNS = [
        "vp_conv_vae_128_0001",
        "vp_vgg19_128_0001",
        "vp_resnet50_128_0001",
        "vp_conv_vae_256_0001",
        "vp_vgg19_256_0001",
        "vp_resnet50_256_0001",
    ]

    # components to build cross-model 2×3 grids for
    COMPONENTS = [
        "height",
        "distance",
        "heading",
        "wrist_angle",
        "wrist_rotation",
        "gripper",
    ]

    # Build the six dirs from your existing OUTPUT_DIR and MODEL_COLUMNS
    dirs = [str(Path(OUTPUT_DIR) / col) for col in MODEL_COLUMNS]  # must be 6 in display order

    for comp in COMPONENTS:
        print(f"\n=== Building cross-model 2×3 grids for component: {comp} ===")

        png_name = f"{comp}_robot_topdown_cvpr.png"
        csv_name = f"{comp}_hemisphere_data.csv"

        # Version with labels under each panel (uses the PNGs)
        make_height_cvpr_2x3_across_models(
            output_root=OUTPUT_DIR,
            model_columns=MODEL_COLUMNS,
            height_png_name=png_name,
            out_name=f"{comp}_across_models_cvpr_2x3.png",
            csv_name=csv_name,
            titles=("VAE-128","VAE-256","VGG19-128","VGG19-256","ResNet50-128","ResNet50-256"),
        )

        # Optional: HemisphereHeatmap CSV-only version (no text labels under panels),
        # only runs if you've added HemisphereHeatmap.make_height_cvpr_2x3_from_folders.
        if hasattr(HemisphereHeatmap, "make_height_cvpr_2x3_from_folders"):
            HemisphereHeatmap.make_height_cvpr_2x3_from_folders(
                model_dirs=dirs,
                output_path=str(Path(OUTPUT_DIR) /
                                f"{comp}_across_models_cvpr_2x3-without-labels.png"),
                csv_name=csv_name,
                cmap="hot_r",
            )


if __name__ == "__main__":
    main()