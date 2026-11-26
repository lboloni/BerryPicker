# Experimental Domain Randomization Preprocessing (`_2026_11_domain_randomization`)

Standalone YAML-configured domain randomization package for visual proprioception experiments, testing whether segmented learning beats domain randomization across clean and cluttered/shifted test suites.

All `.py` modules in this package keep **zero imports** from BerryPicker core packages or `ExpRunFlow`.

## Quick Python Usage

```python
import numpy as np
from _experimental._2026_11_domain_randomization import DomainRandomizer

dr = DomainRandomizer.from_yaml(
    "src/_experimental/_2026_11_domain_randomization/recipes/dr_full.yaml"
)
out_rgb = dr(frame_rgb, mask=robot_mask, rng=np.random.default_rng(42))
```

## Included Recipes (`recipes/`)

- `seg_white.yaml` — Deterministic white-background fill (`background_replace: [255, 255, 255]`).
- `dr_photo.yaml` — Mask-free lighting gradient, white balance, color jitter, gamma, noise, blur, JPEG artifacts.
- `dr_bg_swap.yaml` — Mask-based weighted random synthetic backgrounds (`smooth_field`, `geometric_pattern`, `multi_scale_noise`, `solid_color`).
- `dr_full.yaml` — Composite `dr_bg_swap` + background distractors + lighting/color/noise + cutout (`random_erasing`).
- `eval_ood_clutter.yaml` — Disjoint held-out clutter + lighting shift stress test for `vp_testing`.

## Running the Comparison Flow (`Flow_DomainRandomization.ipynb`)

```bash
PY=/Users/lboloni/Documents/Develop/VirtualEnvs/BerryPicker/bin/python
OUT=/Users/lboloni/Documents/Develop/Data/BerryPicker-Flows/DomainRand_smoke_01
mkdir -p "$OUT"
$PY -m papermill \
  --cwd src/_experimental/_2026_11_domain_randomization \
  src/_experimental/_2026_11_domain_randomization/Flow_DomainRandomization.ipynb \
  "$OUT/Flow_DomainRandomization.ipynb" \
  -k berrypicker
```

## Unit Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  /Users/lboloni/Documents/Develop/VirtualEnvs/BerryPicker/bin/python -m pytest \
  src/test/_experimental/_2026_11_domain_randomization -v
```
