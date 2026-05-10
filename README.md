# RATDS v4.0 — Refraction Amplitude Terrain Detection System

A material detection system for RGB images built on fractal theory, wave physics, and seismic refraction mathematics. **No transformers, no neural networks** — pure equation-based feature extraction with a classical classifier head.

**v4.0 Architectural Change:** Material profiles are now stored as **JSON files** in the `profiles/` directory. You can calibrate any new material by running `calibrate.py` on a folder of images — no code changes required.

---

## Project Structure

```
ratds/
├── profiles/                 # JSON material profiles (auto-discovered)
│   ├── silver_ore.json      # ← Calibrated from real photo
│   ├── gold_ore.json        # ← Calibrated from real photo
│   ├── copper_ore.json      # ← Estimated (calibrate to improve)
│   ├── wood.json            # ← Estimated
│   └── ...
├── calibrate.py             # Generate new profiles from images
├── ratds_core.py            # 2D spectrum, illumination norm, fractal dims
├── ratds_classifier.py      # JSON profile loader, rule-based + k-NN classifier
├── ratds_train.py           # Synthetic data generator (reads profiles/)
├── ratds_viz.py             # Visualization suite
├── main.py                  # Entrypoint
└── README.md                # This file
```

---

## Installation

```bash
pip install numpy scipy scikit-learn pillow matplotlib
```

---

## Quick Start

### Analyze an Image

```bash
python main.py your_image.jpg
```

### Calibrate a New Material (No Code Changes!)

```bash
# Single image
python calibrate.py --name copper_ore --input ./my_copper_photo.jpg

# Directory of images
python calibrate.py --name oak_wood --input ./oak_samples/ --patch-size 128

# Custom output directory
python calibrate.py --name titanium --input ./ti_photos/ --output ./my_profiles/
```

The script analyzes all patches, computes feature distributions, and writes `profiles/{name}.json`. The system automatically discovers it on the next run.

### Python API

```python
from PIL import Image
import numpy as np
from ratds_core import aggregate_patch_features
from ratds_classifier import classify_material, load_profiles

# Load profiles (auto-discovers all JSON files in profiles/)
load_profiles()

img = np.array(Image.open("nuggets.jpg").convert("RGB"))
feats = aggregate_patch_features(img, patch_size=128, stride=64, use_2d=True)
predictions = classify_material(feats, top_k=3)

for material, conf in predictions:
    print(f"{material}: {conf:.1%}")
```

---

## JSON Profile Format

Each material is a JSON file in `profiles/`:

```json
{
  "name": "silver_ore",
  "description": "Raw silver nuggets",
  "display_color": "#78909C",
  "calibrated": true,
  "calibration_date": "2026-05-10T14:42:00",
  "n_images": 1,
  "n_patches": 126,
  "features": {
    "beta": {"lo": 1.5, "hi": 4.5, "mean": 2.84, "std": 0.62, "p5": 1.93, "p95": 4.35},
    "H_hurst": {"lo": -0.1, "hi": 1.2, "mean": 0.42, "std": 0.35},
    "spectral_flatness": {"lo": 0.0, "hi": 0.06, "mean": 0.006, "std": 0.003},
    "beta_ratio_B": {"lo": 0.95, "hi": 1.05, "mean": 0.998, "std": 0.002}
  },
  "synthetic": {
    "color_tint": [160, 165, 170],
    "H_range": [0.1, 0.6]
  }
}
```

### Adding a Material Manually

Create `profiles/my_material.json`:

```json
{
  "name": "my_material",
  "description": "My custom material",
  "display_color": "#FF5733",
  "calibrated": false,
  "features": {
    "beta": {"lo": 1.0, "hi": 3.0},
    "H_hurst": {"lo": 0.0, "hi": 0.5},
    "D_mean": {"lo": 1.5, "hi": 2.0},
    "D_corr": {"lo": 1.0, "hi": 2.0},
    "contrast": {"lo": 10, "hi": 50},
    "spectral_flatness": {"lo": 0.0, "hi": 0.10},
    "beta_ratio_B": {"lo": 0.90, "hi": 1.05}
  },
  "synthetic": {
    "color_tint": [200, 200, 200],
    "H_range": [0.1, 0.5]
  }
}
```

Restart `main.py` — it will auto-discover the new profile.

---

## Calibration Best Practices

1. **Use 5–20 images per material** under varying lighting and angles
2. **Keep patch_size = 128** for most materials
3. **Use `--margin 0.15`** for materials with high variability
4. **Calibrate from the same camera/lens** you will use for detection
5. **Review the JSON** after calibration — tighten ranges if needed

---

## How It Works

### Pipeline

```
RGB Image
    ↓
[Illumination Normalization]  ← Gaussian high-pass filter
    ↓
[2D Radial Power Spectrum]    ← fft2 → radial average → log-log fit
    ↓
[Fractal Dimensions]          ← VFD_2d, Divider, Box-counting, Correlation
    ↓
[Spectral Signature]          ← σ_β (flatness), ρ_B (blue ratio)
    ↓
[Multi-patch Aggregation]     ← Tile → compute per patch → median
    ↓
[JSON Profile Matching]       ← Match against profiles/*.json
    ↓
Material Label + Confidence
```

### Key Features

| Feature | Physical Meaning | Discriminates |
|---------|-----------------|---------------|
| **β₂d** | 2D spectral exponent | Roughness (smooth = high β) |
| **H** | Hurst exponent | Self-similarity |
| **σ_β** | Spectral flatness | Metal type (silver = flat, gold = uneven) |
| **ρ_B** | Blue-channel ratio | Color signature (gold absorbs blue) |
| **D_mean** | Mean fractal dimension | Overall texture complexity |

---

## Performance

| Operation | Time (128×128 patch, CPU) |
|-----------|---------------------------|
| Illumination normalization | ~2 ms |
| 2D radial spectrum | ~2 ms |
| Fractal dimensions (all) | ~8 ms |
| Spectral signature | <1 ms |
| **Total per patch** | **~13 ms** |
| Multi-patch (126 patches) | ~1.6 s |
| JSON profile matching | <1 ms |

---

## Open Datasets for Calibration

| Dataset | Classes | Best For | Link |
|---------|---------|----------|------|
| **KTH-TIPS** | 10 materials | Multi-scale material appearance | [Download](https://www.csc.kth.se/cvap/databases/kth-tips/) |
| **DTD** | 47 textures | Broad texture vocabulary | [Download](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |
| **FMD** | 10 materials | Real-world material photos | [Project](https://people.csail.mit.edu/celiu/CVPR2010/FMD/) |
| **MINC** | 23 materials | Large-scale patches | [Project](http://opensurfaces.cs.cornell.edu/publications/minc/) |

---

## License

MIT. Use freely for research and commercial applications.
