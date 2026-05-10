"""
RATDS Profile Calibration Tool
Generates material profiles from real images as JSON files.

Usage:
    python calibrate.py --name silver_ore --input ./my_silver_photos/ --output ./profiles/
    python calibrate.py --name gold_ore --input gold_nugget.jpg --output ./profiles/
    python calibrate.py --name wood --input ./wood_samples/ --output ./profiles/ --patch-size 128

The script analyzes all images, computes multi-patch features, and writes a JSON profile
with statistically derived feature ranges (5th–95th percentile ± margin).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ratds_core import compute_features, aggregate_patch_features, tile_image


def calibrate_material(name: str,
                        input_path: str,
                        output_dir: str = "profiles",
                        patch_size: int = 128,
                        stride: int = 64,
                        percentile_margin: float = 0.10,
                        display_color: str = None,
                        description: str = "",
                        color_tint: list = None):
    """
    Calibrate a material profile from one or more images.

    Args:
        name: Material identifier (e.g., 'silver_ore')
        input_path: Path to image file or directory of images
        output_dir: Where to write the JSON profile
        patch_size: Patch size for tiling
        stride: Stride for tiling
        percentile_margin: Extra margin around 5th–95th percentile range (0.1 = 10%)
        display_color: Hex color for visualization (auto-generated if None)
        description: Human-readable description
        color_tint: [R, G, B] tint for synthetic generation
    """
    input_path = Path(input_path)

    # Collect image paths
    if input_path.is_dir():
        image_paths = sorted([
            p for p in input_path.glob("*")
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        ])
    else:
        image_paths = [input_path]

    if not image_paths:
        raise ValueError(f"No images found at {input_path}")

    print(f"[calibrate] Material: {name}")
    print(f"[calibrate] Images: {len(image_paths)}")
    print(f"[calibrate] Patch size: {patch_size}, stride: {stride}")

    # Collect features from all patches across all images
    all_features = []
    for img_path in image_paths:
        img = np.array(Image.open(img_path).convert("RGB"))

        # Auto-remove uniform background before patch extraction
        from ratds_core import remove_background, tile_image_masked
        img_clean, mask = remove_background(img, variance_threshold=15.0)
        fg_ratio = mask.sum() / mask.size
        print(f"  [calibrate] {img_path.name}: foreground={fg_ratio:.1%}")

        # Use targeted tiling on foreground only
        patch_count = 0
        for r, c, patch in tile_image_masked(img_clean, mask, patch_size, stride,
                                              min_foreground_ratio=0.10):
            try:
                feats = compute_features(patch, use_2d=True)
                # Skip very low contrast patches (background residue)
                if feats.get('contrast', 0) < 1.0:
                    continue
                all_features.append(feats)
                patch_count += 1
            except Exception as e:
                pass

        print(f"  [calibrate]   Foreground patches: {patch_count}")

        if patch_count < 5:
            # Fallback: try with even lower threshold
            print(f"  [calibrate]   Fallback: using lower threshold")
            for r, c, patch in tile_image_masked(img_clean, mask, patch_size, stride,
                                                  min_foreground_ratio=0.01):
                try:
                    feats = compute_features(patch, use_2d=True)
                    if feats.get('contrast', 0) >= 1.0:
                        all_features.append(feats)
                        patch_count += 1
                except Exception:
                    pass
            print(f"  [calibrate]   After fallback: {patch_count} patches")

        if patch_count < 5:
            # Last resort: analyze full image
            print(f"  [calibrate]   Last resort: analyzing full image")
            for r, c, patch in tile_image(img, patch_size, stride):
                try:
                    feats = compute_features(patch, use_2d=True)
                    if feats.get('contrast', 0) >= 1.0:
                        all_features.append(feats)
                except Exception:
                    pass

    n_patches = len(all_features)
    if n_patches < 5:
        raise ValueError(f"Too few valid patches ({n_patches}). Check image size and patch parameters.")

    print(f"[calibrate] Valid patches: {n_patches}")

    # Compute statistics for each feature
    feature_names = [
        'beta', 'D_spectral', 'H_hurst', 'D_vfd', 'D_divider',
        'D_box', 'D_corr', 'D_mean', 'R_response',
        'beta_R', 'beta_G', 'beta_B',
        'spectral_flatness', 'beta_ratio_R', 'beta_ratio_G', 'beta_ratio_B', 'beta_spread',
        'contrast', 'mean_luminance'
    ]

    features_stats = {}
    for key in feature_names:
        vals = np.array([f[key] for f in all_features if key in f])
        if len(vals) == 0:
            continue
        p5, p95 = np.percentile(vals, [5, 95])
        margin = (p95 - p5) * percentile_margin
        lo = float(p5 - margin)
        hi = float(p95 + margin)
        # Ensure lo < hi and handle edge cases
        if lo >= hi:
            lo, hi = float(vals.min()), float(vals.max())
        features_stats[key] = {
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "p5": round(float(p5), 4),
            "p95": round(float(p95), 4)
        }

    # Auto-generate display color if not provided
    if display_color is None:
        # Generate distinct color from name hash
        import hashlib
        h = hashlib.md5(name.encode()).hexdigest()
        display_color = f"#{h[:6]}"

    # Auto-generate color tint from mean image color if not provided
    if color_tint is None:
        mean_colors = []
        for img_path in image_paths:
            img = np.array(Image.open(img_path).convert("RGB"))
            mean_colors.append(img.mean(axis=(0, 1)))
        avg_color = np.mean(mean_colors, axis=0)
        color_tint = [int(round(c)) for c in avg_color]

    profile = {
        "name": name,
        "description": description or f"Calibrated profile for {name}",
        "display_color": display_color,
        "calibrated": True,
        "calibration_date": datetime.now().isoformat(),
        "n_images": len(image_paths),
        "n_patches": n_patches,
        "patch_size": patch_size,
        "stride": stride,
        "percentile_margin": percentile_margin,
        "source_images": [str(p.name) for p in image_paths],
        "synthetic": {
            "color_tint": color_tint,
            "H_range": [
                round(features_stats.get('H_hurst', {}).get('lo', 0.1), 2),
                round(features_stats.get('H_hurst', {}).get('hi', 0.9), 2)
            ]
        },
        "features": features_stats
    }

    # Write JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"[calibrate] Profile saved → {out_path}")
    print(f"[calibrate] Feature ranges:")
    for key, stats in features_stats.items():
        print(f"  {key:20s}: [{stats['lo']:8.3f}, {stats['hi']:8.3f}]  (mean={stats['mean']:.3f}, std={stats['std']:.3f})")

    return profile


def main():
    parser = argparse.ArgumentParser(description="RATDS Material Profile Calibration")
    parser.add_argument("--name", "-n", required=True, help="Material name (e.g., silver_ore)")
    parser.add_argument("--input", "-i", required=True, help="Image file or directory of images")
    parser.add_argument("--output", "-o", default="profiles", help="Output directory for JSON profile")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size for tiling")
    parser.add_argument("--stride", type=int, default=64, help="Stride for tiling")
    parser.add_argument("--margin", type=float, default=0.10, help="Percentile margin (0.1 = 10%)")
    parser.add_argument("--color", default=None, help="Display hex color (auto if omitted)")
    parser.add_argument("--description", "-d", default="", help="Material description")
    args = parser.parse_args()

    calibrate_material(
        name=args.name,
        input_path=args.input,
        output_dir=args.output,
        patch_size=args.patch_size,
        stride=args.stride,
        percentile_margin=args.margin,
        display_color=args.color,
        description=args.description
    )


if __name__ == '__main__':
    main()
