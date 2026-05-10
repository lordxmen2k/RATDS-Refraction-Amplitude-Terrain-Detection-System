"""
RATDS Material Classifier v4.0 — JSON Profile System
Maps fractal/refraction features → material labels.

ARCHITECTURAL CHANGE in v4.0:
- Material profiles are NO LONGER hardcoded.
- Profiles are loaded from JSON files in the profiles/ directory.
- Run calibrate.py to generate new profiles from real images.
- The system automatically discovers all .json profiles at runtime.
"""

import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import Optional

# Profile loading -------------------------------------------------

_PROFILES: dict = {}
_PROFILES_DIR: Path = Path(__file__).parent / "profiles"


def load_profiles(profiles_dir: str | Path = None) -> dict:
    """
    Load all material profiles from JSON files in the profiles directory.
    Returns dict of {material_name: profile_data}.
    """
    global _PROFILES
    if profiles_dir is not None:
        pdir = Path(profiles_dir)
    else:
        pdir = _PROFILES_DIR

    _PROFILES = {}
    if not pdir.exists():
        print(f"[profiles] WARNING: Profile directory not found: {pdir}")
        return _PROFILES

    for json_file in sorted(pdir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            name = data.get("name", json_file.stem)
            _PROFILES[name] = data
        except Exception as e:
            print(f"[profiles] WARNING: Failed to load {json_file}: {e}")

    calibrated = sum(1 for p in _PROFILES.values() if p.get("calibrated", False))
    estimated = len(_PROFILES) - calibrated
    print(f"[profiles] Loaded {len(_PROFILES)} profiles from {pdir}")
    print(f"[profiles]   Calibrated: {calibrated}  |  Estimated: {estimated}")
    return _PROFILES


def get_profiles() -> dict:
    """Get loaded profiles, loading if necessary."""
    if not _PROFILES:
        load_profiles()
    return _PROFILES


def get_profile(material_name: str) -> dict | None:
    """Get a single profile by name."""
    return get_profiles().get(material_name)


def list_materials() -> list[str]:
    """List all available material names."""
    return sorted(get_profiles().keys())


def get_material_color(material_name: str) -> str:
    """Get display color for a material."""
    prof = get_profile(material_name)
    if prof:
        return prof.get("display_color", "#888888")
    return "#888888"


def extract_feature_ranges(profile_data: dict) -> dict:
    """
    Extract (lo, hi) tuples from a JSON profile for scoring.
    Only includes features that have both 'lo' and 'hi' defined.
    """
    ranges = {}
    for feat_name, stats in profile_data.get("features", {}).items():
        if isinstance(stats, dict) and "lo" in stats and "hi" in stats:
            ranges[feat_name] = (float(stats["lo"]), float(stats["hi"]))
    return ranges


# ─────────────────────────────────────────────
# Rule-Based Classifier (zero-shot)
# ─────────────────────────────────────────────

def score_against_profile(features: dict, profile_data: dict) -> float:
    """
    Score how well a feature vector matches a material profile.
    Uses Gaussian-like score centered on profile range.
    """
    profile_ranges = extract_feature_ranges(profile_data)
    if not profile_ranges:
        return 0.0

    scores = []
    for key, (lo, hi) in profile_ranges.items():
        val = features.get(key)
        if val is None:
            continue
        mid = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0
        if half_width == 0:
            scores.append(1.0 if val == lo else 0.0)
        else:
            dist = abs(val - mid) / half_width
            score = max(0.0, 1.0 - dist)
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def classify_material(features: dict, top_k: int = 3) -> list[tuple[str, float]]:
    """
    Rule-based classification using RATDS features against loaded JSON profiles.
    Returns sorted list of (material, confidence) tuples.
    """
    profiles = get_profiles()
    if not profiles:
        raise RuntimeError("No profiles loaded. Run load_profiles() or check profiles/ directory.")

    scores = {}
    for material, profile_data in profiles.items():
        scores[material] = score_against_profile(features, profile_data)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in ranked[:top_k]) or 1.0
    normalized = [(m, round(s / total, 4)) for m, s in ranked[:top_k]]
    return normalized


# ─────────────────────────────────────────────
# Trainable Classifier (k-NN)
# ─────────────────────────────────────────────

FEATURE_KEYS = [
    'beta', 'D_spectral', 'H_hurst',
    'D_vfd', 'D_divider', 'D_box', 'D_corr', 'D_mean',
    'R_response', 'beta_R', 'beta_G', 'beta_B',
    'spectral_flatness', 'beta_ratio_R', 'beta_ratio_G', 'beta_ratio_B', 'beta_spread',
    'contrast', 'mean_luminance',
]


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=float)


class RATDSClassifier:
    """Trainable k-NN classifier on RATDS features."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[list] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.labels: list = []

    def fit(self, samples: list[tuple[np.ndarray, str]], patch_size: int = 64):
        """samples: list of (image_patch_rgb, material_label)"""
        from ratds_core import compute_features
        X, y = [], []
        for patch, label in samples:
            try:
                feats = compute_features(patch)
                vec = features_to_vector(feats)
                X.append(vec)
                y.append(label)
            except Exception as e:
                print(f"  [skip] patch failed: {e}")
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        self.X_train = (X - self.mean_) / self.std_
        self.y_train = y
        self.labels = sorted(set(y))
        print(f"[RATDSClassifier] Trained on {len(X)} samples, "
              f"{len(self.labels)} classes: {self.labels}")

    def predict(self, patch: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        from ratds_core import compute_features
        feats = compute_features(patch)
        vec = (features_to_vector(feats) - self.mean_) / self.std_
        dists = np.linalg.norm(self.X_train - vec, axis=1)
        nn_idx = np.argsort(dists)[:self.k]
        nn_labels = [self.y_train[i] for i in nn_idx]
        nn_dists = dists[nn_idx]
        weights = 1.0 / (nn_dists + 1e-8)
        vote: dict[str, float] = {}
        for lbl, w in zip(nn_labels, weights):
            vote[lbl] = vote.get(lbl, 0.0) + w
        total = sum(vote.values())
        ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        return [(m, round(w / total, 4)) for m, w in ranked[:top_k]]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[RATDSClassifier] Saved to {path}")

    @staticmethod
    def load(path: str) -> 'RATDSClassifier':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# Spatial Detection Map
# ─────────────────────────────────────────────

def build_detection_map(image_rgb: np.ndarray,
                         patch_size: int = 32,
                         stride: int = 16,
                         classifier: Optional[RATDSClassifier] = None,
                         use_2d: bool = True,
                         remove_bg: bool = True) -> dict:
    from ratds_core import compute_features, detect_uniform_background, tile_image_masked
    H, W, _ = image_rgb.shape

    # Detect background mask
    if remove_bg:
        mask = detect_uniform_background(image_rgb, variance_threshold=15.0)
        print(f"[det_map] Foreground coverage: {mask.sum()/mask.size:.1%}")
    else:
        mask = np.ones((H, W), dtype=bool)

    # Build grid dimensions
    rows = list(range(0, H - patch_size, stride))
    cols = list(range(0, W - patch_size, stride))
    n_rows, n_cols = len(rows), len(cols)
    R_map = np.zeros((n_rows, n_cols))
    D_map = np.zeros_like(R_map)
    label_map = np.full(R_map.shape, '', dtype=object)
    patch_results = []

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            patch_mask = mask[r:r + patch_size, c:c + patch_size]
            if patch_mask.size == 0:
                continue

            # Skip if patch is mostly background
            if remove_bg and patch_mask.sum() / patch_mask.size < 0.15:
                continue

            patch = image_rgb[r:r + patch_size, c:c + patch_size].astype(float)
            try:
                feats = compute_features(patch, use_2d=use_2d)
                R_map[ri, ci] = feats['R_response']
                D_map[ri, ci] = feats['D_mean']
                if classifier is not None:
                    preds = classifier.predict(patch, top_k=1)
                    label_map[ri, ci] = preds[0][0] if preds else 'unknown'
                else:
                    preds = classify_material(feats, top_k=1)
                    label_map[ri, ci] = preds[0][0] if preds else 'unknown'
                patch_results.append((r, c, feats, preds))
            except Exception:
                pass

    return {
        'R_map': R_map,
        'D_map': D_map,
        'label_map': label_map,
        'patch_results': patch_results,
        'patch_size': patch_size,
        'stride': stride,
        'image_shape': (H, W),
        'mask': mask,
    }
