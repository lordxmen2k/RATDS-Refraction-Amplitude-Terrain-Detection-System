"""
RATDS Training & Validation Framework v4.0
Evaluate equation-based feature extraction + classifier accuracy.

FIXES in v4.0:
- Loads material profiles from JSON for synthetic generation
- Supports any material defined in profiles/ directory
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

from ratds_core import compute_features
from ratds_classifier import (
    RATDSClassifier, classify_material,
    features_to_vector, FEATURE_KEYS, get_profiles, get_profile
)


# ─────────────────────────────────────────────
# Synthetic patch generator (2D fBm)
# ─────────────────────────────────────────────

def generate_synthetic_patch(material: str,
                               size: int = 64,
                               seed: int = None) -> np.ndarray:
    """
    Generate a synthetic RGB patch with material-like texture properties.
    Uses true 2D fBm spectral synthesis.
    Loads color tint and H range from the material's JSON profile.
    """
    rng = np.random.default_rng(seed)

    # Load profile from JSON
    prof = get_profile(material)
    if prof is None:
        raise ValueError(f"No profile found for material '{material}'. "
                         f"Run calibrate.py or add a JSON profile to profiles/")

    # Get H range from profile features or synthetic section
    synth = prof.get("synthetic", {})
    H_range = synth.get("H_range", [0.1, 0.9])
    H_lo, H_hi = H_range
    H = rng.uniform(max(0.01, H_lo), min(0.99, H_hi))

    def fbm_2d(n, H):
        """True 2D fractional Brownian motion via spectral synthesis."""
        freq_y = np.fft.fftfreq(n)
        freq_x = np.fft.fftfreq(n)
        fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
        f_radial = np.sqrt(fx**2 + fy**2)
        f_radial[0, 0] = 1e-10
        power = f_radial ** (-(2 * H + 2) / 2)
        phase = rng.uniform(0, 2 * np.pi, (n, n))
        spectrum = power * np.exp(1j * phase)
        texture = np.real(np.fft.ifft2(spectrum))
        return texture

    texture = fbm_2d(size, H)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    patch = np.zeros((size, size, 3))
    for ch in range(3):
        H_ch = np.clip(H + rng.normal(0, 0.05), 0.01, 0.99)
        tex_ch = fbm_2d(size, H_ch)
        tex_ch = (tex_ch - tex_ch.min()) / (tex_ch.max() - tex_ch.min() + 1e-8)
        patch[:, :, ch] = tex_ch * 255

    # Apply material-specific color tint from profile
    tint = np.array(synth.get("color_tint", [200, 200, 200]), dtype=float)
    patch = 0.6 * patch + 0.4 * tint[None, None, :]
    return np.clip(patch, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def build_synthetic_dataset(materials: list = None,
                              n_per_class: int = 30,
                              patch_size: int = 64) -> list[tuple]:
    """Build a synthetic labeled dataset for training/testing."""
    if materials is None:
        materials = list(get_profiles().keys())

    dataset = []
    for mat in materials:
        for i in range(n_per_class):
            patch = generate_synthetic_patch(mat, size=patch_size,
                                              seed=i * 100 + hash(mat) % 1000)
            dataset.append((patch, mat))
    return dataset


def load_image_dataset(root_dir: str,
                        patch_size: int = 64) -> list[tuple]:
    """
    Load real images from a directory structure:
      root_dir/
        metal/  img1.jpg img2.png ...
        wood/   img1.jpg ...
    Each image is divided into non-overlapping patches.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    dataset = []
    root = Path(root_dir)
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in label_dir.glob('*'):
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                continue
            try:
                img = np.array(Image.open(img_path).convert('RGB'))
                H, W, _ = img.shape
                for r in range(0, H - patch_size, patch_size):
                    for c in range(0, W - patch_size, patch_size):
                        patch = img[r:r + patch_size, c:c + patch_size]
                        dataset.append((patch, label))
            except Exception as e:
                print(f"  [skip] {img_path}: {e}")

    print(f"[dataset] Loaded {len(dataset)} patches from {root_dir}")
    return dataset


# ─────────────────────────────────────────────
# Train / Evaluate
# ─────────────────────────────────────────────

def train_test_split(dataset: list, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset)).tolist()
    n_test = int(len(dataset) * test_ratio)
    test_idx = set(idx[:n_test])
    train = [dataset[i] for i in range(len(dataset)) if i not in test_idx]
    test  = [dataset[i] for i in range(len(dataset)) if i in test_idx]
    return train, test


def evaluate(classifier: Optional[RATDSClassifier],
             test_data: list,
             use_rule_based: bool = False) -> dict:
    """Evaluate classifier on test data."""
    labels_all = sorted(set(lbl for _, lbl in test_data))
    label_to_i = {l: i for i, l in enumerate(labels_all)}

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    timing = []

    print(f"\n[eval] Running on {len(test_data)} samples...")
    for patch, true_label in test_data:
        t0 = time.perf_counter()
        try:
            if use_rule_based:
                feats = compute_features(patch)
                preds = classify_material(feats, top_k=1)
            else:
                preds = classifier.predict(patch, top_k=1)
            elapsed = time.perf_counter() - t0
            timing.append(elapsed)

            pred_label = preds[0][0] if preds else 'unknown'
            confusion[true_label][pred_label] += 1
            per_class_total[true_label] += 1
            total += 1
            if pred_label == true_label:
                correct += 1
                per_class_correct[true_label] += 1
        except Exception as e:
            per_class_total[true_label] += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {
        lbl: per_class_correct[lbl] / per_class_total[lbl]
        for lbl in labels_all
        if per_class_total[lbl] > 0
    }

    results = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct': correct,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
        'mean_inference_ms': np.mean(timing) * 1000 if timing else 0.0,
    }

    print(f"\n{'─'*50}")
    print(f"  Overall Accuracy : {accuracy:.1%}  ({correct}/{total})")
    print(f"  Mean Inference   : {results['mean_inference_ms']:.1f} ms/patch")
    print(f"{'─'*50}")
    print(f"  Per-class accuracy:")
    for lbl, acc in sorted(per_class_acc.items(), key=lambda x: -x[1]):
        bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
        print(f"    {lbl:12s}  {bar}  {acc:.1%}")
    print(f"{'─'*50}\n")

    return results


def save_results(results: dict, path: str):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Results saved → {path}")


# ─────────────────────────────────────────────
# Feature importance analysis
# ─────────────────────────────────────────────

def feature_importance_analysis(dataset: list, n_samples: int = 200) -> dict:
    """Compute mean feature values per class to show discriminative power."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    samples = [dataset[i] for i in idx]

    class_features = defaultdict(list)
    for patch, label in samples:
        try:
            feats = compute_features(patch)
            vec = features_to_vector(feats)
            class_features[label].append(vec)
        except Exception:
            pass

    class_means = {
        lbl: np.mean(vecs, axis=0).tolist()
        for lbl, vecs in class_features.items()
    }

    all_means = np.array(list(class_means.values()))
    if all_means.ndim < 2 or all_means.shape[0] < 2:
        inter_var = np.zeros(len(FEATURE_KEYS))
    else:
        inter_var = all_means.var(axis=0)
    if np.isscalar(inter_var):
        inter_var = np.zeros(len(FEATURE_KEYS))
    importance = {
        k: float(inter_var[i])
        for i, k in enumerate(FEATURE_KEYS)
    }

    ranked = sorted(importance.items(), key=lambda x: -x[1])
    print("\n[features] Discriminative power (inter-class variance):")
    for feat, var in ranked:
        bar = '█' * min(30, int(var * 300))
        print(f"  {feat:20s}  {bar}  {var:.5f}")

    return {
        'class_means': class_means,
        'feature_importance': importance,
        'feature_ranking': [f for f, _ in ranked],
    }
