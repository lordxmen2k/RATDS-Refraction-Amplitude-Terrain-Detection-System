"""
RATDS v4.0 — Main entrypoint
Runs the full pipeline: generate → train → evaluate → visualize

ARCHITECTURAL CHANGE in v4.0:
- Material profiles are loaded from JSON files in profiles/
- Run calibrate.py to add new materials without touching code
- All detection uses dynamically loaded profiles
"""

import numpy as np
import sys
from pathlib import Path

from ratds_classifier import load_profiles, list_materials, classify_material, build_detection_map, RATDSClassifier
from ratds_core import compute_features, aggregate_patch_features
from ratds_viz import plot_amplitude_analysis, plot_detection_map, plot_feature_radar


def demo_single_image(image_path: str = None):
    """Run RATDS on a single image file."""
    try:
        from PIL import Image
        if image_path:
            img = np.array(Image.open(image_path).convert('RGB'))
        else:
            from ratds_train import generate_synthetic_patch
            img = generate_synthetic_patch('metal', size=128, seed=42)
            print("[demo] No image path given — using synthetic 'metal' patch.")
    except Exception as e:
        print(f"[error] Could not load image: {e}")
        return

    print(f"[demo] Image shape: {img.shape}")
    h, w = img.shape[:2]

    if max(h, w) <= 128:
        print("[demo] Small image — analyzing directly")
        feats = compute_features(img, use_2d=True)
    else:
        patch_size = min(128, min(h, w) // 2)
        stride = patch_size // 2
        print(f"[demo] Large image — tiling with {patch_size}x{patch_size} patches, stride={stride}")
        feats = aggregate_patch_features(img, patch_size=patch_size, stride=stride, use_2d=True)
        print(f"[demo] Analyzed {feats.get('n_patches', 1)} patches")

    print("\n── Feature Extraction ──────────────────────────")
    for k, v in feats.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")

    print("\n── Rule-Based Classification ───────────────────")
    preds = classify_material(feats, top_k=5)
    for material, conf in preds:
        bar = '█' * int(conf * 40) + '░' * (40 - int(conf * 40))
        print(f"  {material:14s}  {bar}  {conf:.1%}")

    print("\n── Building Detection Map ──────────────────────")
    det = build_detection_map(img, patch_size=min(64, min(h,w)//4),
                               stride=min(32, min(h,w)//8), use_2d=True)
    print(f"  R_map shape  : {det['R_map'].shape}")
    print(f"  D_map mean   : {det['D_map'].mean():.3f}")
    labels_found = set(det['label_map'].flatten())
    print(f"  Labels found : {labels_found}")

    print("\n── Visualization ───────────────────────────────")
    plot_amplitude_analysis(img, save_path='ratds_amplitude.png')
    plot_detection_map(img, det, save_path='ratds_detection_map.png')
    plot_feature_radar(feats, preds[:4], save_path='ratds_radar.png')

    print("\n[done] Outputs saved: ratds_amplitude.png, ratds_detection_map.png, ratds_radar.png")


def demo_training(n_per_class: int = 25):
    """Full training + evaluation on synthetic data."""
    from ratds_train import build_synthetic_dataset, train_test_split, evaluate, feature_importance_analysis, save_results

    materials = list_materials()
    print(f"\n[train] Building synthetic dataset — {n_per_class} patches/class × {len(materials)} classes")
    print(f"[train] Materials: {materials}")
    dataset = build_synthetic_dataset(materials=materials, n_per_class=n_per_class)
    print(f"[train] Total patches: {len(dataset)}")

    train_data, test_data = train_test_split(dataset, test_ratio=0.25)

    print("\n── Feature Importance Analysis ─────────────────")
    importance = feature_importance_analysis(train_data)

    print("\n── Training k-NN Classifier ─────────────────────")
    clf = RATDSClassifier(k=7)
    clf.fit(train_data)

    print("\n── Evaluation: Trained k-NN ─────────────────────")
    results_knn = evaluate(clf, test_data)

    print("\n── Evaluation: Rule-Based (zero-shot) ───────────")
    results_rule = evaluate(None, test_data, use_rule_based=True)

    clf.save('ratds_classifier.pkl')
    save_results({'knn': results_knn, 'rule_based': results_rule,
                  'feature_importance': importance['feature_ranking']},
                 'ratds_results.json')

    return clf, results_knn, results_rule


if __name__ == '__main__':
    # Load profiles at startup
    load_profiles()

    if len(sys.argv) > 1 and sys.argv[1] != '--train':
        demo_single_image(sys.argv[1])
    elif '--train' in sys.argv:
        demo_training(n_per_class=30)
    else:
        demo_single_image()
