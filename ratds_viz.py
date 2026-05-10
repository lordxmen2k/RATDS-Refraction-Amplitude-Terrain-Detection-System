"""
RATDS Visualization & Analysis Tool v4.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from ratds_core import (
    extract_amplitude_signal, amplitude_spectrum,
    estimate_spectral_exponent, variance_fractal_dimension,
    divider_fractal_dimension, compute_features,
    normalize_illumination, to_luminance, radial_power_spectrum,
    estimate_beta_2d
)
from ratds_classifier import build_detection_map, get_profiles, get_material_color


def _get_all_colors():
    """Build color map from loaded JSON profiles."""
    colors = {'unknown': '#EEEEEE'}
    for name, prof in get_profiles().items():
        colors[name] = prof.get('display_color', '#888888')
    return colors


def plot_amplitude_analysis(image_rgb, row=None, save_path=None):
    H, W, _ = image_rgb.shape
    row = row or H // 2
    if H > 256 and W > 256:
        cy, cx = H // 2, W // 2
        patch = image_rgb[cy - 128:cy + 128, cx - 128:cx + 128]
    else:
        patch = image_rgb

    gray = to_luminance(patch)
    gray_norm = normalize_illumination(gray, sigma_ratio=0.25)
    freqs_2d, power_2d = radial_power_spectrum(gray_norm)
    beta_2d = estimate_beta_2d(gray_norm)
    D_2d = (6.0 - beta_2d) / 2.0
    H_2d = (beta_2d - 2.0) / 2.0

    sig = extract_amplitude_signal(patch)
    freqs_1d, amps_1d = amplitude_spectrum(sig)
    beta_1d = estimate_spectral_exponent(freqs_1d, amps_1d) if len(freqs_1d) > 2 else 2.0

    window = max(8, W // 10)
    vfd_vals, vfd_x = [], []
    for start in range(0, len(sig) - window, window // 2):
        seg = sig[start: start + window]
        vfd_vals.append(variance_fractal_dimension(seg))
        vfd_x.append(start + window // 2)

    fig = plt.figure(figsize=(16, 10), facecolor='#0D1117')
    fig.suptitle('RATDS v4.0 — Amplitude & Fractal Analysis', fontsize=14,
                 color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_img = fig.add_subplot(gs[0, :2])
    ax_norm = fig.add_subplot(gs[0, 2:])
    ax_sig = fig.add_subplot(gs[1, :2])
    ax_spec2d = fig.add_subplot(gs[1, 2])
    ax_spec1d = fig.add_subplot(gs[1, 3])
    ax_vfd = fig.add_subplot(gs[2, :2])
    ax_info = fig.add_subplot(gs[2, 2])
    ax_div = fig.add_subplot(gs[2, 3])

    def style(ax, title=''):
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363D')
        if title:
            ax.set_title(title, color='#C9D1D9', fontsize=9, pad=4)

    ax_img.imshow(image_rgb)
    ax_img.axhline(row, color='#58A6FF', linewidth=1.2, linestyle='--', alpha=0.8)
    ax_img.set_title(f'Input Image  ({W}×{H})', color='#C9D1D9', fontsize=9, pad=4)
    ax_img.axis('off')

    ax_norm.imshow(gray_norm, cmap='gray')
    ax_norm.set_title('Illumination Normalized', color='#C9D1D9', fontsize=9, pad=4)
    ax_norm.axis('off')

    style(ax_sig, '1D Amplitude Trace A(x) (legacy)')
    ax_sig.plot(sig, color='#58A6FF', linewidth=0.9)
    ax_sig.fill_between(range(len(sig)), sig, alpha=0.15, color='#58A6FF')
    ax_sig.set_ylabel('Amplitude', color='#8B949E', fontsize=8)

    style(ax_spec2d, f'2D Radial Spectrum  β₂d={beta_2d:.2f}')
    mask = (freqs_2d > 0.01) & (power_2d > 1e-10)
    if mask.sum() > 2:
        ax_spec2d.loglog(freqs_2d[mask], power_2d[mask], color='#F78166', linewidth=0.9)
        log_f = np.log(freqs_2d[mask])
        log_p = np.log(power_2d[mask])
        coeffs = np.polyfit(log_f, log_p, 1)
        fit_p = np.exp(np.poly1d(coeffs)(log_f))
        ax_spec2d.loglog(freqs_2d[mask], fit_p, color='#3FB950', linewidth=1.2, linestyle='--')
    ax_spec2d.set_xlabel('Frequency', color='#8B949E', fontsize=7)
    ax_spec2d.set_ylabel('Power', color='#8B949E', fontsize=7)

    style(ax_spec1d, f'1D Spectrum  β₁d={beta_1d:.2f}')
    ax_spec1d.loglog(freqs_1d, amps_1d, color='#BC8CFF', linewidth=0.9)
    if len(freqs_1d) > 2:
        fit_A = np.exp(np.poly1d(np.polyfit(np.log(freqs_1d), np.log(amps_1d + 1e-10), 1))(np.log(freqs_1d)))
        ax_spec1d.loglog(freqs_1d, fit_A, color='#3FB950', linewidth=1.2, linestyle='--')
    ax_spec1d.set_xlabel('Frequency', color='#8B949E', fontsize=7)
    ax_spec1d.set_ylabel('Amplitude', color='#8B949E', fontsize=7)

    style(ax_vfd, 'Variance Fractal Dimension D_VFD(x)')
    ax_vfd.plot(vfd_x, vfd_vals, color='#E3B341', linewidth=1.0, marker='o', markersize=3)
    ax_vfd.axhline(np.mean(vfd_vals), color='#F78166', linewidth=0.8, linestyle=':')
    ax_vfd.set_ylim(1.0, 2.5)
    ax_vfd.set_ylabel('D_VFD', color='#8B949E', fontsize=8)
    ax_vfd.set_xlabel('x position', color='#8B949E', fontsize=8)

    ax_info.set_facecolor('#161B22')
    ax_info.axis('off')
    feats = compute_features(patch, use_2d=True)
    lines = [
        ('β₂d (2D spectral)', f'{beta_2d:.3f}'),
        ('H₂d (2D Hurst)', f'{H_2d:.3f}'),
        ('D₂d spectral', f'{D_2d:.3f}'),
        ('β₁d (1D trace)', f'{beta_1d:.3f}'),
        ('D VFD', f'{feats["D_vfd"]:.3f}'),
        ('D divider', f'{feats["D_divider"]:.3f}'),
        ('D box', f'{feats["D_box"]:.3f}'),
        ('D corr', f'{feats["D_corr"]:.3f}'),
        ('D mean', f'{feats["D_mean"]:.3f}'),
        ('spectral_flat', f'{feats["spectral_flatness"]:.3f}'),
        ('β_R / β_G / β_B', f'{feats["beta_R"]:.1f}/{feats["beta_G"]:.1f}/{feats["beta_B"]:.1f}'),
        ('R(x)', f'{feats["R_response"]:.4f}'),
        ('contrast', f'{feats["contrast"]:.1f}'),
    ]
    ax_info.set_title('Feature Readout', color='#C9D1D9', fontsize=9, pad=4)
    for i, (k, v) in enumerate(lines):
        ax_info.text(0.04, 0.93 - i * 0.075, k, transform=ax_info.transAxes,
                     color='#8B949E', fontsize=8)
        ax_info.text(0.96, 0.93 - i * 0.075, v, transform=ax_info.transAxes,
                     color='#58A6FF', fontsize=8, ha='right', fontfamily='monospace')

    style(ax_div, 'Divider Method')
    step_sizes = np.unique(np.logspace(np.log10(1), np.log10(max(2, len(sig) // 4)), 12).astype(int))
    log_r, log_L = [], []
    x_arr = np.arange(len(sig), dtype=float)
    for r in step_sizes:
        r = int(r)
        if r < 1: continue
        indices = np.arange(0, len(x_arr) - r, r)
        if len(indices) < 2: continue
        dx = x_arr[indices + r] - x_arr[indices]
        dy = sig[indices + r] - sig[indices]
        L = np.sqrt(dx ** 2 + dy ** 2).sum()
        if L > 0:
            log_r.append(np.log(r))
            log_L.append(np.log(L))
    if len(log_r) > 2:
        ax_div.scatter(log_r, log_L, color='#BC8CFF', s=18, zorder=3)
        p = np.polyfit(log_r, log_L, 1)
        fit_line = np.poly1d(p)(log_r)
        ax_div.plot(log_r, fit_line, color='#3FB950', linewidth=1.2)
        D_div = 1.0 - p[0]
        ax_div.set_title(f'Divider  D={D_div:.3f}', color='#C9D1D9', fontsize=9, pad=4)
    ax_div.set_xlabel('log(r)', color='#8B949E', fontsize=7)
    ax_div.set_ylabel('log(L)', color='#8B949E', fontsize=7)
    ax_div.set_facecolor('#161B22')
    ax_div.tick_params(colors='#8B949E', labelsize=7)
    for sp in ax_div.spines.values():
        sp.set_edgecolor('#30363D')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        print(f"[viz] Saved → {save_path}")
    plt.show()
    return fig


def plot_detection_map(image_rgb, det, save_path=None):
    R_map = det['R_map']
    D_map = det['D_map']
    label_map = det['label_map']
    unique_labels = sorted(set(label_map.flatten().tolist()))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    int_map = np.vectorize(label_to_int.get)(label_map)

    colors = _get_all_colors()
    cmap_colors = [colors.get(l, '#EEEEEE') for l in unique_labels]
    cmap = ListedColormap(cmap_colors)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor='#0D1117')
    fig.suptitle('RATDS v4.0 — Spatial Detection Map', color='white',
                 fontsize=13, fontweight='bold')
    titles = ['Input Image', 'R(x) Response', 'D_mean Map', 'Material Labels']
    for ax, title in zip(axes, titles):
        ax.set_facecolor('#161B22')
        ax.set_title(title, color='#C9D1D9', fontsize=9, pad=5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363D')
    axes[0].imshow(image_rgb)
    R_norm = (R_map - R_map.min()) / (R_map.max() - R_map.min() + 1e-10)
    im1 = axes[1].imshow(R_norm, cmap='plasma', aspect='auto')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(
        colors='#8B949E', labelsize=7)
    im2 = axes[2].imshow(D_map, cmap='viridis', aspect='auto', vmin=1.0, vmax=2.2)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(
        colors='#8B949E', labelsize=7)
    axes[3].imshow(int_map, cmap=cmap, aspect='auto',
                   vmin=0, vmax=len(unique_labels) - 1)
    legend_patches = [
        mpatches.Patch(color=colors.get(l, '#EEE'), label=l)
        for l in unique_labels
    ]
    axes[3].legend(handles=legend_patches, loc='lower right',
                   fontsize=7, framealpha=0.6,
                   facecolor='#161B22', labelcolor='white')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        print(f"[viz] Saved → {save_path}")
    plt.show()
    return fig


def plot_feature_radar(features, predictions, save_path=None):
    feat_keys = ['D_vfd', 'D_divider', 'D_box', 'D_corr',
                 'H_hurst', 'beta', 'contrast', 'spectral_flatness']
    feat_labels = ['D_VFD', 'D_div', 'D_box', 'D_corr',
                   'H', 'β', 'contrast', 'flatness']
    ranges = {
        'D_vfd': (1.0, 2.5), 'D_divider': (1.0, 2.0), 'D_box': (1.0, 2.0),
        'D_corr': (0.1, 3.0), 'H_hurst': (-1.0, 1.0),
        'beta': (0.0, 6.0), 'contrast': (0.0, 100.0),
        'spectral_flatness': (0.0, 0.3),
    }
    vals = []
    for k in feat_keys:
        lo, hi = ranges[k]
        v = np.clip((features.get(k, lo) - lo) / (hi - lo + 1e-8), 0, 1)
        vals.append(v)
    N = len(feat_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_plot = vals + vals[:1]

    fig = plt.figure(figsize=(11, 5), facecolor='#0D1117')
    ax_radar = fig.add_subplot(121, polar=True)
    ax_bar = fig.add_subplot(122)
    ax_radar.set_facecolor('#161B22')
    ax_radar.plot(angles, vals_plot, color='#58A6FF', linewidth=1.5)
    ax_radar.fill(angles, vals_plot, alpha=0.2, color='#58A6FF')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feat_labels, color='#C9D1D9', fontsize=9)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['', '', '', ''], fontsize=7)
    ax_radar.tick_params(colors='#8B949E')
    ax_radar.spines['polar'].set_edgecolor('#30363D')
    ax_radar.set_title('Feature Profile', color='#C9D1D9', fontsize=10, pad=15)

    ax_bar.set_facecolor('#161B22')
    ax_bar.tick_params(colors='#8B949E', labelsize=9)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('#30363D')
    if predictions:
        mats, confs = zip(*predictions)
        colors = _get_all_colors()
        bar_colors = [colors.get(m, '#888') for m in mats]
        bars = ax_bar.barh(list(mats), list(confs), color=bar_colors,
                           height=0.5, edgecolor='#30363D', linewidth=0.5)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel('Confidence', color='#8B949E', fontsize=9)
        ax_bar.set_title('Material Predictions', color='#C9D1D9', fontsize=10, pad=8)
        for bar, conf in zip(bars, confs):
            ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{conf:.1%}', va='center', color='#C9D1D9', fontsize=8)
        ax_bar.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
        print(f"[viz] Saved → {save_path}")
    plt.show()
    return fig
