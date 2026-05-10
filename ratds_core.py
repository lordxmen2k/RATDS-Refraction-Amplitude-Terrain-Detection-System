"""
Refraction Amplitude Terrain Detection System (RATDS) v3.0
Core mathematical engine — RGB image material detection

FIXES in v3.0:
- 2D radial power spectrum
- Illumination normalization (high-pass filtering)
- Local patch tiling for multi-scale analysis
- Correct β_2D = 2H + 2 relationship for 2D surfaces
- SPECTRAL_FLATNESS feature: std dev of per-channel betas
  * Metals (silver, steel): flatness ≈ 0.0-0.05 (neutral reflectance)
  * Gold: flatness ≈ 0.10-0.20 (strong blue absorption)
  * Copper: flatness ≈ 0.05-0.15 (moderate blue absorption)
- Per-patch feature aggregation with uncertainty
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft, fft2, fftfreq
from scipy.spatial.distance import pdist


# ─────────────────────────────────────────────
# 0. Preprocessing: Illumination Normalization
# ─────────────────────────────────────────────

def normalize_illumination(patch_gray: np.ndarray, sigma_ratio: float = 0.25) -> np.ndarray:
    """Remove low-frequency shading via high-pass filter."""
    h, w = patch_gray.shape
    sigma = max(h, w) * sigma_ratio
    low_freq = ndimage.gaussian_filter(patch_gray.astype(float), sigma=sigma)
    high_pass = patch_gray.astype(float) - low_freq
    high_pass = high_pass - high_pass.min()
    if high_pass.max() > 0:
        high_pass = high_pass / high_pass.max() * 255.0
    return high_pass




# ─────────────────────────────────────────────
# 0b. Background Removal (Near-Uniform Region Masking)
# ─────────────────────────────────────────────

def detect_uniform_background(image_rgb: np.ndarray,
                               variance_threshold: float = 15.0,
                               min_region_size: float = 0.05) -> np.ndarray:
    """
    Detect and mask near-uniform background regions in product photos.

    Algorithm:
      1. Compute local variance in small windows
      2. Threshold to separate uniform (background) from textured (foreground)
      3. Remove small noise regions via morphological opening
      4. Return binary mask: True = foreground, False = background

    Args:
        image_rgb: Input RGB image
        variance_threshold: Std dev below this is considered uniform/background
        min_region_size: Minimum fraction of image for a region to be kept

    Returns:
        Binary mask (H, W), True = keep, False = mask out
    """
    h, w, _ = image_rgb.shape
    gray = to_luminance(image_rgb)

    # Compute local standard deviation in 5x5 windows
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray.astype(float), size=5)
    local_mean_sq = uniform_filter(gray.astype(float) ** 2, size=5)
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    # Low variance = uniform = background
    mask = local_std > variance_threshold

    # Remove small isolated regions (noise) via binary opening
    from scipy.ndimage import binary_opening, binary_closing
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=2)

    # If the mask is too small, the whole image might be textured
    # or the threshold might be too aggressive. Fall back to keeping everything.
    if mask.sum() < (h * w * min_region_size):
        # Try a more lenient threshold
        mask = local_std > (variance_threshold * 0.5)
        mask = binary_opening(mask, iterations=1)
        mask = binary_closing(mask, iterations=2)

    # If still too small, keep everything (no background detected)
    if mask.sum() < (h * w * 0.01):
        return np.ones((h, w), dtype=bool)

    return mask


def remove_background(image_rgb: np.ndarray,
                       variance_threshold: float = 15.0,
                       fill_value: int = 0) -> tuple:
    """
    Remove near-uniform background from product photos.

    Returns:
        (image_no_bg, mask) where background pixels are set to fill_value
    """
    mask = detect_uniform_background(image_rgb, variance_threshold)
    result = image_rgb.copy()
    result[~mask] = fill_value
    return result, mask


def filter_patches_by_mask(rows: range, cols: range, patch_size: int,
                            mask: np.ndarray,
                            min_foreground_ratio: float = 0.30) -> list:
    """
    Filter patch positions to only include those with sufficient foreground.

    Args:
        rows, cols: Patch position iterators
        patch_size: Size of each patch
        mask: Binary foreground mask
        min_foreground_ratio: Minimum fraction of patch that must be foreground

    Returns:
        List of (r, c) tuples for valid patches
    """
    valid = []
    for r in rows:
        for c in cols:
            patch_mask = mask[r:r + patch_size, c:c + patch_size]
            if patch_mask.size == 0:
                continue
            ratio = patch_mask.sum() / patch_mask.size
            if ratio >= min_foreground_ratio:
                valid.append((r, c))
    return valid

def to_luminance(patch: np.ndarray) -> np.ndarray:
    """RGB → perceptual luminance."""
    weights = np.array([0.2126, 0.7152, 0.0722])
    return patch @ weights


# ─────────────────────────────────────────────
# 1. 2D Radial Power Spectrum & β estimation
# ─────────────────────────────────────────────

def radial_power_spectrum(patch_gray: np.ndarray) -> tuple:
    """Compute 2D FFT and radially average the power spectrum."""
    h, w = patch_gray.shape
    centered = patch_gray - patch_gray.mean()
    win_h = np.hanning(h)[:, None]
    win_w = np.hanning(w)[None, :]
    windowed = centered * win_h * win_w
    F = fft2(windowed)
    P = np.abs(F) ** 2
    P_shifted = np.fft.fftshift(P)
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.hypot(x - cx, y - cy).astype(int)
    r_max = min(h, w) // 2
    radial_mean = np.bincount(r.ravel(), P_shifted.ravel())[:r_max]
    radial_count = np.bincount(r.ravel())[:r_max]
    radial_power = radial_mean / (radial_count + 1e-10)
    freqs = np.arange(r_max) / r_max * 0.5
    return freqs, radial_power


def estimate_beta_2d(patch_gray: np.ndarray, f_min: float = 0.01, f_max: float = 0.45) -> float:
    """Fit β from 2D radial power spectrum. P(f) ∝ f^(-β_2d), β_2d = 2H + 2."""
    freqs, power = radial_power_spectrum(patch_gray)
    mask = (freqs > f_min) & (freqs < f_max) & (power > 1e-10)
    if mask.sum() < 3:
        return 2.0
    f_sel = freqs[mask]
    p_sel = power[mask]
    log_f = np.log(f_sel)
    log_p = np.log(p_sel)
    coeffs = np.polyfit(log_f, log_p, 1)
    beta = -coeffs[0]
    return float(np.clip(beta, 0.0, 6.0))


def beta2d_to_fractal_dimension(beta: float) -> float:
    """β_2D = 2H + 2 = 6 - 2D  →  D = (6 - β) / 2"""
    return (6.0 - beta) / 2.0


def beta2d_to_hurst(beta: float) -> float:
    """β_2D = 2H + 2  →  H = (β - 2) / 2"""
    return (beta - 2.0) / 2.0


# ─────────────────────────────────────────────
# 2. Per-channel 2D spectral analysis
# ─────────────────────────────────────────────

def estimate_channel_betas_2d(patch: np.ndarray) -> dict:
    """Estimate β_2D for each RGB channel independently."""
    betas = {}
    for i, ch in enumerate(['R', 'G', 'B']):
        gray = patch[:, :, i].astype(float)
        norm = normalize_illumination(gray, sigma_ratio=0.25)
        betas[ch] = estimate_beta_2d(norm)
    return betas


def spectral_flatness(channel_betas: dict) -> float:
    """
    Measure how similar the spectral exponents are across channels.
    Low flatness  (~0.0-0.05): neutral metal (silver, steel, aluminum)
    High flatness (~0.10-0.20): gold (strong blue absorption)
    Mid flatness  (~0.05-0.15): copper (moderate blue absorption)
    """
    vals = np.array([channel_betas['R'], channel_betas['G'], channel_betas['B']])
    return float(np.std(vals))


def spectral_signature(channel_betas: dict) -> dict:
    """
    Return relative spectral signature: which channel has highest/lowest beta.
    Gold:  beta_R > beta_G >> beta_B  (blue strongly absorbed)
    Silver: beta_R ≈ beta_G ≈ beta_B  (neutral)
    Copper: beta_R >> beta_G > beta_B  (red enhanced)
    """
    r, g, b = channel_betas['R'], channel_betas['G'], channel_betas['B']
    mean_beta = np.mean([r, g, b])
    return {
        'beta_ratio_R': r / (mean_beta + 1e-8),
        'beta_ratio_G': g / (mean_beta + 1e-8),
        'beta_ratio_B': b / (mean_beta + 1e-8),
        'beta_spread': max(r, g, b) - min(r, g, b),
    }


# ─────────────────────────────────────────────
# 3. Legacy 1D amplitude (kept for compatibility)
# ─────────────────────────────────────────────

def extract_amplitude_signal(patch: np.ndarray) -> np.ndarray:
    gray = to_luminance(patch)
    return gray.mean(axis=0)


def extract_channel_traces(patch: np.ndarray) -> dict:
    return {
        'R': patch[:, :, 0].mean(axis=0).astype(float),
        'G': patch[:, :, 1].mean(axis=0).astype(float),
        'B': patch[:, :, 2].mean(axis=0).astype(float),
    }


def amplitude_spectrum(signal_1d: np.ndarray, eps: float = 1e-10):
    N = len(signal_1d)
    A = np.abs(fft(signal_1d - signal_1d.mean()))[:N // 2]
    f = fftfreq(N)[:N // 2]
    mask = f > eps
    return f[mask], A[mask]


def estimate_spectral_exponent(freqs: np.ndarray, amplitudes: np.ndarray) -> float:
    log_f = np.log(freqs + 1e-10)
    log_A = np.log(amplitudes + 1e-10)
    coeffs = np.polyfit(log_f, log_A, 1)
    beta = -coeffs[0]
    return float(beta)


def beta_to_fractal_dimension(beta: float) -> float:
    return (5.0 - beta) / 2.0


def beta_to_hurst(beta: float) -> float:
    return (beta - 1.0) / 2.0


# ─────────────────────────────────────────────
# 4. Variance Fractal Dimension (VFD) — 1D
# ─────────────────────────────────────────────

def variance_fractal_dimension(signal_1d: np.ndarray, scales: list = None) -> float:
    if scales is None:
        max_scale = max(4, len(signal_1d) // 4)
        scales = [2 ** i for i in range(1, int(np.log2(max_scale)) + 1)]
    log_s, log_var = [], []
    for s in scales:
        if s >= len(signal_1d):
            continue
        n_windows = len(signal_1d) // s
        windows = signal_1d[:n_windows * s].reshape(n_windows, s)
        var = windows.var(axis=1).mean()
        if var > 0:
            log_s.append(np.log(s))
            log_var.append(np.log(var))
    if len(log_s) < 2:
        return 2.0
    slope, _ = np.polyfit(log_s, log_var, 1)
    H = slope / 2.0
    H = np.clip(H, 0.01, 0.99)
    return float(3.0 - H)


# ─────────────────────────────────────────────
# 5. Divider Method — 1D
# ─────────────────────────────────────────────

def divider_fractal_dimension(signal_1d: np.ndarray, n_steps: int = 12) -> float:
    x = np.arange(len(signal_1d), dtype=float)
    y = signal_1d.astype(float)
    step_sizes = np.logspace(
        np.log10(1), np.log10(max(2, len(signal_1d) // 4)), n_steps
    ).astype(int)
    step_sizes = np.unique(step_sizes)
    log_r, log_L = [], []
    for r in step_sizes:
        r = int(r)
        if r < 1:
            continue
        indices = np.arange(0, len(x) - r, r)
        if len(indices) < 2:
            continue
        dx = x[indices + r] - x[indices]
        dy = y[indices + r] - y[indices]
        seg_lengths = np.sqrt(dx ** 2 + dy ** 2)
        L = seg_lengths.sum()
        if L > 0:
            log_r.append(np.log(r))
            log_L.append(np.log(L))
    if len(log_r) < 2:
        return 1.5
    slope, _ = np.polyfit(log_r, log_L, 1)
    D = 1.0 - slope
    return float(np.clip(D, 1.0, 2.0))


# ─────────────────────────────────────────────
# 6. Box-Counting Dimension (2D)
# ─────────────────────────────────────────────

def box_counting_dimension(patch_gray: np.ndarray, min_box: int = 2, max_box: int = None) -> float:
    img = patch_gray.astype(float)
    threshold = img.mean()
    binary = (img > threshold).astype(np.uint8)
    if max_box is None:
        max_box = min(binary.shape) // 2
    box_sizes = np.unique(
        np.logspace(np.log10(min_box), np.log10(max(min_box + 1, max_box)), 10).astype(int)
    )
    log_d, log_N = [], []
    for d in box_sizes:
        d = int(d)
        if d < 1:
            continue
        H, W = binary.shape
        count = 0
        for i in range(0, H, d):
            for j in range(0, W, d):
                block = binary[i:i + d, j:j + d]
                if block.sum() > 0:
                    count += 1
        if count > 0:
            log_d.append(np.log(d))
            log_N.append(np.log(count))
    if len(log_d) < 2:
        return 1.5
    slope, _ = np.polyfit(log_d, log_N, 1)
    return float(np.clip(-slope, 1.0, 2.0))


# ─────────────────────────────────────────────
# 7. Correlation Dimension Dc — 1D
# ─────────────────────────────────────────────

def correlation_dimension(signal_1d: np.ndarray, embedding_dim: int = 2, n_radii: int = 20) -> float:
    tau = max(1, len(signal_1d) // 20)
    N = len(signal_1d) - (embedding_dim - 1) * tau
    if N < 10:
        return 1.5
    vectors = np.array([
        signal_1d[i: i + embedding_dim * tau: tau]
        for i in range(N)
    ])
    if N > 300:
        idx = np.random.choice(N, 300, replace=False)
        vectors = vectors[idx]
        N = 300
    dists = pdist(vectors)
    r_min = np.percentile(dists, 5)
    r_max = np.percentile(dists, 50)
    if r_min <= 0 or r_max <= r_min:
        return 1.5
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    total_pairs = len(dists)
    log_r, log_C = [], []
    for r in radii:
        count = (dists < r).sum()
        if count > 0:
            C = count / total_pairs
            log_r.append(np.log(r))
            log_C.append(np.log(C))
    if len(log_r) < 2:
        return 1.5
    slope, _ = np.polyfit(log_r, log_C, 1)
    return float(np.clip(slope, 0.1, 3.0))


# ─────────────────────────────────────────────
# 8. 2D VFD
# ─────────────────────────────────────────────

def variance_fractal_dimension_2d(patch_gray: np.ndarray, scales: list = None) -> float:
    h, w = patch_gray.shape
    if scales is None:
        max_s = min(h, w) // 4
        scales = [2 ** i for i in range(1, int(np.log2(max_s)) + 1)]
    log_s, log_var = [], []
    for s in scales:
        if s > h or s > w:
            continue
        n_h = h // s
        n_w = w // s
        windows = patch_gray[:n_h * s, :n_w * s].reshape(n_h, s, n_w, s)
        windows = windows.transpose(0, 2, 1, 3).reshape(-1, s, s)
        var = windows.var(axis=(1, 2)).mean()
        if var > 0:
            log_s.append(np.log(s))
            log_var.append(np.log(var))
    if len(log_s) < 2:
        return 2.0
    slope, _ = np.polyfit(log_s, log_var, 1)
    H = slope / 2.0
    H = np.clip(H, 0.01, 0.99)
    return float(3.0 - H)


# ─────────────────────────────────────────────
# 9. Unified Detection Response R(x)
# ─────────────────────────────────────────────

def detection_response(signal_1d: np.ndarray, D_terrain: float,
                        f_min: float = 0.01, f_max: float = 0.5, A0: float = 1.0) -> float:
    freqs, amps = amplitude_spectrum(signal_1d)
    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() < 2:
        return 0.0
    f_sel = freqs[mask]
    A_sel = amps[mask]
    exponent = 5.0 - 2.0 * D_terrain
    weights = A0 / (f_sel ** exponent + 1e-10)
    if hasattr(np, 'trapezoid'):
        integral = np.trapezoid(A_sel * weights, f_sel)
    else:
        integral = np.trapz(A_sel * weights, f_sel)
    window = max(4, len(signal_1d) // 8)
    D_vals = []
    for start in range(0, len(signal_1d) - window, window // 2):
        seg = signal_1d[start: start + window]
        D_vals.append(variance_fractal_dimension(seg))
    D_arr = np.array(D_vals)
    grad_mean = np.abs(np.gradient(D_arr)).mean() if len(D_arr) > 1 else 0.0
    return float(integral * grad_mean)


# ─────────────────────────────────────────────
# 10. Multi-patch tiling
# ─────────────────────────────────────────────

def tile_image(image_rgb: np.ndarray, patch_size: int = 128, stride: int = 64):
    """Standard grid tiling over the entire image."""
    H, W, _ = image_rgb.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            yield r, c, image_rgb[r:r + patch_size, c:c + patch_size]


def tile_image_masked(image_rgb: np.ndarray, mask: np.ndarray,
                       patch_size: int = 128, stride: int = 64,
                       min_foreground_ratio: float = 0.30):
    """
    Targeted tiling: only yield patches from masked (foreground) regions.

    Args:
        image_rgb: Input RGB image
        mask: Binary foreground mask (True = foreground)
        patch_size: Size of each patch
        stride: Stride for tiling
        min_foreground_ratio: Minimum fraction of patch that must be foreground

    Yields:
        (row, col, patch) tuples only for patches meeting the foreground threshold
    """
    H, W, _ = image_rgb.shape
    n_yielded = 0
    n_skipped = 0

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch_mask = mask[r:r + patch_size, c:c + patch_size]
            if patch_mask.size == 0:
                continue
            ratio = patch_mask.sum() / patch_mask.size
            if ratio >= min_foreground_ratio:
                yield r, c, image_rgb[r:r + patch_size, c:c + patch_size]
                n_yielded += 1
            else:
                n_skipped += 1

    if n_yielded == 0 and n_skipped > 0:
        print(f"[tile_masked] WARNING: All {n_skipped} patches skipped. Lowering threshold.")
        # Fallback: use lower threshold
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch_mask = mask[r:r + patch_size, c:c + patch_size]
                if patch_mask.size == 0:
                    continue
                ratio = patch_mask.sum() / patch_mask.size
                if ratio >= 0.05:  # Very lenient fallback
                    yield r, c, image_rgb[r:r + patch_size, c:c + patch_size]


def find_foreground_regions(mask: np.ndarray, min_area: int = 100):
    """
    Find connected foreground regions in the mask.
    Returns list of bounding boxes (r0, c0, r1, c1) for each region.
    """
    from scipy.ndimage import label
    labeled, n_features = label(mask)
    regions = []
    for i in range(1, n_features + 1):
        region_mask = labeled == i
        if region_mask.sum() < min_area:
            continue
        coords = np.argwhere(region_mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        regions.append((r0, c0, r1, c1))
    return regions


def aggregate_patch_features(image_rgb: np.ndarray, patch_size: int = 128,
                              stride: int = 64, use_2d: bool = True,
                              remove_bg: bool = True,
                              bg_variance_threshold: float = 15.0) -> dict:
    """
    Compute features on targeted patches from foreground regions and aggregate (median).

    Args:
        remove_bg: If True, auto-detect and mask uniform background regions
        bg_variance_threshold: Std dev threshold for background detection
    """
    # Step 1: Detect and remove uniform background
    if remove_bg:
        img_clean, mask = remove_background(image_rgb, variance_threshold=bg_variance_threshold)

        # Step 2: Tile ONLY on masked foreground regions
        all_features = []
        for r, c, patch in tile_image_masked(img_clean, mask, patch_size, stride,
                                              min_foreground_ratio=0.30):
            try:
                feats = compute_features(patch, use_2d=use_2d)
                # Skip patches with near-zero contrast (background residue)
                if feats.get('contrast', 0) < 1.0:
                    continue
                all_features.append(feats)
            except Exception:
                pass

        if all_features:
            print(f"[agg] Targeted tiling: {len(all_features)} foreground patches analyzed")
        else:
            print("[agg] Warning: No valid foreground patches. Falling back to full image.")
            all_features = []
            for r, c, patch in tile_image(img_clean, patch_size, stride):
                try:
                    feats = compute_features(patch, use_2d=use_2d)
                    if feats.get('contrast', 0) >= 1.0:
                        all_features.append(feats)
                except Exception:
                    pass
    else:
        # Standard tiling without background removal
        all_features = []
        for r, c, patch in tile_image(image_rgb, patch_size, stride):
            try:
                feats = compute_features(patch, use_2d=use_2d)
                if feats.get('contrast', 0) >= 1.0:
                    all_features.append(feats)
            except Exception:
                pass

    if not all_features:
        return compute_features(image_rgb, use_2d=use_2d)

    aggregated = {}
    for key in all_features[0].keys():
        vals = [f[key] for f in all_features if key in f]
        if vals:
            aggregated[key] = float(np.median(vals))
        else:
            aggregated[key] = 0.0
    for key in ['beta', 'D_mean', 'H_hurst', 'spectral_flatness']:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            aggregated[f'{key}_std'] = float(np.std(vals))
    aggregated['n_patches'] = len(all_features)
    return aggregated


# ─────────────────────────────────────────────
# 11. Feature Vector for a Patch
# ─────────────────────────────────────────────

def compute_features(patch: np.ndarray, use_2d: bool = True) -> dict:
    h, w, _ = patch.shape
    gray = to_luminance(patch)
    gray_norm = normalize_illumination(gray, sigma_ratio=0.25)

    if use_2d and h >= 32 and w >= 32:
        beta = estimate_beta_2d(gray_norm)
        D_spectral = beta2d_to_fractal_dimension(beta)
        H_hurst = beta2d_to_hurst(beta)
        D_vfd = variance_fractal_dimension_2d(gray_norm)
        ch_betas = estimate_channel_betas_2d(patch)
        D_box = box_counting_dimension(gray_norm)
    else:
        sig = extract_amplitude_signal(patch)
        freqs, amps = amplitude_spectrum(sig)
        beta = estimate_spectral_exponent(freqs, amps) if len(freqs) > 2 else 2.0
        D_spectral = beta_to_fractal_dimension(beta)
        H_hurst = beta_to_hurst(beta)
        D_vfd = variance_fractal_dimension(sig)
        channels = extract_channel_traces(patch)
        ch_betas = {}
        for ch, tr in channels.items():
            f, a = amplitude_spectrum(tr)
            ch_betas[ch] = estimate_spectral_exponent(f, a) if len(f) > 2 else 2.0
        D_box = box_counting_dimension(gray)

    sig = extract_amplitude_signal(patch)
    D_div = divider_fractal_dimension(sig)
    D_corr = correlation_dimension(sig)
    D_mean = np.mean([D_spectral, D_vfd, D_div])
    R = detection_response(sig, D_terrain=D_mean)
    contrast = gray_norm.std()
    mean_lum = gray_norm.mean()

    # NEW: Spectral signature features
    flatness = spectral_flatness(ch_betas)
    sig_dict = spectral_signature(ch_betas)

    return {
        'beta': beta,
        'D_spectral': D_spectral,
        'H_hurst': H_hurst,
        'D_vfd': D_vfd,
        'D_divider': D_div,
        'D_box': D_box,
        'D_corr': D_corr,
        'D_mean': D_mean,
        'R_response': R,
        'beta_R': ch_betas['R'],
        'beta_G': ch_betas['G'],
        'beta_B': ch_betas['B'],
        'spectral_flatness': flatness,
        'beta_ratio_R': sig_dict['beta_ratio_R'],
        'beta_ratio_G': sig_dict['beta_ratio_G'],
        'beta_ratio_B': sig_dict['beta_ratio_B'],
        'beta_spread': sig_dict['beta_spread'],
        'contrast': contrast,
        'mean_luminance': mean_lum,
    }
