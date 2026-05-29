"""Dermoscopy-inspired structural features.

Implements lesion-relative percentile-based detectors for clinical
dermoscopy diagnostic structures that the existing handcrafted feature
blocks do not explicitly model:

1. Clinical color diversity (Lab-based spread, not fixed HSV thresholds)
2. Blue-white structure (lesion-relative blue-shifted bright pixels)
3. Regression-like areas (lesion-relative bright low-saturation patches)
4. PCA-axis spatial color asymmetry (rotation-invariant via principal axis,
   with low-eccentricity fallback to zeros)
5. Vascular emphasis (lesion-relative red-shifted pixels)

All thresholds are defined relative to each lesion's own Lab percentiles,
so the features are robust to global illumination differences across
dermatoscopes without requiring a global color-constancy preprocessing
step. This is deliberate: shades_of_gray was documented as net-negative
on the strong cascade variant (see docs/OVERNIGHT_EXPLORATION_RESULTS.md),
and color is signal for these features, not just noise to normalize.

Output keys are all prefixed `dermoscopy_*` for traceability.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color


_MIN_LESION_PIXELS = 50
_MIN_ECCENTRICITY_FOR_ASYMMETRY = 0.3
_FEATURE_KEYS: tuple[str, ...] = (
    # 1. color diversity (6)
    "dermoscopy_lab_L_range_p5_95",
    "dermoscopy_lab_a_range_p5_95",
    "dermoscopy_lab_b_range_p5_95",
    "dermoscopy_lab_octant_entropy",
    "dermoscopy_lab_pca1_std",
    "dermoscopy_lab_max_octant_fraction",
    # 2. blue-white structure (3)
    "dermoscopy_bluewhite_area_ratio",
    "dermoscopy_bluewhite_max_component_ratio",
    "dermoscopy_bluewhite_n_components",
    # 3. regression-like (white + blue-gray, 4)
    "dermoscopy_regression_white_area_ratio",
    "dermoscopy_regression_white_max_component_ratio",
    "dermoscopy_regression_white_n_components",
    "dermoscopy_regression_bluegray_area_ratio",
    # 4. PCA-axis spatial color asymmetry (7)
    "dermoscopy_eccentricity",
    "dermoscopy_asym_pca1_L",
    "dermoscopy_asym_pca1_a",
    "dermoscopy_asym_pca1_b",
    "dermoscopy_asym_pca2_L",
    "dermoscopy_asym_pca2_a",
    "dermoscopy_asym_pca2_b",
    # 5. vascular emphasis (2)
    "dermoscopy_vascular_area_ratio",
    "dermoscopy_vascular_max_a_within_lesion",
)


def _zero_features() -> dict:
    return {key: 0.0 for key in _FEATURE_KEYS}


def _lesion_lab_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return Lab pixel array of shape (N, 3) for lesion pixels only."""
    rgb_f = np.asarray(image, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    return lab[mask]


def _color_diversity(lab_pixels: np.ndarray) -> dict:
    """Lesion-relative color spread in Lab space."""
    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]

    L_range = float(np.percentile(L, 95) - np.percentile(L, 5))
    a_range = float(np.percentile(a, 95) - np.percentile(a, 5))
    b_range = float(np.percentile(b, 95) - np.percentile(b, 5))

    # Octant occupancy relative to lesion median: 8 bins from (L, a, b) > median
    L_med = float(np.median(L))
    a_med = float(np.median(a))
    b_med = float(np.median(b))
    bits = (
        ((L > L_med).astype(np.int8) << 2)
        | ((a > a_med).astype(np.int8) << 1)
        | (b > b_med).astype(np.int8)
    )
    counts = np.bincount(bits, minlength=8).astype(np.float64)
    fractions = counts / counts.sum()
    nonzero = fractions[fractions > 0]
    octant_entropy = float(-(nonzero * np.log2(nonzero)).sum()) if nonzero.size else 0.0
    max_octant_fraction = float(fractions.max())

    # PCA-1 std in Lab: dominant axis of color variation
    centered = lab_pixels - lab_pixels.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(len(lab_pixels) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    pca1_std = float(np.sqrt(max(eigvals[-1], 0.0)))

    return {
        "dermoscopy_lab_L_range_p5_95": L_range,
        "dermoscopy_lab_a_range_p5_95": a_range,
        "dermoscopy_lab_b_range_p5_95": b_range,
        "dermoscopy_lab_octant_entropy": octant_entropy,
        "dermoscopy_lab_pca1_std": pca1_std,
        "dermoscopy_lab_max_octant_fraction": max_octant_fraction,
    }


def _component_stats(binary_mask: np.ndarray, lesion_area: int) -> tuple[float, float, int]:
    """Connected-component summary over a binary mask, normalized by lesion area."""
    if binary_mask.sum() == 0 or lesion_area == 0:
        return 0.0, 0.0, 0
    labelled, n = ndi.label(binary_mask)
    if n == 0:
        return 0.0, 0.0, 0
    sizes = np.bincount(labelled.ravel())[1:]
    n_significant = int((sizes >= max(10, lesion_area // 200)).sum())
    return (
        float(binary_mask.sum() / lesion_area),
        float(sizes.max() / lesion_area),
        n_significant,
    )


def _blue_white_structure(image: np.ndarray, mask: np.ndarray) -> dict:
    """Lesion-relative blue-shifted, bright, low-chroma pixels (blue-white veil-like)."""
    rgb_f = np.asarray(image, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)

    L_lesion = L[mask]
    b_lesion = b[mask]
    chroma_lesion = chroma[mask]

    L_thresh = float(np.percentile(L_lesion, 60))
    b_thresh = float(np.percentile(b_lesion, 40))  # blue-shifted = lower b*
    chroma_thresh = float(np.percentile(chroma_lesion, 50))

    candidate = mask & (L > L_thresh) & (b < b_thresh) & (chroma < chroma_thresh)
    area_ratio, max_comp_ratio, n_components = _component_stats(candidate, int(mask.sum()))
    return {
        "dermoscopy_bluewhite_area_ratio": area_ratio,
        "dermoscopy_bluewhite_max_component_ratio": max_comp_ratio,
        "dermoscopy_bluewhite_n_components": float(n_components),
    }


def _regression_areas(image: np.ndarray, mask: np.ndarray) -> dict:
    """Lesion-relative bright low-saturation (white) and dark gray (pepper) patches."""
    rgb_f = np.asarray(image, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)

    L_lesion = L[mask]
    chroma_lesion = chroma[mask]

    # White regression: top 25% L within lesion, bottom 25% chroma
    L_hi = float(np.percentile(L_lesion, 75))
    chroma_lo = float(np.percentile(chroma_lesion, 25))
    white_mask = mask & (L > L_hi) & (chroma < chroma_lo)
    w_area, w_max, w_n = _component_stats(white_mask, int(mask.sum()))

    # Blue-gray "pepper": bottom 50% L within lesion AND low chroma
    L_med = float(np.percentile(L_lesion, 50))
    bluegray_mask = mask & (L < L_med) & (chroma < chroma_lo)
    bluegray_area_ratio = float(bluegray_mask.sum() / max(mask.sum(), 1))

    return {
        "dermoscopy_regression_white_area_ratio": w_area,
        "dermoscopy_regression_white_max_component_ratio": w_max,
        "dermoscopy_regression_white_n_components": float(w_n),
        "dermoscopy_regression_bluegray_area_ratio": bluegray_area_ratio,
    }


def _spatial_color_asymmetry(image: np.ndarray, mask: np.ndarray) -> dict:
    """Color asymmetry along lesion's PCA principal axes.

    Computes PCA on the mask coordinates. For low-eccentricity (near-round)
    lesions the principal axis direction is unstable, so we return zeros
    with the eccentricity feature still populated.
    """
    ys, xs = np.where(mask)
    coords = np.stack([ys.astype(np.float64), xs.astype(np.float64)], axis=1)
    coords_centered = coords - coords.mean(axis=0, keepdims=True)
    cov = (coords_centered.T @ coords_centered) / max(len(coords) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; swap so [0] is dominant
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    if eigvals[0] <= 0:
        return {key: 0.0 for key in _FEATURE_KEYS if key.startswith("dermoscopy_asym_") or key == "dermoscopy_eccentricity"}
    eccentricity = float(np.sqrt(max(1.0 - eigvals[1] / eigvals[0], 0.0)))
    result = {"dermoscopy_eccentricity": eccentricity}
    asym_keys = (
        "dermoscopy_asym_pca1_L", "dermoscopy_asym_pca1_a", "dermoscopy_asym_pca1_b",
        "dermoscopy_asym_pca2_L", "dermoscopy_asym_pca2_a", "dermoscopy_asym_pca2_b",
    )
    if eccentricity < _MIN_ECCENTRICITY_FOR_ASYMMETRY:
        for key in asym_keys:
            result[key] = 0.0
        return result

    rgb_f = np.asarray(image, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    lab_lesion = lab[mask]  # aligns with coords order

    # Project lesion coords onto principal axes and split at median (= zero after centering)
    projections = coords_centered @ eigvecs  # shape (N, 2)
    for axis_idx, prefix in enumerate(["pca1", "pca2"]):
        proj = projections[:, axis_idx]
        side_a = proj >= 0
        side_b = ~side_a
        if side_a.sum() < 10 or side_b.sum() < 10:
            for ch_idx, ch in enumerate(["L", "a", "b"]):
                result[f"dermoscopy_asym_{prefix}_{ch}"] = 0.0
            continue
        for ch_idx, ch in enumerate(["L", "a", "b"]):
            mean_a = float(lab_lesion[side_a, ch_idx].mean())
            mean_b = float(lab_lesion[side_b, ch_idx].mean())
            result[f"dermoscopy_asym_{prefix}_{ch}"] = abs(mean_a - mean_b)
    return result


def _vascular_emphasis(image: np.ndarray, mask: np.ndarray) -> dict:
    """Red-shifted pixels within lesion (lesion-relative a* upper quartile)."""
    rgb_f = np.asarray(image, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    L = lab[..., 0]
    a = lab[..., 1]

    a_lesion = a[mask]
    L_lesion = L[mask]
    a_thresh = float(np.percentile(a_lesion, 75))

    # Clamp on L to exclude pure black (L<20) and pure white (L>85) regions
    candidate = mask & (a > a_thresh) & (L > 20) & (L < 85)
    area_ratio = float(candidate.sum() / max(mask.sum(), 1))
    max_a = float(a_lesion.max())
    return {
        "dermoscopy_vascular_area_ratio": area_ratio,
        "dermoscopy_vascular_max_a_within_lesion": max_a,
    }


def extract_dermoscopy_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """Compute all dermoscopy structural features.

    Args:
        image: uint8 RGB array, shape (H, W, 3).
        mask:  bool array, shape (H, W). Lesion = True.

    Returns:
        dict mapping ``dermoscopy_*`` feature names to floats. On empty or
        tiny lesion masks all features are zero (with the same keys).
    """
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < _MIN_LESION_PIXELS:
        return _zero_features()

    lab_pixels = _lesion_lab_pixels(image, mask)

    features = {}
    features.update(_color_diversity(lab_pixels))
    features.update(_blue_white_structure(image, mask))
    features.update(_regression_areas(image, mask))
    features.update(_spatial_color_asymmetry(image, mask))
    features.update(_vascular_emphasis(image, mask))
    # Guarantee all keys present even if a sub-function returned a partial dict
    for key in _FEATURE_KEYS:
        features.setdefault(key, 0.0)
    return features
