"""Enhanced boundary / shape features for skin lesion classification.

Features implemented:
  - Circularity (圆度)
  - Boundary complexity (边界复杂度, perimeter²/area)
  - Solidity (凸包比, area/convex_hull_area)
  - Eccentricity (偏心率, from second moments)
  - Elongation (细长比, from eigenvalues of inertia tensor)
  - Equivalent diameter (等效直径)
  - Feret ratios (最大/最小卡尺直径比)
  - Hu moments (7 invariant moments)
  - Concavity (凹度)
  - Boundary fractal dimension estimate (边界分形维数估计)
  - Number of concave regions along contour
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_mask_props(mask):
    """Return (area, perimeter, coords) or None if mask is empty."""
    area = float(mask.sum())
    if area == 0:
        return None
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    ys, xs = np.where(mask)
    return area, perimeter, ys, xs


def _second_moments(ys, xs):
    """Compute central second moments of the binary region."""
    y_c = ys.mean()
    x_c = xs.mean()
    n = len(ys)
    mu20 = float(np.sum((xs - x_c) ** 2) / n)
    mu02 = float(np.sum((ys - y_c) ** 2) / n)
    mu11 = float(np.sum((xs - x_c) * (ys - y_c)) / n)
    return mu20, mu02, mu11


def _hu_moments(mask):
    """Compute 7 Hu invariant moments from a binary mask.

    Uses the standard formulation via raw and central moments.
    """
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return {f"boundary_hu_{i}": 0.0 for i in range(7)}

    # Raw moments
    m00 = float(len(ys))
    x_bar = float(xs.mean())
    y_bar = float(ys.mean())

    # Central moments (normalized by m00^(p/2+1) for scale invariance)
    def central(p, q):
        return float(np.sum((xs - x_bar) ** p * (ys - y_bar) ** q))

    mu20 = central(2, 0)
    mu02 = central(0, 2)
    mu11 = central(1, 1)
    mu30 = central(3, 0)
    mu03 = central(0, 3)
    mu21 = central(2, 1)
    mu12 = central(1, 2)

    # Scale normalization: eta_pq = mu_pq / m00^((p+q)/2 + 1)
    def eta(pq_val, p, q):
        return pq_val / (m00 ** ((p + q) / 2 + 1))

    e20 = eta(mu20, 2, 0)
    e02 = eta(mu02, 0, 2)
    e11 = eta(mu11, 1, 1)
    e30 = eta(mu30, 3, 0)
    e03 = eta(mu03, 0, 3)
    e21 = eta(mu21, 2, 1)
    e12 = eta(mu12, 1, 2)

    # Hu moments
    hu = [
        e20 + e02,
        (e20 - e02) ** 2 + 4 * e11 ** 2,
        (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2,
        (e30 + e12) ** 2 + (e21 + e03) ** 2,
        (e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
        + (3 * e21 - e03) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2),
        (e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2)
        + 4 * e11 * (e30 + e12) * (e21 + e03),
        (3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
        - (e30 - 3 * e12) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2),
    ]

    # Log-transform to compress range (skip sign for hu7)
    features = {}
    for i, val in enumerate(hu):
        if val >= 0:
            features[f"boundary_hu_{i}"] = float(np.log1p(abs(val)))
        else:
            features[f"boundary_hu_{i}"] = float(-np.log1p(abs(val)))
    return features


def _convex_hull_area(mask):
    """Compute the area of the convex hull of the lesion region."""
    ys, xs = np.where(mask)
    if len(ys) < 3:
        return float(mask.sum())
    # Subsample if too many points (ConvexHull is O(n log n))
    n = len(ys)
    if n > 2000:
        idx = np.random.default_rng(42).choice(n, 2000, replace=False)
        pts = np.column_stack([xs[idx], ys[idx]])
    else:
        pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
        return hull.volume  # volume = area in 2D
    except Exception:
        return float(mask.sum())


def _feret_diameter(mask):
    """Estimate max and min Feret (caliper) diameters via rotating calipers approximation."""
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return 0.0, 0.0

    # Use convex hull vertices for efficiency
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) > 1000:
        idx = np.random.default_rng(42).choice(len(pts), 1000, replace=False)
        pts = pts[idx]
    try:
        hull = ConvexHull(pts)
        vertices = pts[hull.vertices]
    except Exception:
        vertices = pts

    # Sample angles and measure projection width
    max_d = 0.0
    min_d = float("inf")
    for angle in np.linspace(0, np.pi, 36, endpoint=False):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        projections = vertices[:, 0] * cos_a + vertices[:, 1] * sin_a
        width = projections.max() - projections.min()
        if width > max_d:
            max_d = width
        if width < min_d:
            min_d = width

    if min_d == float("inf"):
        min_d = 0.0
    return max_d, min_d


def _boundary_fractal_dimension(mask, max_iter=5):
    """Estimate boundary fractal dimension using box-counting on the contour.

    Returns the slope of log(N) vs log(1/s), which approximates fractal dimension.
    A smooth circle has D ≈ 1.0; more irregular boundaries have D > 1.0.
    """
    # Get boundary pixels
    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    if not boundary.any():
        return 1.0

    ys, xs = np.where(boundary)
    if len(ys) < 2:
        return 1.0

    coords = np.column_stack([xs, ys]).astype(np.float64)
    # Normalize to [0, 1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    coords_n = (coords - mins) / ranges

    sizes = []
    counts = []
    for i in range(1, max_iter + 1):
        s = 2 ** i
        bin_size = 1.0 / s
        bins_x = (coords_n[:, 0] / bin_size).astype(int)
        bins_y = (coords_n[:, 1] / bin_size).astype(int)
        occupied = set(zip(bins_x.tolist(), bins_y.tolist()))
        sizes.append(s)
        counts.append(len(occupied))

    if len(sizes) < 2:
        return 1.0

    # Linear regression on log-log
    log_s = np.log(sizes)
    log_n = np.log(counts)
    slope = float(np.polyfit(log_s, log_n, 1)[0])
    return max(slope, 1.0)


def _count_concavities(mask, n_sectors=8):
    """Count how many sectors (out of n_sectors) have concave boundaries.

    Divide the lesion into n_sectors angular sectors from the centroid.
    In each sector, check if the boundary goes inward (concave) at any point.
    More concavities → more irregular boundary (typical of melanoma).
    """
    ys, xs = np.where(mask)
    if len(ys) < 3:
        return 0

    y_c, x_c = ys.mean(), xs.mean()
    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    bys, bxs = np.where(boundary)
    if len(bys) == 0:
        return 0

    # Compute angles of boundary points relative to centroid
    angles = np.arctan2(bys - y_c, bxs - x_c)  # range [-pi, pi]
    # Distance from centroid
    dists = np.sqrt((bys - y_c) ** 2 + (bxs - x_c) ** 2)

    # Average distance per sector
    sector_size = 2 * np.pi / n_sectors
    concavities = 0
    for i in range(n_sectors):
        a_min = -np.pi + i * sector_size
        a_max = a_min + sector_size
        mask_sector = (angles >= a_min) & (angles < a_max)
        if mask_sector.sum() < 2:
            continue
        sector_dists = dists[mask_sector]
        # If the max distance is much larger than the min, there's a concavity
        # Use coefficient of variation as a proxy
        if sector_dists.mean() > 0:
            cv = sector_dists.std() / sector_dists.mean()
            if cv > 0.15:  # threshold for "irregular"
                concavities += 1

    return concavities


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def extract_boundary_features(image, mask):
    """Extract enhanced boundary and shape features.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) RGB image (not used directly but kept for API consistency)
    mask  : np.ndarray  (H, W) bool mask of the lesion region

    Returns
    -------
    dict of feature_name → float
    """
    props = _safe_mask_props(mask)
    if props is None:
        return {
            "boundary_circularity": 0.0,
            "boundary_complexity": 0.0,
            "boundary_solidity": 0.0,
            "boundary_eccentricity": 0.0,
            "boundary_elongation": 0.0,
            "boundary_equiv_diameter": 0.0,
            "boundary_feret_ratio": 0.0,
            "boundary_feret_max": 0.0,
            "boundary_feret_min": 0.0,
            "boundary_concavity": 0.0,
            "boundary_fractal_dim": 1.0,
            "boundary_n_concavities": 0,
            **{f"boundary_hu_{i}": 0.0 for i in range(7)},
        }

    area, perimeter, ys, xs = props
    features = {}

    # 1. Circularity: 4π·area/perimeter²  (1.0 = perfect circle)
    features["boundary_circularity"] = float(
        4 * np.pi * area / max(perimeter ** 2, 1)
    )

    # 2. Boundary complexity: perimeter²/area  (related to compactness inverse)
    features["boundary_complexity"] = float(perimeter ** 2 / max(area, 1))

    # 3. Solidity: area / convex_hull_area
    hull_area = _convex_hull_area(mask)
    features["boundary_solidity"] = float(area / max(hull_area, 1))

    # 4. Eccentricity from second moments
    mu20, mu02, mu11 = _second_moments(ys, xs)
    # Eigenvalues of inertia tensor
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 ** 2
    disc = max(trace ** 2 - 4 * det, 0)
    lambda1 = (trace + np.sqrt(disc)) / 2  # larger eigenvalue
    lambda2 = (trace - np.sqrt(disc)) / 2  # smaller eigenvalue
    if lambda1 > 0:
        features["boundary_eccentricity"] = float(np.sqrt(1 - lambda2 / lambda1))
    else:
        features["boundary_eccentricity"] = 0.0

    # 5. Elongation: ratio of major to minor axis of equivalent ellipse
    features["boundary_elongation"] = float(lambda1 / max(lambda2, 1e-10))

    # 6. Equivalent diameter: diameter of circle with same area
    features["boundary_equiv_diameter"] = float(np.sqrt(4 * area / np.pi))

    # 7. Feret diameters
    feret_max, feret_min = _feret_diameter(mask)
    features["boundary_feret_max"] = feret_max
    features["boundary_feret_min"] = feret_min
    features["boundary_feret_ratio"] = float(feret_max / max(feret_min, 1))

    # 8. Concavity: 1 - solidity (how much area is missing vs convex hull)
    features["boundary_concavity"] = 1.0 - features["boundary_solidity"]

    # 9. Fractal dimension estimate
    features["boundary_fractal_dim"] = _boundary_fractal_dimension(mask)

    # 10. Number of concave sectors
    features["boundary_n_concavities"] = _count_concavities(mask)

    # 11. Hu moments (7 invariant moments)
    features.update(_hu_moments(mask))

    return features
