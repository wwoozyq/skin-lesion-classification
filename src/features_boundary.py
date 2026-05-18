import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


def _empty_features():
    features = {
        "boundary_circularity": 0.0,
        "boundary_complexity": 0.0,
        "boundary_solidity": 0.0,
        "boundary_eccentricity": 0.0,
        "boundary_elongation": 0.0,
        "boundary_equiv_diameter": 0.0,
        "boundary_feret_max": 0.0,
        "boundary_feret_min": 0.0,
        "boundary_feret_ratio": 0.0,
        "boundary_concavity": 0.0,
        "boundary_fractal_dim": 1.0,
        "boundary_n_irregular_sectors": 0.0,
    }
    features.update({f"boundary_moment_{idx}": 0.0 for idx in range(6)})
    return features


def _convex_hull_area(xs, ys):
    if len(xs) < 3:
        return float(len(xs))
    points = np.column_stack([xs, ys]).astype(np.float64)
    if len(points) > 2000:
        points = points[np.random.default_rng(42).choice(len(points), 2000, replace=False)]
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return float(len(xs))


def _feret_diameters(xs, ys):
    if len(xs) < 2:
        return 0.0, 0.0
    points = np.column_stack([xs, ys]).astype(np.float64)
    if len(points) > 1000:
        points = points[np.random.default_rng(42).choice(len(points), 1000, replace=False)]
    try:
        points = points[ConvexHull(points).vertices]
    except Exception:
        pass

    widths = []
    for angle in np.linspace(0, np.pi, 36, endpoint=False):
        projection = points[:, 0] * np.cos(angle) + points[:, 1] * np.sin(angle)
        widths.append(projection.max() - projection.min())
    return float(max(widths)), float(min(widths))


def _boundary_fractal_dimension(mask):
    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    ys, xs = np.where(boundary)
    if len(xs) < 2:
        return 1.0

    coords = np.column_stack([xs, ys]).astype(np.float64)
    mins = coords.min(axis=0)
    ranges = coords.max(axis=0) - mins
    ranges[ranges == 0] = 1
    coords = (coords - mins) / ranges

    sizes = np.array([2, 4, 8, 16, 32], dtype=np.float64)
    counts = []
    for size in sizes:
        bins = np.floor(coords * size).astype(int)
        counts.append(len(set(map(tuple, bins))))
    counts = np.array(counts, dtype=np.float64)
    valid = counts > 0
    if valid.sum() < 2:
        return 1.0
    slope = np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)[0]
    return float(max(slope, 1.0))


def _irregular_sector_count(mask, n_sectors=8):
    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    ys, xs = np.where(boundary)
    if len(xs) < 3:
        return 0.0
    center_x = xs.mean()
    center_y = ys.mean()
    angles = np.arctan2(ys - center_y, xs - center_x)
    radius = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    count = 0
    sector_width = 2 * np.pi / n_sectors
    for idx in range(n_sectors):
        low = -np.pi + idx * sector_width
        high = low + sector_width
        sector = (angles >= low) & (angles < high)
        if sector.sum() < 2:
            continue
        sector_radius = radius[sector]
        if sector_radius.std() / max(sector_radius.mean(), 1e-6) > 0.15:
            count += 1
    return float(count)


def _central_moments(xs, ys):
    x_center = xs.mean()
    y_center = ys.mean()
    x = xs - x_center
    y = ys - y_center
    scale = max(len(xs), 1)
    return {
        "boundary_moment_0": float((x ** 2).sum() / scale),
        "boundary_moment_1": float((y ** 2).sum() / scale),
        "boundary_moment_2": float((x * y).sum() / scale),
        "boundary_moment_3": float((x ** 3).sum() / scale),
        "boundary_moment_4": float((y ** 3).sum() / scale),
        "boundary_moment_5": float((x ** 2 * y ** 2).sum() / max(scale ** 2, 1)),
    }


def extract_boundary_features(image, mask):
    del image
    area = float(mask.sum())
    if area == 0:
        return _empty_features()

    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    perimeter = float(boundary.sum())
    ys, xs = np.where(mask)

    mu20 = np.mean((xs - xs.mean()) ** 2)
    mu02 = np.mean((ys - ys.mean()) ** 2)
    mu11 = np.mean((xs - xs.mean()) * (ys - ys.mean()))
    trace = mu20 + mu02
    determinant = mu20 * mu02 - mu11 ** 2
    discriminant = max(trace ** 2 - 4 * determinant, 0.0)
    lambda1 = (trace + np.sqrt(discriminant)) / 2
    lambda2 = (trace - np.sqrt(discriminant)) / 2

    hull_area = _convex_hull_area(xs, ys)
    feret_max, feret_min = _feret_diameters(xs, ys)
    features = {
        "boundary_circularity": float(4 * np.pi * area / max(perimeter ** 2, 1.0)),
        "boundary_complexity": float(perimeter ** 2 / max(area, 1.0)),
        "boundary_solidity": float(area / max(hull_area, 1.0)),
        "boundary_eccentricity": float(np.sqrt(max(1 - lambda2 / max(lambda1, 1e-10), 0.0))),
        "boundary_elongation": float(lambda1 / max(lambda2, 1e-10)),
        "boundary_equiv_diameter": float(np.sqrt(4 * area / np.pi)),
        "boundary_feret_max": feret_max,
        "boundary_feret_min": feret_min,
        "boundary_feret_ratio": float(feret_max / max(feret_min, 1.0)),
        "boundary_concavity": float(1.0 - area / max(hull_area, 1.0)),
        "boundary_fractal_dim": _boundary_fractal_dimension(mask),
        "boundary_n_irregular_sectors": _irregular_sector_count(mask),
    }
    features.update(_central_moments(xs.astype(np.float32), ys.astype(np.float32)))
    return features
