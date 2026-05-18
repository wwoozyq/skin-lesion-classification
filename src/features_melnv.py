import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _safe_mask(mask):
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _safe_pixels(arr, mask):
    pixels = arr[mask]
    if pixels.size == 0:
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1)
    return pixels


def _foreground_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _entropy(values, bins=16, value_range=None):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _mad(values):
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _finalize(features):
    clean = {}
    for key, value in features.items():
        value = float(value)
        clean[key] = value if np.isfinite(value) else 0.0
    return clean


def _lab_spread_features(lab_pixels):
    features = {}
    channel_names = ["l", "a", "b"]
    for idx, name in enumerate(channel_names):
        channel = lab_pixels[:, idx].astype(np.float32)
        features[f"melnv_lab_{name}_std"] = channel.std()
        features[f"melnv_lab_{name}_iqr"] = np.percentile(channel, 75) - np.percentile(channel, 25)
        features[f"melnv_lab_{name}_p95_p05"] = np.percentile(channel, 95) - np.percentile(channel, 5)
        features[f"melnv_lab_{name}_mad"] = _mad(channel)
        features[f"melnv_lab_{name}_entropy"] = _entropy(channel, bins=16, value_range=(0, 255))

    a = lab_pixels[:, 1].astype(np.float32) - 128.0
    b = lab_pixels[:, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    features["melnv_lab_chroma_mean"] = chroma.mean()
    features["melnv_lab_chroma_std"] = chroma.std()
    features["melnv_lab_chroma_p90_p10"] = np.percentile(chroma, 90) - np.percentile(chroma, 10)
    features["melnv_lab_chroma_entropy"] = _entropy(chroma, bins=16)
    return features


def _hsv_variegation_features(hsv_pixels):
    h = hsv_pixels[:, 0].astype(np.float32)
    s = hsv_pixels[:, 1].astype(np.float32)
    v = hsv_pixels[:, 2].astype(np.float32)

    angle = h / 255.0 * 2.0 * np.pi
    weights = np.clip(s, 0, 255) / 255.0
    if weights.sum() > 1e-6:
        mean_cos = np.sum(np.cos(angle) * weights) / weights.sum()
        mean_sin = np.sum(np.sin(angle) * weights) / weights.sum()
        hue_concentration = np.sqrt(mean_cos ** 2 + mean_sin ** 2)
    else:
        hue_concentration = 0.0

    high_sat = s > 110
    low_value = v < 95
    return {
        "melnv_hsv_s_std": s.std(),
        "melnv_hsv_s_p90_p10": np.percentile(s, 90) - np.percentile(s, 10),
        "melnv_hsv_v_std": v.std(),
        "melnv_hsv_v_p90_p10": np.percentile(v, 90) - np.percentile(v, 10),
        "melnv_hsv_high_sat_ratio": high_sat.mean(),
        "melnv_hsv_low_value_ratio": low_value.mean(),
        "melnv_hsv_dark_high_sat_ratio": (high_sat & low_value).mean(),
        "melnv_hsv_hue_entropy": _entropy(h[high_sat] if high_sat.any() else h, bins=12, value_range=(0, 255)),
        "melnv_hsv_hue_circular_variance": 1.0 - hue_concentration,
    }


def _pca_frame(mask):
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs, ys]).astype(np.float64)
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if not np.isfinite(vh).all():
        return None
    major = vh[0]
    minor = vh[1]
    proj_major = centered[:, 0] * major[0] + centered[:, 1] * major[1]
    proj_minor = centered[:, 0] * minor[0] + centered[:, 1] * minor[1]
    return coords, proj_major, proj_minor


def _split_features(prefix, projection, lab_pixels):
    pos = projection >= 0
    neg = projection < 0
    features = {
        f"{prefix}_area_balance": 0.0,
        f"{prefix}_lab_mean_diff_norm": 0.0,
        f"{prefix}_lab_l_mean_diff": 0.0,
        f"{prefix}_lab_a_mean_diff": 0.0,
        f"{prefix}_lab_b_mean_diff": 0.0,
        f"{prefix}_lab_l_std_diff": 0.0,
        f"{prefix}_dark_ratio_diff": 0.0,
    }
    if pos.sum() == 0 or neg.sum() == 0:
        return features

    pos_lab = lab_pixels[pos].astype(np.float32)
    neg_lab = lab_pixels[neg].astype(np.float32)
    features[f"{prefix}_area_balance"] = abs(pos.sum() - neg.sum()) / max(pos.sum() + neg.sum(), 1)
    diff = np.abs(pos_lab.mean(axis=0) - neg_lab.mean(axis=0))
    features[f"{prefix}_lab_mean_diff_norm"] = np.linalg.norm(diff) / 255.0
    features[f"{prefix}_lab_l_mean_diff"] = diff[0]
    features[f"{prefix}_lab_a_mean_diff"] = diff[1]
    features[f"{prefix}_lab_b_mean_diff"] = diff[2]
    features[f"{prefix}_lab_l_std_diff"] = abs(pos_lab[:, 0].std() - neg_lab[:, 0].std())
    features[f"{prefix}_dark_ratio_diff"] = abs((pos_lab[:, 0] < 95).mean() - (neg_lab[:, 0] < 95).mean())
    return features


def _pca_asymmetry_features(lab, mask):
    features = {}
    frame = _pca_frame(mask)
    empty_keys = [
        "melnv_pca_major_area_balance",
        "melnv_pca_major_lab_mean_diff_norm",
        "melnv_pca_major_lab_l_mean_diff",
        "melnv_pca_major_lab_a_mean_diff",
        "melnv_pca_major_lab_b_mean_diff",
        "melnv_pca_major_lab_l_std_diff",
        "melnv_pca_major_dark_ratio_diff",
        "melnv_pca_minor_area_balance",
        "melnv_pca_minor_lab_mean_diff_norm",
        "melnv_pca_minor_lab_l_mean_diff",
        "melnv_pca_minor_lab_a_mean_diff",
        "melnv_pca_minor_lab_b_mean_diff",
        "melnv_pca_minor_lab_l_std_diff",
        "melnv_pca_minor_dark_ratio_diff",
        "melnv_pca_quadrant_area_cv",
        "melnv_pca_quadrant_dominant_area_ratio",
        "melnv_pca_quadrant_lab_l_mean_range",
        "melnv_pca_quadrant_lab_a_mean_range",
        "melnv_pca_quadrant_lab_b_mean_range",
        "melnv_pca_quadrant_dark_ratio_range",
    ]
    if frame is None:
        return {key: 0.0 for key in empty_keys}

    _, proj_major, proj_minor = frame
    lab_pixels = lab[mask].astype(np.float32)
    features.update(_split_features("melnv_pca_major", proj_major, lab_pixels))
    features.update(_split_features("melnv_pca_minor", proj_minor, lab_pixels))

    quadrants = [
        (proj_major >= 0) & (proj_minor >= 0),
        (proj_major >= 0) & (proj_minor < 0),
        (proj_major < 0) & (proj_minor >= 0),
        (proj_major < 0) & (proj_minor < 0),
    ]
    means = []
    dark_ratios = []
    area_ratios = []
    for quadrant in quadrants:
        count = quadrant.sum()
        area_ratios.append(count / max(len(lab_pixels), 1))
        if count < 5:
            continue
        q_lab = lab_pixels[quadrant]
        means.append(q_lab.mean(axis=0))
        dark_ratios.append((q_lab[:, 0] < 95).mean())

    area_ratios = np.array(area_ratios, dtype=np.float32)
    features["melnv_pca_quadrant_area_cv"] = area_ratios.std() / max(area_ratios.mean(), 1e-6)
    features["melnv_pca_quadrant_dominant_area_ratio"] = area_ratios.max() if area_ratios.size else 0.0
    if len(means) >= 2:
        means = np.vstack(means)
        features["melnv_pca_quadrant_lab_l_mean_range"] = means[:, 0].max() - means[:, 0].min()
        features["melnv_pca_quadrant_lab_a_mean_range"] = means[:, 1].max() - means[:, 1].min()
        features["melnv_pca_quadrant_lab_b_mean_range"] = means[:, 2].max() - means[:, 2].min()
        features["melnv_pca_quadrant_dark_ratio_range"] = max(dark_ratios) - min(dark_ratios) if dark_ratios else 0.0
    else:
        features["melnv_pca_quadrant_lab_l_mean_range"] = 0.0
        features["melnv_pca_quadrant_lab_a_mean_range"] = 0.0
        features["melnv_pca_quadrant_lab_b_mean_range"] = 0.0
        features["melnv_pca_quadrant_dark_ratio_range"] = 0.0
    return features


def _core_border_features(lab, hsv, mask):
    core = ndi.binary_erosion(mask, iterations=2)
    if core.sum() < 10:
        core = ndi.binary_erosion(mask, iterations=1)
    border = mask & ~core

    keys = [
        "melnv_core_border_lab_mean_diff_norm",
        "melnv_core_border_l_diff",
        "melnv_core_border_a_diff",
        "melnv_core_border_b_diff",
        "melnv_core_border_saturation_diff",
        "melnv_core_border_value_diff",
        "melnv_core_border_dark_ratio_diff",
        "melnv_core_border_chroma_std_diff",
    ]
    if core.sum() == 0 or border.sum() == 0:
        return {key: 0.0 for key in keys}

    core_lab = lab[core].astype(np.float32)
    border_lab = lab[border].astype(np.float32)
    core_hsv = hsv[core].astype(np.float32)
    border_hsv = hsv[border].astype(np.float32)

    diff = border_lab.mean(axis=0) - core_lab.mean(axis=0)
    core_chroma = np.sqrt((core_lab[:, 1] - 128) ** 2 + (core_lab[:, 2] - 128) ** 2)
    border_chroma = np.sqrt((border_lab[:, 1] - 128) ** 2 + (border_lab[:, 2] - 128) ** 2)
    return {
        "melnv_core_border_lab_mean_diff_norm": np.linalg.norm(diff) / 255.0,
        "melnv_core_border_l_diff": diff[0],
        "melnv_core_border_a_diff": diff[1],
        "melnv_core_border_b_diff": diff[2],
        "melnv_core_border_saturation_diff": border_hsv[:, 1].mean() - core_hsv[:, 1].mean(),
        "melnv_core_border_value_diff": border_hsv[:, 2].mean() - core_hsv[:, 2].mean(),
        "melnv_core_border_dark_ratio_diff": (border_lab[:, 0] < 95).mean() - (core_lab[:, 0] < 95).mean(),
        "melnv_core_border_chroma_std_diff": border_chroma.std() - core_chroma.std(),
    }


def _grid_features(lab, hsv, mask):
    x0, y0, x1, y1 = _foreground_bbox(mask)
    xs = np.linspace(x0, x1, 4).astype(int)
    ys = np.linspace(y0, y1, 4).astype(int)
    lesion_area = max(mask.sum(), 1)
    min_pixels = max(5, int(lesion_area * 0.005))

    lab_means = []
    lab_stds = []
    dark_ratios = []
    sat_ratios = []
    area_ratios = []
    for row in range(3):
        for col in range(3):
            cell = np.zeros(mask.shape, dtype=bool)
            cell[ys[row]:ys[row + 1], xs[col]:xs[col + 1]] = True
            cell_mask = mask & cell
            count = cell_mask.sum()
            area_ratios.append(count / lesion_area)
            if count < min_pixels:
                continue
            cell_lab = lab[cell_mask].astype(np.float32)
            cell_hsv = hsv[cell_mask].astype(np.float32)
            lab_means.append(cell_lab.mean(axis=0))
            lab_stds.append(cell_lab.std(axis=0))
            dark_ratios.append((cell_lab[:, 0] < 95).mean())
            sat_ratios.append((cell_hsv[:, 1] > 110).mean())

    area_ratios = np.array(area_ratios, dtype=np.float32)
    features = {
        "melnv_grid_occupied_ratio": len(lab_means) / 9.0,
        "melnv_grid_area_cv": area_ratios.std() / max(area_ratios.mean(), 1e-6),
        "melnv_grid_dominant_area_ratio": area_ratios.max() if area_ratios.size else 0.0,
        "melnv_grid_lab_l_mean_std": 0.0,
        "melnv_grid_lab_a_mean_std": 0.0,
        "melnv_grid_lab_b_mean_std": 0.0,
        "melnv_grid_lab_l_std_mean": 0.0,
        "melnv_grid_dark_ratio_range": 0.0,
        "melnv_grid_high_sat_ratio_range": 0.0,
    }
    if len(lab_means) >= 2:
        lab_means = np.vstack(lab_means)
        lab_stds = np.vstack(lab_stds)
        features["melnv_grid_lab_l_mean_std"] = lab_means[:, 0].std()
        features["melnv_grid_lab_a_mean_std"] = lab_means[:, 1].std()
        features["melnv_grid_lab_b_mean_std"] = lab_means[:, 2].std()
        features["melnv_grid_lab_l_std_mean"] = lab_stds[:, 0].mean()
        features["melnv_grid_dark_ratio_range"] = max(dark_ratios) - min(dark_ratios) if dark_ratios else 0.0
        features["melnv_grid_high_sat_ratio_range"] = max(sat_ratios) - min(sat_ratios) if sat_ratios else 0.0
    return features


def _component_stats(binary, lesion_area):
    labels, n_labels = ndi.label(binary)
    if n_labels == 0:
        return 0.0, 0.0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(n_labels), float(sizes.max() / lesion_area), float(sizes.sum() / lesion_area)


def _component_features(lab, hsv, mask):
    lesion_area = max(float(mask.sum()), 1.0)
    lab_l = lab[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    lesion_l = lab_l[mask]
    lesion_sat = sat[mask]
    dark_threshold = min(float(np.percentile(lesion_l, 35)), 105.0)
    high_sat_threshold = max(float(np.percentile(lesion_sat, 75)), 110.0)

    masks = {
        "dark": mask & (lab_l < dark_threshold),
        "high_sat": mask & (sat > high_sat_threshold),
        "dark_high_sat": mask & (lab_l < dark_threshold) & (sat > 80) & (value < 150),
        "light": mask & (lab_l > max(float(np.percentile(lesion_l, 85)), 160.0)),
    }

    features = {}
    for name, binary in masks.items():
        count, largest, total = _component_stats(binary, lesion_area)
        features[f"melnv_{name}_component_count"] = count
        features[f"melnv_{name}_largest_component_ratio"] = largest
        features[f"melnv_{name}_total_component_ratio"] = total
    return features


def extract_melnv_features(image, mask):
    mask = _safe_mask(mask)
    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    lab_pixels = _safe_pixels(lab, mask).astype(np.float32)
    hsv_pixels = _safe_pixels(hsv, mask).astype(np.float32)

    features = {}
    features.update(_lab_spread_features(lab_pixels))
    features.update(_hsv_variegation_features(hsv_pixels))
    features.update(_pca_asymmetry_features(lab, mask))
    features.update(_core_border_features(lab, hsv, mask))
    features.update(_grid_features(lab, hsv, mask))
    features.update(_component_features(lab, hsv, mask))
    return _finalize(features)
