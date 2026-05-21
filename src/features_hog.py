import numpy as np
from PIL import Image
from skimage.feature import hog


def _safe_mask(mask):
    mask = np.asarray(mask).astype(bool)
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _bbox(mask, pad_ratio=0.15):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    width = x1 - x0
    height = y1 - y0
    pad = int(max(width, height) * pad_ratio)
    return (
        max(x0 - pad, 0),
        max(y0 - pad, 0),
        min(x1 + pad, mask.shape[1]),
        min(y1 + pad, mask.shape[0]),
    )


def _resize_array(arr, size, resample):
    return np.asarray(Image.fromarray(arr).resize((size, size), resample=resample))


def _hog_vector(arr):
    return hog(
        arr.astype(np.float32),
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_hog_features(image, mask, size=64):
    mask = _safe_mask(mask)
    gray = np.asarray(Image.fromarray(image).convert("L"))
    x0, y0, x1, y1 = _bbox(mask)
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255

    gray_small = _resize_array(gray_crop, size, Image.BILINEAR).astype(np.float32)
    mask_small = _resize_array(mask_crop, size, Image.NEAREST) > 0
    if mask_small.any():
        fill_value = float(np.median(gray_small[mask_small]))
    else:
        fill_value = float(np.median(gray_small))

    focused_gray = gray_small.copy()
    focused_gray[~mask_small] = fill_value
    focused_gray = focused_gray / 255.0
    mask_float = mask_small.astype(np.float32)

    gray_hog = _hog_vector(focused_gray)
    mask_hog = _hog_vector(mask_float)

    features = {}
    for idx, value in enumerate(gray_hog):
        features[f"hog_gray_{idx:03d}"] = float(value)
    for idx, value in enumerate(mask_hog):
        features[f"hog_mask_{idx:03d}"] = float(value)
    return features
