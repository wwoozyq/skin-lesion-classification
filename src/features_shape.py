import numpy as np
from scipy import ndimage as ndi


def extract_shape_features(image, mask):
    h, w = mask.shape
    area = float(mask.sum())
    image_area = float(h * w)

    if area == 0:
        return {
            "mask_area_ratio": 0.0,
            "perimeter": 0.0,
            "bbox_aspect_ratio": 0.0,
            "extent": 0.0,
            "center_y": 0.0,
            "center_x": 0.0,
        }

    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    bbox_area = float(max(bbox_h * bbox_w, 1))

    return {
        "mask_area_ratio": area / image_area,
        "perimeter": perimeter,
        "bbox_aspect_ratio": float(bbox_w / max(bbox_h, 1)),
        "extent": area / bbox_area,
        "center_y": float(ys.mean() / max(h - 1, 1)),
        "center_x": float(xs.mean() / max(w - 1, 1)),
    }
