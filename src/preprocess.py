import numpy as np
from scipy import ndimage as ndi


def clean_mask(mask, min_area_ratio=0.001):
    """Denoise a binary lesion mask while preserving the main lesion region."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask

    filled = ndi.binary_fill_holes(mask)
    labeled, num_labels = ndi.label(filled)
    if num_labels == 0:
        return mask

    component_sizes = np.bincount(labeled.reshape(-1))
    component_sizes[0] = 0
    image_area = mask.size
    min_area = max(int(image_area * min_area_ratio), 8)
    keep_labels = np.where(component_sizes >= min_area)[0]
    if len(keep_labels) == 0:
        keep_labels = [int(component_sizes.argmax())]

    cleaned = np.isin(labeled, keep_labels)
    if not cleaned.any():
        cleaned = labeled == component_sizes.argmax()

    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndi.binary_opening(cleaned, structure=structure)
    cleaned = ndi.binary_closing(cleaned, structure=structure)
    cleaned = ndi.binary_fill_holes(cleaned)

    if not cleaned.any():
        return mask
    return cleaned.astype(bool)


def prepare_mask(mask, mask_mode="raw"):
    if mask_mode == "raw":
        return mask.astype(bool)
    if mask_mode == "clean":
        return clean_mask(mask)
    raise ValueError(f"Unknown mask_mode={mask_mode}")
