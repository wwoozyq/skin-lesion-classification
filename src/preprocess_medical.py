"""Medical-image preprocessing for dermatoscopy.

Three transforms applied to RGB images before feature extraction:

- ``shades_of_gray``: Finlayson & Trezzi (2004) Minkowski-norm color constancy.
- ``clahe_lab_l``: CLAHE (Zuiderveld 1994) on the CIE Lab L* channel only.
- ``hb_melanin``: Tsumura-style hemoglobin/melanin decomposition reconstruction.

All transforms take an ``(H, W, 3)`` uint8 RGB array and return the same shape.
They are pure functions with no global state; safe to call from feature
extraction loops.
"""

import warnings

import numpy as np
from skimage import color, exposure


PREPROCESSING_NONE = "none"
PREPROCESSING_SHADES = "shades_of_gray"
PREPROCESSING_CLAHE = "clahe_lab_l"
PREPROCESSING_HBMEL = "hb_melanin"
PREPROCESSING_SHADES_CLAHE = "shades_of_gray+clahe_lab_l"

SUPPORTED_PREPROCESSING = {
    PREPROCESSING_NONE,
    PREPROCESSING_SHADES,
    PREPROCESSING_CLAHE,
    PREPROCESSING_HBMEL,
    PREPROCESSING_SHADES_CLAHE,
}


def shades_of_gray(rgb, p=6, eps=1e-6):
    """Finlayson & Trezzi (2004) shades-of-gray color constancy.

    Estimates a per-channel illuminant via Minkowski-p mean and rescales
    channels so the illuminant becomes neutral. ``p=6`` is the canonical
    recommendation from the paper (best generic performance on real images).
    """
    image = np.asarray(rgb, dtype=np.float64)
    flat = image.reshape(-1, 3)
    illum = np.power(np.mean(np.power(np.clip(flat, 0, None), p), axis=0), 1.0 / p)
    illum = np.maximum(illum, eps)
    illum_unit = illum / np.linalg.norm(illum)
    gain = (1.0 / np.sqrt(3.0)) / illum_unit
    corrected = image * gain
    return np.clip(corrected, 0, 255).astype(np.uint8)


def clahe_lab_l(rgb, clip_limit=0.01, kernel_size=None, nbins=256):
    """CLAHE on the CIE Lab L* channel; keeps a* and b* untouched.

    Defaults match ``skimage.exposure.equalize_adapthist`` (clip_limit 0.01,
    kernel_size None → 1/8 of each image dim, nbins 256). Operating on L*
    preserves chromaticity so color-based features see the same channels.
    """
    rgb_f = np.asarray(rgb, dtype=np.float64) / 255.0
    lab = color.rgb2lab(rgb_f)
    l_channel = lab[..., 0] / 100.0  # L* is in [0, 100]; equalize_adapthist needs [0, 1]
    l_clahe = exposure.equalize_adapthist(
        l_channel,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
    )
    lab[..., 0] = np.clip(l_clahe, 0.0, 1.0) * 100.0
    with warnings.catch_warnings():
        # lab2rgb routinely clips a handful of out-of-gamut pixels; not informative for our use.
        warnings.simplefilter("ignore", category=UserWarning)
        rgb_out = color.lab2rgb(lab)
    return np.clip(rgb_out * 255.0, 0, 255).astype(np.uint8)


# Tsumura-style chromophore directions in optical-density (OD) space.
# Coefficients are approximate; values fall within the spread published
# across dermoscopy reproductions of Tsumura (1999/2003).
_HEMOGLOBIN_OD = np.array([0.55, 0.71, 0.43], dtype=np.float64)
_MELANIN_OD = np.array([0.40, 0.59, 0.70], dtype=np.float64)


def hb_melanin(rgb, eps=1.0 / 255.0):
    """Tsumura hemoglobin/melanin decomposition + neutral reconstruction.

    Projects each pixel's optical density onto the (hb, mel) basis, then
    reconstructs the RGB image using only those two chromophore components.
    Serves as a noise-suppressing preprocessor that strips channel content
    unrelated to skin biology.
    """
    image = np.ascontiguousarray(np.asarray(rgb, dtype=np.float64) / 255.0)
    image = np.clip(image, eps, 1.0)
    optical_density = -np.log(image)

    chromophore_basis = np.stack([_HEMOGLOBIN_OD, _MELANIN_OD], axis=1)  # (3, 2)
    basis_pinv = np.linalg.pinv(chromophore_basis)  # (2, 3)

    od_flat = np.ascontiguousarray(optical_density.reshape(-1, 3).T)  # (3, N)
    # BLAS matmul can raise spurious divide/overflow RuntimeWarnings on some
    # numpy versions even when inputs and outputs are finite; suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chromophore_amounts = basis_pinv @ od_flat  # (2, N)
        chromophore_amounts = np.clip(chromophore_amounts, 0.0, 50.0)  # negative or extreme amounts are unphysical
        reconstructed_od = (chromophore_basis @ chromophore_amounts).T  # (N, 3)

    reconstructed_image = np.exp(-reconstructed_od).reshape(image.shape)
    return np.clip(reconstructed_image * 255.0, 0, 255).astype(np.uint8)


def apply_medical_preprocessing(rgb, name):
    """Dispatch ``rgb`` through the named preprocessing.

    ``name`` ∈ :data:`SUPPORTED_PREPROCESSING`. ``"none"`` is a passthrough.
    The combined ``shades_of_gray+clahe_lab_l`` runs shades-of-gray first
    (lighting correction) then CLAHE (local contrast), which is the order
    the dermoscopy literature uses when stacking them.
    """
    if name == PREPROCESSING_NONE:
        return rgb
    if name == PREPROCESSING_SHADES:
        return shades_of_gray(rgb)
    if name == PREPROCESSING_CLAHE:
        return clahe_lab_l(rgb)
    if name == PREPROCESSING_HBMEL:
        return hb_melanin(rgb)
    if name == PREPROCESSING_SHADES_CLAHE:
        return clahe_lab_l(shades_of_gray(rgb))
    raise ValueError(
        f"Unknown preprocessing '{name}'. Supported: {sorted(SUPPORTED_PREPROCESSING)}"
    )
