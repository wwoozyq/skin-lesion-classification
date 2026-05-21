"""Multi-scale Local Binary Pattern features (R=1, R=3).

Complements features_texture.py which only computes R=2 LBP.
Multi-scale LBP is one of the most consistently effective texture descriptors
for melanoma vs nevus discrimination in dermoscopy.

References
----------
- Ojala et al. (2002) "Multiresolution gray-scale and rotation invariant
  texture classification with local binary patterns" IEEE TPAMI 24(7):971-987.
  https://doi.org/10.1109/TPAMI.2002.1017623
- Barata et al. (2014) "Two Systems for the Detection of Melanomas in
  Dermoscopy Images Using Texture and Color Features" IEEE Systems Journal
  8(3):965-979. https://doi.org/10.1109/JSYST.2013.2271540
- Barata et al. (2019) "A Survey of Feature Extraction in Dermoscopy Image
  Analysis of Skin Cancer" IEEE J. Biomed. Health Inform. 23(3):1096-1109.
  https://doi.org/10.1109/JBHI.2018.2845939
"""

import numpy as np
from PIL import Image
from skimage import feature as skfeature


def _safe_mask_pixels(arr, mask):
    if mask.any():
        return arr[mask]
    return arr.reshape(-1)


def extract_lbp_multi_features(image, mask):
    gray = np.asarray(Image.fromarray(image).convert("L"))
    features = {}

    # 多尺度: R=1 (微纹理), R=3 (中尺度纹理)
    # 与 features_texture.py 中已有的 R=2 互补
    for radius in (1, 3):
        n_points = 8 * radius
        lbp = skfeature.local_binary_pattern(
            gray, n_points, radius, method="uniform"
        )
        lbp_pixels = _safe_mask_pixels(lbp, mask)
        n_bins = int(n_points + 2)
        hist, _ = np.histogram(
            lbp_pixels, bins=n_bins, range=(0, n_bins), density=True
        )
        for idx, value in enumerate(hist):
            features[f"lbp_r{radius}_h{idx}"] = float(value)
        features[f"lbp_r{radius}_entropy"] = _entropy(hist)
    return features


def _entropy(prob_hist):
    prob = prob_hist[prob_hist > 0]
    if prob.size == 0:
        return 0.0
    return float(-(prob * np.log2(prob)).sum())
