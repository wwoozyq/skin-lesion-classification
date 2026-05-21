"""Quadrant-based color asymmetry features (ABCD 'A' rule, fine-grained).

Splits the lesion mask into 4 quadrants by its centroid, computes LAB color
statistics per quadrant plus cross-quadrant asymmetry measures. Provides a
more granular asymmetry signal than the binary L/R + U/D asymmetry already
in features_abcd.py.

References
----------
- Stolz et al. (1994) "ABCD rule of dermatoscopy: a new practical method for
  early recognition of malignant melanoma" Eur. J. Dermatol. 4(7):521-527.
- Celebi et al. (2007) "A methodological approach to the classification of
  dermoscopy images" Computerized Medical Imaging and Graphics 31(6):362-373.
  https://doi.org/10.1016/j.compmedimag.2007.01.003
- Mendonca et al. (2013) "PH2 - A dermoscopic image database for research and
  benchmarking" IEEE EMBC 2013, pp. 5437-5440.
  https://doi.org/10.1109/EMBC.2013.6610779
- Argenziano et al. (2003) "Dermoscopy of pigmented skin lesions: results of
  a consensus meeting via the Internet" J. Am. Acad. Dermatol. 48(5):679-693.
"""

import numpy as np
from PIL import Image


_EMPTY_KEYS = (
    [f"subregion_q{i}_l_mean" for i in range(4)]
    + [f"subregion_q{i}_a_mean" for i in range(4)]
    + [f"subregion_q{i}_b_mean" for i in range(4)]
    + [f"subregion_q{i}_area_ratio" for i in range(4)]
    + [
        "subregion_l_asymmetry",
        "subregion_a_asymmetry",
        "subregion_b_asymmetry",
        "subregion_area_asymmetry",
        "subregion_lab_centroid_max_dist",
    ]
)


def _empty():
    return {k: 0.0 for k in _EMPTY_KEYS}


def extract_subregion_features(image, mask):
    if not mask.any():
        return _empty()

    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    ys, xs = np.where(mask)
    cy, cx = float(ys.mean()), float(xs.mean())

    # 4 个象限相对病灶质心
    quadrants = [
        (ys < cy) & (xs < cx),
        (ys < cy) & (xs >= cx),
        (ys >= cy) & (xs < cx),
        (ys >= cy) & (xs >= cx),
    ]

    features = {}
    lab_means = []
    areas = []
    total_area = float(len(ys))

    for i, q in enumerate(quadrants):
        q_ys, q_xs = ys[q], xs[q]
        if len(q_ys) == 0:
            features[f"subregion_q{i}_l_mean"] = 0.0
            features[f"subregion_q{i}_a_mean"] = 0.0
            features[f"subregion_q{i}_b_mean"] = 0.0
            features[f"subregion_q{i}_area_ratio"] = 0.0
            lab_means.append(np.zeros(3, dtype=np.float32))
            areas.append(0.0)
            continue
        q_lab = lab[q_ys, q_xs].astype(np.float32)
        mean = q_lab.mean(axis=0)
        features[f"subregion_q{i}_l_mean"] = float(mean[0])
        features[f"subregion_q{i}_a_mean"] = float(mean[1])
        features[f"subregion_q{i}_b_mean"] = float(mean[2])
        features[f"subregion_q{i}_area_ratio"] = float(len(q_ys) / total_area)
        lab_means.append(mean)
        areas.append(float(len(q_ys) / total_area))

    lab_means = np.asarray(lab_means)
    areas = np.asarray(areas, dtype=np.float32)

    # 4 象限上各 LAB 通道的极差 = "颜色不对称度"
    features["subregion_l_asymmetry"] = float(lab_means[:, 0].max() - lab_means[:, 0].min())
    features["subregion_a_asymmetry"] = float(lab_means[:, 1].max() - lab_means[:, 1].min())
    features["subregion_b_asymmetry"] = float(lab_means[:, 2].max() - lab_means[:, 2].min())
    features["subregion_area_asymmetry"] = float(areas.max() - areas.min())

    # 4 象限 LAB 均值两两距离的最大值, 反映"最远色对"
    max_dist = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(lab_means[i] - lab_means[j]))
            if d > max_dist:
                max_dist = d
    features["subregion_lab_centroid_max_dist"] = max_dist / 255.0

    return features
