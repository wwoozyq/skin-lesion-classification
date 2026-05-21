"""Gabor filter response statistics across orientations and frequencies.

Gabor filters approximate receptive fields of mammalian visual cortex and are
classical tools for capturing oriented texture patterns (e.g. pigment network,
streaks, dots in dermoscopy). The orientation diversity statistic helps
detect 'irregular' streaks more characteristic of melanoma than nevi.

References
----------
- Daugman (1985) "Uncertainty relation for resolution in space, spatial
  frequency, and orientation optimized by two-dimensional visual cortical
  filters" J. Opt. Soc. Am. A 2(7):1160-1169.
  https://doi.org/10.1364/JOSAA.2.001160
- Sadeghi et al. (2013) "Detection and Analysis of Irregular Streaks in
  Dermoscopic Images of Skin Lesions" IEEE Trans. Med. Imaging 32(5):849-861.
  https://doi.org/10.1109/TMI.2013.2239307
- Iyatomi et al. (2008) "An improved internet-based melanoma screening system
  with dermatologist-like tumor area extraction algorithm" Computerized
  Medical Imaging and Graphics 32(7):566-579.
  https://doi.org/10.1016/j.compmedimag.2008.06.005
"""

import numpy as np
from PIL import Image
from skimage.filters import gabor


def _safe_pixels(arr, mask):
    if mask.any():
        return arr[mask]
    return arr.reshape(-1)


def extract_gabor_features(image, mask):
    gray = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32) / 255.0
    features = {}

    # 4 方向 x 3 频率 = 12 个 Gabor 响应
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    frequencies = [0.10, 0.25, 0.40]   # 低 / 中 / 高频纹理

    for freq in frequencies:
        per_orient_means = []
        for theta in orientations:
            real, imag = gabor(gray, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            mag_in = _safe_pixels(magnitude, mask)
            mean_val = float(mag_in.mean())
            std_val = float(mag_in.std())
            tag = f"gabor_f{int(freq * 100):02d}_t{int(np.degrees(theta)):03d}"
            features[f"{tag}_mean"] = mean_val
            features[f"{tag}_std"] = std_val
            per_orient_means.append(mean_val)
        # 同频率不同方向响应均值的标准差 = "各向异性强度"
        # 值越大说明该频率下纹理越有方向性 (典型 mel: 不规则放射状)
        per_orient_means = np.asarray(per_orient_means, dtype=np.float32)
        features[f"gabor_f{int(freq * 100):02d}_orient_std"] = float(per_orient_means.std())
        features[f"gabor_f{int(freq * 100):02d}_orient_range"] = float(
            per_orient_means.max() - per_orient_means.min()
        )
    return features
