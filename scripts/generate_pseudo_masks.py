import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(image_dir, max_images=None):
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def otsu_threshold(values, bins=256):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return float(values.mean())
    return float(centers[:-1][np.argmax(variance12)])


def border_region(shape, width_ratio=0.08):
    height, width = shape
    border = np.zeros(shape, dtype=bool)
    border_width = max(4, int(min(height, width) * width_ratio))
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    return border


def largest_component(mask):
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask

    sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


def center_weight(shape):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dist = ((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2
    return np.exp(-dist * 5.0)


def remove_dark_frame(rgb):
    intensity = rgb.mean(axis=2)
    return intensity > 8


def generate_mask(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    valid = remove_dark_frame(rgb)

    lab = np.asarray(image.convert("LAB"), dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)

    border = border_region((height, width)) & valid
    if border.sum() < 10:
        border = valid

    bg_lab = np.median(lab[border], axis=0)
    lab_distance = np.linalg.norm(lab - bg_lab, axis=2)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    bg_value = np.median(value[border])
    bg_saturation = np.median(saturation[border])

    darker_than_skin = np.maximum(bg_value - value, 0.0) / 255.0
    saturation_delta = np.abs(saturation - bg_saturation) / 255.0
    color_distance = lab_distance / max(np.percentile(lab_distance[valid], 95), 1.0)

    score = 0.55 * color_distance + 0.30 * darker_than_skin + 0.15 * saturation_delta
    score = ndi.gaussian_filter(score, sigma=max(1.0, min(height, width) / 180.0))
    score = score * valid * center_weight((height, width))

    threshold = otsu_threshold(score[valid])
    mask = (score > threshold) & valid

    min_size = max(32, int(height * width * 0.002))
    labeled, count = ndi.label(mask)
    if count > 0:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, count + 1))
        keep = np.zeros_like(mask, dtype=bool)
        for idx, size in enumerate(sizes, start=1):
            if size >= min_size:
                keep |= labeled == idx
        mask = keep

    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)

    structure_size = max(2, int(min(height, width) * 0.01))
    structure = np.ones((structure_size, structure_size), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure)
    mask = ndi.binary_closing(mask, structure=structure)
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask)

    if mask.sum() < min_size:
        fallback_threshold = np.percentile(score[valid], 85)
        mask = (score > fallback_threshold) & valid
        mask = largest_component(ndi.binary_fill_holes(mask))

    return mask


def save_mask(mask, output_path):
    output = (mask.astype(np.uint8) * 255)
    Image.fromarray(output, mode="L").save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--prefix", default="mask_")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(image_dir, args.max_images)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            mask = generate_mask(image)
        output_path = output_dir / f"{args.prefix}{image_path.stem}.jpg"
        save_mask(mask, output_path)
        print(f"Saved {output_path}")

    print(f"Generated {len(image_paths)} pseudo masks.")


if __name__ == "__main__":
    main()
