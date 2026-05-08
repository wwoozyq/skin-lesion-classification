import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_paths(image_dir, max_images=None):
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def find_mask(mask_dir, image_id):
    candidates = [
        Path(mask_dir) / f"mask_{image_id}.jpg",
        Path(mask_dir) / f"mask_{image_id}.png",
        Path(mask_dir) / f"{image_id}.jpg",
        Path(mask_dir) / f"{image_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Mask not found for image_id={image_id}")


def make_overlay(image, mask, alpha=0.45):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L")) > 0

    overlay_color = np.array([255, 40, 40], dtype=np.float32)
    output = rgb.copy()
    output[mask_arr] = (1.0 - alpha) * output[mask_arr] + alpha * overlay_color
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(args.image_dir, args.max_images)
    for image_path in paths:
        mask_path = find_mask(args.mask_dir, image_path.stem)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            overlay = make_overlay(image, mask, alpha=args.alpha)
        output_path = output_dir / f"overlay_{image_path.stem}.jpg"
        overlay.save(output_path, quality=95)
        print(f"Saved {output_path}")

    print(f"Generated {len(paths)} overlay previews.")


if __name__ == "__main__":
    main()
