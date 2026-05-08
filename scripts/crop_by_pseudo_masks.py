import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(image_dir, max_images=None):
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
    return None


def mask_bbox(mask, fallback_size, padding_ratio):
    width, height = fallback_size
    pixels = mask.convert("L").load()

    xs = []
    ys = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] > 0:
                xs.append(x)
                ys.append(y)

    if not xs:
        return 0, 0, width, height

    x0, x1 = min(xs), max(xs) + 1
    y0, y1 = min(ys), max(ys) + 1

    box_w = x1 - x0
    box_h = y1 - y0
    pad = int(max(box_w, box_h) * padding_ratio)

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    return x0, y0, x1, y1


def square_box(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    side = max(box_w, box_h)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height

    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def crop_image(image_path, mask_path, output_path, output_size, padding_ratio, square):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if mask_path is None:
            box = (0, 0, image.width, image.height)
            used_mask = False
        else:
            with Image.open(mask_path) as mask:
                box = mask_bbox(mask, image.size, padding_ratio)
            used_mask = True

        if square:
            box = square_box(box, image.size)

        cropped = image.crop(box)
        if output_size is not None:
            cropped = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)

    return {
        "image_id": image_path.stem,
        "image_file": image_path.name,
        "crop_file": output_path.name,
        "mask_file": mask_path.name if mask_path is not None else "",
        "used_mask": used_mask,
        "x0": box[0],
        "y0": box[1],
        "x1": box[2],
        "y1": box[3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_size", type=int, default=224)
    parser.add_argument("--padding_ratio", type=float, default=0.35)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--no_square", action="store_true")
    args = parser.parse_args()

    image_paths = iter_images(args.image_dir, args.max_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in image_paths:
        mask_path = find_mask(args.mask_dir, image_path.stem)
        output_path = output_dir / f"{image_path.stem}.jpg"
        row = crop_image(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            output_size=args.output_size,
            padding_ratio=args.padding_ratio,
            square=not args.no_square,
        )
        rows.append(row)
        print(f"Saved {output_path}")

    pd.DataFrame(rows).to_csv(output_dir / "crop_manifest.csv", index=False)
    print(f"Generated {len(rows)} crops.")
    print(f"Saved {output_dir / 'crop_manifest.csv'}")


if __name__ == "__main__":
    main()
