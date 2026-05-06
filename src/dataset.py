from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import mask_name_for_image_id


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(data_dir):
    label_path = Path(data_dir) / "label.csv"
    if not label_path.exists():
        return None
    labels = pd.read_csv(label_path)
    labels["image_id"] = labels["image_id"].astype(str)
    return labels


def list_image_ids(data_dir):
    image_dir = Path(data_dir) / "image"
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )
    return [p.stem for p in image_paths]


def image_path(data_dir, image_id):
    image_dir = Path(data_dir) / "image"
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for image_id={image_id}")


def mask_path(data_dir, image_id):
    mask_dir = Path(data_dir) / "mask"
    candidates = [
        mask_dir / mask_name_for_image_id(image_id),
        mask_dir / f"mask_{image_id}.png",
        mask_dir / f"{image_id}.jpg",
        mask_dir / f"{image_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Mask not found for image_id={image_id}")


def load_image(data_dir, image_id):
    path = image_path(data_dir, image_id)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(data_dir, image_id):
    path = mask_path(data_dir, image_id)
    with Image.open(path) as mask:
        return np.asarray(mask.convert("L")) > 0
