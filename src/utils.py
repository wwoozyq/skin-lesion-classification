from pathlib import Path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_id_from_path(path):
    return Path(path).stem


def mask_name_for_image_id(image_id):
    return f"mask_{image_id}.jpg"

