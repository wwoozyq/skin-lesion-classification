from pathlib import Path
import re


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_id_from_path(path):
    return Path(path).stem


def mask_name_for_image_id(image_id):
    return f"mask_{image_id}.jpg"


def base_id_from_image_id(image_id):
    stem = Path(str(image_id)).stem
    patterns = [
        r"(_aug\d+)$",
        r"(_flip(?:ud|lr|x|y)?)$",
        r"(_rot(?:ate)?\d+)$",
        r"(_brightness\d*)$",
        r"(_contrast\d*)$",
        r"(_color\d*)$",
    ]
    base_id = stem
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", base_id, flags=re.IGNORECASE)
            if updated != base_id:
                base_id = updated
                changed = True
    return base_id
