from pathlib import Path


LABELS = ["mel", "nv", "vasc"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

DEFAULT_MODEL_DIR = Path("outputs/models")
DEFAULT_METRICS_DIR = Path("outputs/metrics")

