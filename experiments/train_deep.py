import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError as exc:
    raise SystemExit(
        "Deep learning dependencies are optional. Install them with:\n"
        "  uv pip install --python .venv/bin/python -r requirements-deep.txt"
    ) from exc

from src.config import LABELS, LABEL_TO_ID
from src.dataset import image_path, load_labels, load_mask
from src.preprocess import prepare_mask
from src.utils import base_id_from_image_id, ensure_dir


class LesionDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None, crop=True, mask_mode="clean"):
        self.data_dir = Path(data_dir)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform
        self.crop = crop
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.labels)

    def _load_image(self, image_id):
        with Image.open(image_path(self.data_dir, image_id)) as image:
            image = image.convert("RGB")
        if not self.crop:
            return image

        mask = prepare_mask(load_mask(self.data_dir, image_id), mask_mode=self.mask_mode)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return image

        width, height = image.size
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width, x1 + pad)
        y1 = min(height, y1 + pad)
        return image.crop((x0, y0, x1, y1))

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image = self._load_image(str(row["image_id"]))
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_ID[row["dx"]]
        return image, label


def _transforms(image_size):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, valid_tf


def _model(name, pretrained=False):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        return model
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    raise ValueError(f"Unknown model={name}")


def _evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(labels.numpy().tolist())
    labels = list(range(len(LABELS)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=LABELS, zero_division=0),
    }


def train_deep(args):
    data_dir = Path(args.data_dir)
    labels = load_labels(data_dir)
    labels["image_id"] = labels["image_id"].astype(str)
    groups = labels["image_id"].map(base_id_from_image_id)
    split = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    train_idx, valid_idx = next(split.split(labels, labels["dx"], groups=groups))
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    valid_labels = labels.iloc[valid_idx].reset_index(drop=True)

    train_tf, valid_tf = _transforms(args.image_size)
    train_ds = LesionDataset(data_dir, train_labels, train_tf, crop=args.crop, mask_mode=args.mask_mode)
    valid_ds = LesionDataset(data_dir, valid_labels, valid_tf, crop=args.crop, mask_mode=args.mask_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _model(args.model, pretrained=args.pretrained).to(device)

    class_counts = train_labels["dx"].value_counts().reindex(LABELS).to_numpy(dtype=np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = _evaluate(model, valid_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(
            f"epoch={epoch:02d} loss={metrics['train_loss']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_acc={metrics['accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    ensure_dir(Path(args.output_dir))
    rows = [{k: v for k, v in item.items() if k != "report"} for item in history]
    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "deep_history.csv", index=False)
    (Path(args.output_dir) / "deep_classification_report.txt").write_text(history[-1]["report"])
    if best_state is not None:
        torch.save(best_state, Path(args.output_dir) / f"{args.model}_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--output_dir", default="outputs/deep/resnet18_grouped")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--mask_mode", default="clean", choices=["raw", "clean"])
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train_deep(args)


if __name__ == "__main__":
    main()
