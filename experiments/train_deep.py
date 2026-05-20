import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/deep/torch_cache"))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
    def __init__(
        self,
        data_dir,
        labels,
        transform=None,
        crop=True,
        mask_mode="clean",
        crop_pad=0.25,
        cache_images=False,
    ):
        self.data_dir = Path(data_dir)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform
        self.crop = crop
        self.mask_mode = mask_mode
        self.crop_pad = crop_pad
        self.cache_images = cache_images
        self._image_cache = {}

    def __len__(self):
        return len(self.labels)

    def _load_image(self, image_id):
        if self.cache_images and image_id in self._image_cache:
            return self._image_cache[image_id].copy()

        with Image.open(image_path(self.data_dir, image_id)) as image:
            image = image.convert("RGB")
        if self.crop:
            image = self._crop_by_mask(image, image_id)

        if self.cache_images:
            self._image_cache[image_id] = image.copy()
        return image

    def _crop_by_mask(self, image, image_id):
        mask = prepare_mask(load_mask(self.data_dir, image_id), mask_mode=self.mask_mode)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return image

        width, height = image.size
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pad = int(self.crop_pad * max(x1 - x0, y1 - y0))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width, x1 + pad)
        y1 = min(height, y1 + pad)
        return image.crop((x0, y0, x1, y1))

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image_id = str(row["image_id"])
        image = self._load_image(image_id)
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_ID[row["dx"]]
        return image, label


def _transforms(image_size, augment_strength="standard"):
    if augment_strength == "light":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
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


def _head_parameters(model, name):
    if name == "resnet18":
        return model.fc.parameters()
    if name == "mobilenet_v2":
        return model.classifier.parameters()
    raise ValueError(f"Unknown model={name}")


def _set_backbone_trainable(model, name, trainable):
    for param in model.parameters():
        param.requires_grad = trainable
    for param in _head_parameters(model, name):
        param.requires_grad = True


def _make_optimizer(model, lr, weight_decay):
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _device(preferred="auto"):
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device=mps, but MPS is not available.")
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device=cuda, but CUDA is not available.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_logits(model, images, eval_tta=False):
    if not eval_tta:
        return model(images)

    logits = [
        model(images),
        model(torch.flip(images, dims=[3])),
        model(torch.flip(images, dims=[2])),
        model(torch.flip(images, dims=[2, 3])),
    ]
    return torch.stack(logits, dim=0).mean(dim=0)


def _evaluate(model, loader, device, eval_tta=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = _forward_logits(model, images, eval_tta=eval_tta)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(labels.numpy().tolist())
    label_ids = list(range(len(LABELS)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids),
        "report": classification_report(y_true, y_pred, labels=label_ids, target_names=LABELS, zero_division=0),
    }


def _class_weights(labels, device):
    class_counts = labels["dx"].value_counts().reindex(LABELS).to_numpy(dtype=np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _sampler(labels):
    class_counts = labels["dx"].value_counts().to_dict()
    sample_weights = labels["dx"].map(lambda label: 1.0 / class_counts[label]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _metric(metrics, name):
    if name == "accuracy":
        return metrics["accuracy"]
    if name == "macro_f1":
        return metrics["macro_f1"]
    if name == "balanced_accuracy":
        return metrics["balanced_accuracy"]
    raise ValueError(f"Unknown monitor={name}")


def train_deep(args):
    _set_seed(args.random_state)
    data_dir = Path(args.data_dir)
    labels = load_labels(data_dir)
    labels["image_id"] = labels["image_id"].astype(str)
    groups = labels["image_id"].map(base_id_from_image_id)
    split = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    train_idx, valid_idx = next(split.split(labels, labels["dx"], groups=groups))
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    valid_labels = labels.iloc[valid_idx].reset_index(drop=True)

    train_tf, valid_tf = _transforms(args.image_size, augment_strength=args.augment_strength)
    train_ds = LesionDataset(
        data_dir,
        train_labels,
        train_tf,
        crop=args.crop,
        mask_mode=args.mask_mode,
        crop_pad=args.crop_pad,
        cache_images=args.cache_images,
    )
    valid_ds = LesionDataset(
        data_dir,
        valid_labels,
        valid_tf,
        crop=args.crop,
        mask_mode=args.mask_mode,
        crop_pad=args.crop_pad,
        cache_images=args.cache_images,
    )
    sampler = _sampler(train_labels) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = _device(args.device)
    model = _model(args.model, pretrained=args.pretrained).to(device)

    if args.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, args.model, trainable=False)

    criterion_weight = None if args.no_class_weights else _class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight, label_smoothing=args.label_smoothing)
    optimizer = _make_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score = -1.0
    best_state = None
    best_metrics = None
    history = []
    epochs_without_improvement = 0

    print(f"device={device} pretrained={args.pretrained} model={args.model} eval_tta={args.eval_tta}")
    print(f"train={len(train_labels)} valid={len(valid_labels)} monitor={args.monitor}")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            _set_backbone_trainable(model, args.model, trainable=True)
            optimizer = _make_optimizer(model, args.finetune_lr, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - args.freeze_backbone_epochs, 1),
            )
            print(f"epoch={epoch:02d} unfreezing backbone lr={args.finetune_lr:g}")

        model.train()
        losses = []
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        metrics = _evaluate(model, valid_loader, device, eval_tta=args.eval_tta)
        score = _metric(metrics, args.monitor)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "phase": "frozen" if epoch <= args.freeze_backbone_epochs else "finetune",
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} phase={row['phase']:8s} loss={row['train_loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_bal_acc={metrics['balanced_accuracy']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch:02d} best_{args.monitor}={best_score:.4f}")
            break

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    pd.DataFrame(history).to_csv(output_dir / "deep_history.csv", index=False)
    if best_state is not None:
        torch.save(best_state, output_dir / f"{args.model}_best.pt")
    if best_metrics is not None:
        (output_dir / "deep_best_classification_report.txt").write_text(best_metrics["report"])
        pd.DataFrame(
            best_metrics["confusion_matrix"],
            index=LABELS,
            columns=LABELS,
        ).to_csv(output_dir / "deep_best_confusion_matrix.csv")
        pd.DataFrame([{
            "model": args.model,
            "pretrained": args.pretrained,
            "monitor": args.monitor,
            "best_score": best_score,
            "accuracy": best_metrics["accuracy"],
            "macro_f1": best_metrics["macro_f1"],
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "random_state": args.random_state,
            "device": str(device),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "finetune_lr": args.finetune_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "patience": args.patience,
            "crop": args.crop,
            "crop_pad": args.crop_pad,
            "mask_mode": args.mask_mode,
            "augment_strength": args.augment_strength,
            "eval_tta": args.eval_tta,
            "balanced_sampler": args.balanced_sampler,
            "class_weights": not args.no_class_weights,
        }]).to_csv(output_dir / "deep_best_metrics.csv", index=False)
        print(
            f"best {args.monitor}={best_score:.4f} "
            f"acc={best_metrics['accuracy']:.4f} macro_f1={best_metrics['macro_f1']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", default="mobilenet_v2", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--output_dir", default="outputs/deep/mobilenet_v2_grouped")
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--mask_mode", default="clean", choices=["raw", "clean"])
    parser.add_argument("--crop_pad", type=float, default=0.25)
    parser.add_argument("--augment_strength", default="standard", choices=["light", "standard"])
    parser.add_argument("--monitor", default="accuracy", choices=["accuracy", "macro_f1", "balanced_accuracy"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--cache_images", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--eval_tta", action="store_true")
    args = parser.parse_args()
    train_deep(args)


if __name__ == "__main__":
    main()
