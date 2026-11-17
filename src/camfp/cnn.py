"""Deep learning approach: a CNN that learns camera traces from small image patches.

The first layer is a *constrained convolution* (Bayar & Stamm, 2018). Each filter is forced
to predict a pixel from its neighbours and output the prediction error: the centre weight
is fixed at -1 and the others sum to +1. That suppresses image content (what is in the
photo) and passes on the subtle processing traces left by the camera, which the rest of the
network then learns to classify.

Training uses random 64x64 patches; at test time, predictions from a grid of patches across
the photo are averaged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConstrainedConv2d(nn.Conv2d):
    def constrain(self) -> None:
        """Project weights so centre = -1 and the rest sum to 1 (call after each step)."""
        with torch.no_grad():
            w = self.weight
            c = w.shape[-1] // 2
            w[:, :, c, c] = 0
            w /= w.sum(dim=(2, 3), keepdim=True) + 1e-8
            w[:, :, c, c] = -1


class CameraNet(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()
        self.constrained = ConstrainedConv2d(in_channels, 6, kernel_size=5, padding=2, bias=False)
        self.constrained.constrain()

        def block(cin, cout, k, stride=1):
            return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(6, 64, 7, 2), nn.MaxPool2d(3, 2, 1),
            block(64, 64, 5), nn.MaxPool2d(3, 2, 1),
            block(64, 128, 5), nn.MaxPool2d(3, 2, 1),
            block(128, 128, 1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.constrained(x)))


def to_tensor(patches: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) scaled to roughly [-1, 1]."""
    return torch.from_numpy(patches).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def random_patches(img: np.ndarray, n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    ys = rng.integers(0, (h - size) // 2 + 1, n) * 2  # even offsets keep the colour-filter phase
    xs = rng.integers(0, (w - size) // 2 + 1, n) * 2
    return np.stack([img[y : y + size, x : x + size] for y, x in zip(ys, xs)])


def grid_patches(img: np.ndarray, size: int, stride: int | None = None) -> np.ndarray:
    stride = stride or size
    h, w = img.shape[:2]
    return np.stack([img[y : y + size, x : x + size]
                     for y in range(0, h - size + 1, stride)
                     for x in range(0, w - size + 1, stride)])


def train(images: list[np.ndarray], labels: list[int], n_classes: int,
          val_images: list[np.ndarray] | None = None, val_labels: list[int] | None = None,
          epochs: int = 20, patches_per_image: int = 32, patch: int = 64, batch: int = 128,
          lr: float = 1e-3, seed: int = 0, log=print) -> CameraNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    model = CameraNet(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    log(f"Training on {device} | {len(images)} images | {n_classes} classes")

    for epoch in range(1, epochs + 1):
        model.train()
        x = np.concatenate([random_patches(im, patches_per_image, patch, rng) for im in images])
        y = np.repeat(labels, patches_per_image)
        order = rng.permutation(len(x))
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            xb, yb = to_tensor(x[idx]).to(device), torch.from_numpy(y[idx]).long().to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            model.constrained.constrain()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(idx)
        sched.step()
        msg = f"epoch {epoch:3d}  loss {loss_sum / total:.3f}  patch acc {correct / total:.3f}"
        if val_images:
            preds = [predict(model, im, patch).argmax() for im in val_images]
            msg += f"  val image acc {np.mean(np.array(preds) == np.array(val_labels)):.3f}"
        log(msg)
    return model


@torch.no_grad()
def predict(model: CameraNet, img: np.ndarray, patch: int = 64) -> np.ndarray:
    """Average class probabilities over a grid of patches covering the image."""
    model.eval()
    device = next(model.parameters()).device
    probs = torch.softmax(model(to_tensor(grid_patches(img, patch)).to(device)), dim=1)
    return probs.mean(0).cpu().numpy()


def save(model: CameraNet, classes: list[str], folder: str | Path, patch: int) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "classes.json").write_text(json.dumps({"classes": classes, "patch": patch}))


def load(folder: str | Path) -> tuple[CameraNet, list[str], int]:
    folder = Path(folder)
    meta = json.loads((folder / "classes.json").read_text())
    model = CameraNet(len(meta["classes"]))
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    return model.to(pick_device()), meta["classes"], meta["patch"]
