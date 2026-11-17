"""Index image datasets into one table and split them without leakage.

Every image becomes a row with: path, dataset, brand, model, device, platform, kind, source.

* platform -- "native" (straight from the camera) or a social media version such as
  "whatsapp", "facebook_high", "facebook_low"
* kind     -- "flat" (plain surfaces such as sky or walls, ideal for estimating fingerprints)
  or "natural" (normal photos)
* source   -- identifies the original shot. Social media copies share the source of their
  original, so a photo and its WhatsApp copy always land in the same split. Without this,
  the test set would contain near-copies of training images and results would be inflated.

Supported layouts
-----------------
VISION (Shullani et al., 2017):  <root>/D01_Samsung_GalaxyS3Mini/images/{Flat,Nat,NatWA,NatFBH,NatFBL}/*.jpg
Dresden (Gloe & Bohme, 2010):    <root>/Canon_Ixus70_0_3554.JPG   (brand_model_instance_id)
Your own photos:                 <root>/<brand>_<model>_<owner>/*.jpg   e.g. Apple_iPhone13_josef
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

VISION_PLATFORMS = {
    "flat": ("native", "flat"),
    "nat": ("native", "natural"),
    "natwa": ("whatsapp", "natural"),
    "natfbh": ("facebook_high", "natural"),
    "natfbl": ("facebook_low", "natural"),
}

COLUMNS = ["path", "dataset", "brand", "model", "device", "platform", "kind", "source"]


def _images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _trailing_number(stem: str) -> str:
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def index_vision(root: str | Path) -> pd.DataFrame:
    rows = []
    for device_dir in sorted(Path(root).glob("D[0-9][0-9]_*")):
        parts = device_dir.name.split("_")
        device, brand, model = parts[0], parts[1], "_".join(parts[2:])
        images_dir = device_dir / "images"
        if not images_dir.exists():
            continue
        for sub in images_dir.iterdir():
            key = sub.name.lower()
            if key not in VISION_PLATFORMS:
                continue
            platform, kind = VISION_PLATFORMS[key]
            for p in _images(sub):
                source = f"vision_{device}_{kind}_{_trailing_number(p.stem)}"
                rows.append([str(p), "vision", brand, f"{brand}_{model}", f"vision_{device}",
                             platform, kind, source])
    return pd.DataFrame(rows, columns=COLUMNS)


def index_dresden(root: str | Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^(?P<brand>[^_]+)_(?P<model>.+)_(?P<inst>\d+)_(?P<id>\d+)$")
    for p in _images(Path(root)):
        m = pattern.match(p.stem)
        if not m:
            continue
        brand, model, inst = m["brand"], m["model"], m["inst"]
        kind = "flat" if "flat" in str(p.parent).lower() else "natural"
        rows.append([str(p), "dresden", brand, f"{brand}_{model}", f"dresden_{brand}_{model}_{inst}",
                     "native", kind, f"dresden_{p.stem}"])
    return pd.DataFrame(rows, columns=COLUMNS)


def index_own(root: str | Path) -> pd.DataFrame:
    """Folders named <brand>_<model>_<owner>, each containing photos from one device."""
    rows = []
    for device_dir in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        parts = device_dir.name.split("_")
        if len(parts) < 3:
            print(f"Skipping {device_dir.name}: name it <brand>_<model>_<owner>")
            continue
        brand, model = parts[0], "_".join(parts[1:-1])
        for p in _images(device_dir):
            kind = "flat" if "flat" in p.parent.name.lower() else "natural"
            rows.append([str(p), "own", brand, f"{brand}_{model}", f"own_{device_dir.name}",
                         "native", kind, f"own_{device_dir.name}_{p.stem}"])
    return pd.DataFrame(rows, columns=COLUMNS)


def build_index(vision: str | None = None, dresden: str | None = None,
                own: str | None = None) -> pd.DataFrame:
    frames = []
    if vision:
        frames.append(index_vision(vision))
    if dresden:
        frames.append(index_dresden(dresden))
    if own:
        frames.append(index_own(own))
    if not frames:
        raise ValueError("Give at least one dataset folder")
    return pd.concat(frames, ignore_index=True)


def split_index(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Assign train/test per device, grouped by source so copies never cross the split.

    Flat images always go to train: they are only used to build fingerprints.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"
    natural = df[df["kind"] == "natural"]
    for _, group in natural.groupby("device"):
        sources = rng.permutation(np.asarray(group["source"].unique(), dtype=object))
        n_test = max(1, round(len(sources) * test_frac)) if len(sources) > 1 else 0
        test_sources = set(sources[:n_test])
        df.loc[group.index[group["source"].isin(test_sources)], "split"] = "test"
    return df


def describe(df: pd.DataFrame) -> str:
    lines = [(f"{len(df)} images, {df['brand'].nunique()} brands, "
              f"{df['model'].nunique()} models, {df['device'].nunique()} devices")]
    counts = df.groupby(["platform", "kind"]).size()
    lines += [f"  {plat:<14} {kind:<8} {n}" for (plat, kind), n in counts.items()]
    if "split" in df:
        lines.append("  " + ", ".join(f"{k}: {v}" for k, v in df["split"].value_counts().items()))
    return "\n".join(lines)
