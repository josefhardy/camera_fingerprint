"""Experiments. Each writes a per-image CSV, a metrics JSON and plots to an output folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import transforms as T
from .defence import denoise_defence, image_quality, subtract_fingerprint
from .images import load_rgb, to_gray
from .prnu import PCE_THRESHOLD, PRNUIdentifier, match_score


def _lookup(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """device -> (brand, model)"""
    first = df.drop_duplicates("device").set_index("device")
    return {d: (r["brand"], r["model"]) for d, r in first.iterrows()}


def _load_many(rows: pd.DataFrame, crop: int, desc: str):
    for _, row in tqdm(list(rows.iterrows()), desc=desc, leave=False):
        img = load_rgb(row["path"], crop)
        if img is not None:
            yield row, img


def _write(out: Path, name: str, results: pd.DataFrame, metrics: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f"{name}_per_image.csv", index=False)
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))


def summarise(results: pd.DataFrame, group: str = "condition") -> dict:
    """Accuracy at device, model and brand level for each condition."""
    out = {}
    for cond, g in results.groupby(group, sort=False):
        m = {"n": len(g)}
        for level in ("device", "model", "brand"):
            if f"pred_{level}" in g:
                m[f"{level}_acc"] = float((g[f"pred_{level}"] == g[level]).mean())
        if "pce_true" in g:
            m["median_pce_true"] = float(g["pce_true"].median())
            m["verified_rate"] = float((g["pce_true"] > PCE_THRESHOLD).mean())
            m["false_match_rate"] = float((g["max_pce_other"] > PCE_THRESHOLD).mean())
        out[cond] = m
    return out


def build_prnu(df: pd.DataFrame, crop: int, min_flat: int = 5, max_images: int = 50,
               sigma: float = 3.0) -> PRNUIdentifier:
    """One fingerprint per device: from flat images if there are enough, else training photos."""
    ident = PRNUIdentifier(sigma)
    train = df[(df["split"] == "train") & (df["platform"] == "native")]
    for device, g in tqdm(list(train.groupby("device")), desc="Fingerprints"):
        flat = g[g["kind"] == "flat"]
        use = flat if len(flat) >= min_flat else g[g["kind"] == "natural"]
        grays = (to_gray(img) for _, img in _load_many(use.head(max_images), crop, device))
        try:
            ident.add_device(device, grays)
        except ValueError:
            print(f"  skipped {device}: no usable images at crop {crop}")
    return ident


def run_prnu(df: pd.DataFrame, out: str | Path, crop: int = 512,
             transform_specs: list[str] = ("none",), sigma: float = 3.0) -> dict:
    out = Path(out)
    lookup = _lookup(df)
    ident = build_prnu(df, crop, sigma=sigma)
    test = df[(df["split"] == "test") & (df["device"].isin(ident.fingerprints))]
    rows = []

    def record(row, gray, condition):
        pred, _, scores = ident.identify(gray)
        others = [s for d, s in scores.items() if d != row["device"]]
        rows.append({
            "path": row["path"], "condition": condition, "device": row["device"],
            "model": row["model"], "brand": row["brand"], "pred_device": pred,
            "pred_model": lookup[pred][1], "pred_brand": lookup[pred][0],
            "pce_true": scores.get(row["device"], np.nan),
            "max_pce_other": max(others) if others else np.nan,
        })

    native = test[test["platform"] == "native"]
    for spec in transform_specs:
        fn = T.parse(spec)
        for row, img in _load_many(native, crop, f"PRNU {spec}"):
            record(row, to_gray(fn(img)), spec)
    for platform, g in test[test["platform"] != "native"].groupby("platform"):
        for row, img in _load_many(g, crop, f"PRNU {platform}"):
            record(row, to_gray(img), f"real_{platform}")

    results = pd.DataFrame(rows)
    metrics = summarise(results)
    _write(out, "prnu", results, metrics)
    plot_conditions(metrics, out / "prnu_robustness.png", "PRNU: device identification accuracy")
    plot_pce(results, out / "prnu_pce.png")
    return metrics


def _cnn_data(df: pd.DataFrame, split: str, crop: int, level: str):
    rows = df[(df["split"] == split) & (df["platform"] == "native") & (df["kind"] == "natural")]
    images, labels = [], []
    for row, img in _load_many(rows, crop, f"Loading {split}"):
        images.append(img)
        labels.append(row[level])
    return images, labels


def run_cnn_train(df: pd.DataFrame, out: str | Path, level: str = "model", crop: int = 512,
                  epochs: int = 20, patch: int = 64, patches_per_image: int = 32, seed: int = 0) -> dict:
    from . import cnn

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr_img, tr_lab = _cnn_data(df, "train", crop, level)
    te_img, te_lab = _cnn_data(df, "test", crop, level)
    classes = sorted(set(tr_lab))
    idx = {c: i for i, c in enumerate(classes)}
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    model = cnn.train(tr_img, [idx[x] for x in tr_lab], len(classes),
                      [im for im, lab in zip(te_img, te_lab) if lab in idx],
                      [idx[lab] for lab in te_lab if lab in idx],
                      epochs=epochs, patch=patch, patches_per_image=patches_per_image,
                      seed=seed, log=log)
    cnn.save(model, classes, out, patch)
    (out / "train_log.txt").write_text("\n".join(log_lines))
    return {"level": level, "classes": len(classes), "train_images": len(tr_img)}


def run_cnn_eval(df: pd.DataFrame, model_dir: str | Path, out: str | Path, level: str = "model",
                 crop: int = 512, transform_specs: list[str] = ("none",)) -> dict:
    from . import cnn

    model, classes, patch = cnn.load(model_dir)
    lookup = _lookup(df)
    test = df[(df["split"] == "test") & (df["kind"] == "natural") & df[level].isin(classes)]
    rows = []

    def record(row, img, condition):
        pred = classes[int(cnn.predict(model, img, patch).argmax())]
        r = {"path": row["path"], "condition": condition, "device": row["device"],
             "model": row["model"], "brand": row["brand"], f"pred_{level}": pred}
        if level == "device":
            r["pred_model"], r["pred_brand"] = lookup[pred][1], lookup[pred][0]
        elif level == "model":
            r["pred_brand"] = df.loc[df["model"] == pred, "brand"].iloc[0]
        rows.append(r)

    native = test[test["platform"] == "native"]
    for spec in transform_specs:
        fn = T.parse(spec)
        for row, img in _load_many(native, crop, f"CNN {spec}"):
            record(row, fn(img), spec)
    for platform, g in test[test["platform"] != "native"].groupby("platform"):
        for row, img in _load_many(g, crop, f"CNN {platform}"):
            record(row, img, f"real_{platform}")

    results = pd.DataFrame(rows)
    metrics = summarise(results)
    _write(Path(out), f"cnn_{level}", results, metrics)
    plot_conditions(metrics, Path(out) / f"cnn_{level}_robustness.png",
                    f"CNN: {level} identification accuracy", key=f"{level}_acc")
    return metrics


def run_defence(df: pd.DataFrame, out: str | Path, crop: int = 512,
                defence_specs: list[str] = ("denoise:4", "denoise:8", "subtract:1.0"),
                max_images: int = 200, sigma: float = 3.0) -> dict:
    """Does the defence stop PRNU identification, and how much does it change the photo?"""
    out = Path(out)
    ident = build_prnu(df, crop, sigma=sigma)
    test = df[(df["split"] == "test") & (df["platform"] == "native") & (df["kind"] == "natural")
              & df["device"].isin(ident.fingerprints)].head(max_images)
    rows = []
    for row, img in _load_many(test, crop, "Defences"):
        K = ident.fingerprints[row["device"]]
        for spec in ["none", *defence_specs]:
            name, _, arg = spec.partition(":")
            if name == "none":
                protected = img
            elif name == "denoise":
                protected = denoise_defence(img, float(arg))
            elif name == "subtract":
                protected = subtract_fingerprint(img, K, float(arg))
            else:
                raise ValueError(f"Unknown defence {spec!r}")
            gray = to_gray(protected)
            pred, _, _ = ident.identify(gray)
            rows.append({"path": row["path"], "condition": spec, "device": row["device"],
                         "identified": pred == row["device"],
                         "pce_true": match_score(gray, K, sigma),
                         **image_quality(img, protected)})
    results = pd.DataFrame(rows)
    metrics = {c: {"n": len(g), "identified_rate": float(g["identified"].mean()),
                   "median_pce_true": float(g["pce_true"].median()),
                   "still_matches_rate": float((g["pce_true"] > PCE_THRESHOLD).mean()),
                   "mean_psnr": float(g["psnr"].replace(np.inf, np.nan).mean()),
                   "mean_ssim": float(g["ssim"].mean())}
               for c, g in results.groupby("condition", sort=False)}
    _write(out, "defence", results, metrics)
    plot_defence(metrics, out / "defence.png")
    return metrics


# ---- plots ---------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_conditions(metrics: dict, path: Path, title: str, key: str = "device_acc") -> None:
    plt = _plt()
    names = list(metrics)
    vals = [metrics[n].get(key, np.nan) for n in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(names)), 3.6))
    ax.bar(range(len(names)), vals, color="#3b6ea5")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pce(results: pd.DataFrame, path: Path) -> None:
    """Distribution of match scores for the true camera vs the best wrong camera."""
    plt = _plt()
    clean = results[results["condition"] == "none"]
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.logspace(-1, 6, 50)
    ax.hist(np.clip(clean["pce_true"], 0.1, None), bins=bins, alpha=0.7, label="true camera")
    ax.hist(np.clip(clean["max_pce_other"], 0.1, None), bins=bins, alpha=0.7, label="best wrong camera")
    ax.axvline(PCE_THRESHOLD, color="#333", linestyle="--", linewidth=1)
    ax.text(PCE_THRESHOLD * 1.2, ax.get_ylim()[1] * 0.9, "threshold 60", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("PCE (match strength)")
    ax.set_ylabel("Photos")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_defence(metrics: dict, path: Path) -> None:
    """Privacy vs image quality: lower-right is best (hidden camera, unchanged image)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, m in metrics.items():
        ax.scatter(m["mean_ssim"], m["still_matches_rate"], s=60)
        ax.annotate(name, (m["mean_ssim"], m["still_matches_rate"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Image similarity to original (SSIM)")
    ax.set_ylabel("Photos still matched to camera")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Privacy defences: protection vs image quality")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
