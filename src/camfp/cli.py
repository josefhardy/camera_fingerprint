"""Command-line interface.

    camfp demo                         # whole pipeline on simulated cameras, no downloads
    camfp index --vision PATH --out data/index.csv
    camfp prnu --index data/index.csv --out results/prnu --transforms none,jpeg:90,jpeg:70,whatsapp
    camfp cnn-train --index data/index.csv --level model --out results/cnn_model
    camfp cnn-eval --index data/index.csv --model results/cnn_model --level model --out results/cnn_model
    camfp defend --index data/index.csv --out results/defence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data, experiments


def _specs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" not in df:
        raise SystemExit("Index has no split column; create it with `camfp index`.")
    return df


def cmd_index(a):
    df = data.split_index(data.build_index(a.vision, a.dresden, a.own), a.test_frac, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(data.describe(df))
    print(f"Saved {a.out}")


def cmd_prnu(a):
    m = experiments.run_prnu(_load_index(a.index), a.out, a.crop, _specs(a.transforms), a.sigma)
    print(json.dumps(m, indent=2))


def cmd_cnn_train(a):
    m = experiments.run_cnn_train(_load_index(a.index), a.out, a.level, a.crop, a.epochs,
                                  a.patch, a.patches_per_image, a.seed)
    print(json.dumps(m, indent=2))


def cmd_cnn_eval(a):
    m = experiments.run_cnn_eval(_load_index(a.index), a.model, a.out, a.level, a.crop,
                                 _specs(a.transforms))
    print(json.dumps(m, indent=2))


def cmd_defend(a):
    m = experiments.run_defence(_load_index(a.index), a.out, a.crop, _specs(a.defences), a.max_images)
    print(json.dumps(m, indent=2))


def cmd_demo(a):
    from .synthetic import make_cameras, write_dataset

    root = Path(a.out)
    print("Simulating 8 cameras (2 brands x 2 models x 2 units each)...")
    write_dataset(root / "images", make_cameras(2, 2, 2, size=256, seed=0), n_natural=20, n_flat=8)
    df = data.split_index(data.index_own(root / "images"), 0.3, 0)
    df.to_csv(root / "index.csv", index=False)
    print(data.describe(df))
    print("\nPRNU fingerprinting:")
    m = experiments.run_prnu(df, root / "prnu", 256, ["none", "jpeg:90", "jpeg:70", "whatsapp", "shift:2"])
    for cond, v in m.items():
        print(f"  {cond:<10} device acc {v['device_acc']:.2f}   median PCE {v['median_pce_true']:.0f}")
    print("\nPrivacy defences:")
    m = experiments.run_defence(df, root / "defence", 256)
    for cond, v in m.items():
        print(f"  {cond:<13} still matched {v['still_matches_rate']:.2f}   SSIM {v['mean_ssim']:.3f}")
    if a.cnn:
        print("\nCNN (camera model level):")
        experiments.run_cnn_train(df, root / "cnn", "model", 256, a.epochs, 64, 16)
        m = experiments.run_cnn_eval(df, root / "cnn", root / "cnn", "model", 256, ["none", "jpeg:70"])
        for cond, v in m.items():
            print(f"  {cond:<10} model acc {v['model_acc']:.2f}")
    print(f"\nResults and plots saved in {root}. Synthetic data only proves the code runs.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="camfp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="index datasets and make a leak-free train/test split")
    s.add_argument("--vision")
    s.add_argument("--dresden")
    s.add_argument("--own")
    s.add_argument("--out", default="data/index.csv")
    s.add_argument("--test-frac", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_index)

    def common(s, transforms=True):
        s.add_argument("--index", default="data/index.csv")
        s.add_argument("--out", required=True)
        s.add_argument("--crop", type=int, default=512)
        if transforms:
            s.add_argument("--transforms", default="none,jpeg:90,jpeg:70,jpeg:50,rescale:0.5,whatsapp")

    s = sub.add_parser("prnu", help="classic sensor fingerprint experiment")
    common(s)
    s.add_argument("--sigma", type=float, default=3.0)
    s.set_defaults(fn=cmd_prnu)

    s = sub.add_parser("cnn-train", help="train the CNN")
    common(s, transforms=False)
    s.add_argument("--level", choices=["brand", "model", "device"], default="model")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--patch", type=int, default=64)
    s.add_argument("--patches-per-image", type=int, default=32)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_cnn_train)

    s = sub.add_parser("cnn-eval", help="evaluate a trained CNN, including robustness")
    common(s)
    s.add_argument("--model", required=True)
    s.add_argument("--level", choices=["brand", "model", "device"], default="model")
    s.set_defaults(fn=cmd_cnn_eval)

    s = sub.add_parser("defend", help="privacy defence experiment")
    common(s, transforms=False)
    s.add_argument("--defences", default="denoise:4,denoise:8,subtract:0.5,subtract:1.0")
    s.add_argument("--max-images", type=int, default=200)
    s.set_defaults(fn=cmd_defend)

    s = sub.add_parser("demo", help="run everything on simulated cameras")
    s.add_argument("--out", default="results/demo")
    s.add_argument("--cnn", action="store_true", help="also train a small CNN (needs torch)")
    s.add_argument("--epochs", type=int, default=5)
    s.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
