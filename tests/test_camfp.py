import numpy as np
import pandas as pd
import pytest
from PIL import Image

from camfp import data, experiments
from camfp import transforms as T
from camfp.defence import denoise_defence, image_quality, subtract_fingerprint
from camfp.images import center_crop, to_gray
from camfp.prnu import PCE_THRESHOLD, PRNUIdentifier, match_score, zero_mean
from camfp.synthetic import capture, make_cameras, random_scene, write_dataset


@pytest.fixture(scope="module")
def cameras():
    return make_cameras(n_brands=2, models_per_brand=1, devices_per_model=2, size=128, seed=0)


@pytest.fixture(scope="module")
def identifier(cameras):
    rng = np.random.default_rng(1)
    ident = PRNUIdentifier()
    for cam in cameras:
        ident.add_device(cam.folder_name,
                         [to_gray(capture(cam, random_scene(128, rng, flat=True), rng)) for _ in range(8)])
    return ident


def photo(cam, seed):
    rng = np.random.default_rng(seed)
    return capture(cam, random_scene(128, rng), rng)


# ---- dataset indexing --------------------------------------------------------------
def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)


def test_index_vision_layout(tmp_path):
    dev = tmp_path / "D01_Samsung_GalaxyS3Mini" / "images"
    for sub, name in [("Flat", "D01_I_flat_0001.jpg"), ("Nat", "D01_I_nat_0001.jpg"),
                      ("NatWA", "D01_I_natWA_0001.jpg"), ("NatFBH", "D01_I_natFBH_0001.jpg")]:
        _touch(dev / sub / name)
    df = data.index_vision(tmp_path)
    assert set(df["platform"]) == {"native", "whatsapp", "facebook_high"}
    assert df["model"].iloc[0] == "Samsung_GalaxyS3Mini"
    natural = df[df["kind"] == "natural"]
    assert natural["source"].nunique() == 1  # original and its social copies share a source


def test_index_dresden_names(tmp_path):
    _touch(tmp_path / "Canon_Ixus70_0_3554.JPG")
    _touch(tmp_path / "Canon_Ixus70_1_3600.JPG")
    df = data.index_dresden(tmp_path)
    assert df["device"].nunique() == 2 and df["model"].nunique() == 1


def test_split_keeps_copies_together_and_flats_in_train():
    rows = []
    for i in range(20):
        for plat in ("native", "whatsapp"):
            rows.append([f"{i}{plat}", "x", "B", "B_M", "dev1", plat, "natural", f"src{i}"])
    rows.append(["flat", "x", "B", "B_M", "dev1", "native", "flat", "flat0"])
    df = data.split_index(pd.DataFrame(rows, columns=data.COLUMNS), test_frac=0.3, seed=0)
    assert (df.groupby("source")["split"].nunique() == 1).all()
    assert df.loc[df["kind"] == "flat", "split"].eq("train").all()
    assert 0 < (df["split"] == "test").mean() < 1


# ---- images ------------------------------------------------------------------------
def test_center_crop_uses_even_offsets():
    arr = np.arange(11 * 13).reshape(11, 13)
    crop = center_crop(arr, 4)
    top, left = np.argwhere(arr == crop[0, 0])[0]
    assert top % 2 == 0 and left % 2 == 0
    assert center_crop(arr, 20) is None


# ---- PRNU --------------------------------------------------------------------------
def test_zero_mean_removes_row_and_column_means():
    x = zero_mean(np.random.default_rng(0).normal(5, 1, (32, 32)))
    assert np.allclose(x.mean(axis=0), 0) and np.allclose(x.mean(axis=1), 0)


def test_prnu_identifies_correct_device(cameras, identifier):
    for i, cam in enumerate(cameras):
        best, score, scores = identifier.identify(to_gray(photo(cam, 100 + i)))
        assert best == cam.folder_name
        assert score > PCE_THRESHOLD
        assert all(s < PCE_THRESHOLD for d, s in scores.items() if d != cam.folder_name)


def test_jpeg_weakens_match_and_shift_breaks_it(cameras, identifier):
    cam = cameras[0]
    img = photo(cam, 7)
    K = identifier.fingerprints[cam.folder_name]
    clean = match_score(to_gray(img), K)
    assert match_score(to_gray(T.jpeg(img, 60)), K) < clean
    assert match_score(to_gray(T.shift(img, 3)), K) < PCE_THRESHOLD


# ---- transforms --------------------------------------------------------------------
@pytest.mark.parametrize("spec", ["none", "jpeg:80", "rescale:0.5", "blur:1", "shift:2", "whatsapp"])
def test_transforms_keep_shape_and_type(spec):
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = T.parse(spec)(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        T.parse("instagram")


# ---- defences ----------------------------------------------------------------------
def test_fingerprint_subtraction_hides_camera(cameras, identifier):
    cam = cameras[1]
    img = photo(cam, 11)
    K = identifier.fingerprints[cam.folder_name]
    protected = subtract_fingerprint(img, K, 1.0)
    assert match_score(to_gray(protected), K) < PCE_THRESHOLD < match_score(to_gray(img), K)
    assert image_quality(img, protected)["ssim"] > 0.95


def test_denoise_defence_changes_image_a_little(cameras):
    img = photo(cameras[0], 3)
    q = image_quality(img, denoise_defence(img, 4))
    assert 25 < q["psnr"] < 60 and 0.8 < q["ssim"] < 1


# ---- end to end --------------------------------------------------------------------
def test_prnu_experiment_end_to_end(tmp_path, cameras):
    write_dataset(tmp_path / "imgs", cameras, n_natural=6, n_flat=6, seed=2)
    df = data.split_index(data.index_own(tmp_path / "imgs"), 0.34, 0)
    metrics = experiments.run_prnu(df, tmp_path / "out", 128, ["none", "jpeg:70"])
    assert metrics["none"]["device_acc"] == 1.0
    assert (tmp_path / "out" / "prnu_robustness.png").exists()
    d = experiments.run_defence(df, tmp_path / "def", 128, ["subtract:1.0"])
    assert d["none"]["still_matches_rate"] > d["subtract:1.0"]["still_matches_rate"]


# ---- CNN (skipped if PyTorch isn't installed) --------------------------------------
def test_constrained_layer_and_training(cameras):
    torch = pytest.importorskip("torch")
    from camfp import cnn

    layer = cnn.ConstrainedConv2d(3, 4, 5, padding=2, bias=False)
    torch.nn.init.normal_(layer.weight)
    layer.constrain()
    w = layer.weight.detach()
    assert torch.allclose(w[:, :, 2, 2], torch.tensor(-1.0))
    assert torch.allclose(w.sum(dim=(2, 3)), torch.zeros(4, 3), atol=1e-5)  # -1 centre + 1 rest

    images = [photo(c, s) for c in cameras for s in range(3)]
    labels = [i for i in range(len(cameras)) for _ in range(3)]
    model = cnn.train(images, labels, len(cameras), epochs=1, patches_per_image=4, log=lambda m: None)
    probs = cnn.predict(model, images[0])
    assert probs.shape == (len(cameras),) and abs(probs.sum() - 1) < 1e-4
