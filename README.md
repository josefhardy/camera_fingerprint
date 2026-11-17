# Camera Fingerprint

**Which camera took this photo, and can you hide it?**

Every camera sensor has tiny manufacturing flaws that leave a faint, unique noise pattern in
every photo it takes, like a fingerprint. This project identifies the source of a photo at
three levels (**brand**, **model** and the **individual device**) and compares two approaches:

1. **Classic sensor fingerprinting (PRNU):** the established forensic method, built from
   signal processing.
2. **Deep learning:** a CNN with a constrained first layer that learns camera traces
   directly from image patches.

It then asks two practical questions: **does identification survive social media
compression**, and **can a photographer protect their privacy** by removing the fingerprint
without visibly changing the photo?

> **Status:** methods and experiment pipeline complete and tested on simulated cameras.
> Experiments on real datasets in progress; results will be added below.

## Why it matters

- **Verifying evidence:** linking photos to a device in criminal investigations, or checking
  that a photo from a conflict zone really came from its claimed source.
- **Detecting fakes:** regions without the camera's fingerprint suggest editing, and
  AI-generated images have no real sensor fingerprint at all.
- **Fraud:** spotting insurance claim photos that were downloaded rather than taken.
- **Privacy:** the same technique could expose anonymous whistleblowers or activists, which
  is why this project also studies defences.

## Experiments

| Experiment | Question |
|---|---|
| PRNU vs CNN | Which method is better at brand, model and individual device level? |
| Robustness | How much do JPEG compression, resizing, small crops and WhatsApp/Facebook uploads hurt each method? |
| Real social media | Using VISION's genuine WhatsApp and Facebook copies, not just simulated compression |
| Privacy defence | Can denoising or fingerprint subtraction hide the camera, and at what cost to image quality? |

**Avoiding misleading results:** the train/test split is grouped by original shot, so a
photo and its WhatsApp copy never end up on opposite sides; flat images are only used to
build fingerprints; crops are aligned to the sensor's colour filter grid; and images are
never auto-rotated, because that would misalign the fingerprint.

## Methods

**PRNU** (`src/camfp/prnu.py`), following Lukas, Fridrich & Goljan (2006) and Chen et al. (2008):
1. Extract each photo's noise residual with a wavelet denoiser: `W = I - denoise(I)`.
2. Estimate the device fingerprint from many residuals: `K = Σ(W·I) / Σ(I²)`.
3. Score a new photo by correlating its residual with `I·K`, measured by Peak-to-Correlation
   Energy (PCE). PCE above 60 is the standard threshold for a match.

**CNN** (`src/camfp/cnn.py`): a constrained convolution layer (Bayar & Stamm, 2018) forces
the first filters to output prediction errors, suppressing image content and passing on
camera processing traces. It is trained on random 64×64 patches, with predictions averaged
over a grid of patches at test time. Runs on Apple Silicon (MPS), NVIDIA GPUs or CPU.

**Defences** (`src/camfp/defence.py`):
- *Denoise and re-noise:* needs nothing in advance, but strong settings soften detail.
- *Fingerprint subtraction:* estimate your own camera's fingerprint, then divide it out of
  your photos: `I' = I / (1 + αK)`.
- Image quality is measured with PSNR and SSIM.

## Results

*Coming soon.*

## Setup

```bash
git clone https://github.com/josefhardy/camera_fingerprint.git
cd camera_fingerprint
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cnn]"
```

Try the whole pipeline on simulated cameras, with no downloads needed:

```bash
camfp demo --cnn
```

## Data

| Dataset | Contents | Use |
|---|---|---|
| [VISION](https://lesc.dinfo.unifi.it/VISION/) (Shullani et al., 2017) | 35 phones and tablets; flat and natural photos plus their WhatsApp and Facebook copies | Main dataset, and real social media tests |
| Dresden Image Database (Gloe & Böhme, 2010) | About 70 standalone cameras, several units of the same model | Telling apart identical models |
| Your own photos | Phones and cameras you and friends own | Testing on current devices |

**Your own photos:** for each device, put about 30 normal photos and 10 *flat* photos (plain
sky, a white wall, or an out-of-focus blank surface) in a folder named
`<brand>_<model>_<owner>`, with the flat ones in a `flat` subfolder:

```
data/own/Apple_iPhone13_josef/IMG_0001.jpg
data/own/Apple_iPhone13_josef/flat/IMG_0101.jpg
```

Transfer original files, not compressed copies. iPhones save HEIC by default, so either set
the camera to *Most Compatible* (JPEG) before shooting or install `pillow-heif`. Only
include devices whose owners agree.

## Running

```bash
# 1. Index datasets and make a leak-free train/test split
camfp index --vision data/VISION --own data/own --out data/index.csv

# 2. Classic PRNU, with robustness tests
camfp prnu --out results/prnu --transforms none,jpeg:90,jpeg:70,jpeg:50,rescale:0.5,shift:2,whatsapp

# 3. CNN at each level
camfp cnn-train --level model --out results/cnn_model --epochs 20
camfp cnn-eval  --level model --model results/cnn_model --out results/cnn_model

# 4. Privacy defences
camfp defend --out results/defence
```

Each command writes a per-image CSV, a metrics JSON and plots. Start with a smaller
`--crop` (e.g. 256) for quick runs.

## Tests

```bash
pytest
```

The tests use simulated cameras with known fingerprints, so they check that each method
really recovers the right device.

## Ideas to extend

- Fourier-domain Wiener filtering of fingerprints to remove periodic artefacts
- Re-aligning fingerprints for resized or cropped images, e.g. a search over scale and shift
- Phones with several lenses: does each have its own fingerprint?
- Adversarial attacks that fool the CNN, and whether they also fool PRNU
- Video frames instead of still photos

## Ethics

Identifying who took a photo can harm people who share images anonymously for good reasons.
This project uses public research datasets and devices whose owners agreed, and studies
defences alongside identification.

## References

- Lukas, Fridrich & Goljan (2006). Digital camera identification from sensor pattern noise. *IEEE TIFS*.
- Chen, Fridrich, Goljan & Lukas (2008). Determining image origin and integrity using sensor noise. *IEEE TIFS*.
- Mihcak, Kozintsev & Ramchandran (1999). Spatially adaptive statistical modeling of wavelet image coefficients and its application to denoising. *ICASSP*.
- Bayar & Stamm (2018). Constrained convolutional neural networks: a new approach towards general purpose image manipulation detection. *IEEE TIFS*.
- Gloe & Böhme (2010). The Dresden Image Database for benchmarking digital image forensics. *ACM SAC*.
- Shullani et al. (2017). VISION: a video and image dataset for source identification. *EURASIP JIS*.

## Credits

Built by Josef Hardy with AI-assisted coding.
