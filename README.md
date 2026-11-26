# HandwritingRecognition

Converts an image of handwritten text into typed text using a CNN (trained on EMNIST, 47 classes) and OpenCV character segmentation, with a Kivy desktop UI.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python frontend.py
```

Click **choose image to convert**, pick an image (samples are in `dataset/`), then click **convert**.

## Accuracy

`evaluate.py` compares the output for every image in `dataset/` with the transcriptions in
`dataset/ground_truth.json` (character accuracy = 1 − edit distance / length, case-insensitive,
because the model's 47 EMNIST classes merge upper and lower case for many letters).

```bash
.venv/bin/python evaluate.py -v
```

| | char accuracy | ignoring spaces |
|---|---|---|
| Original pipeline, original CNN | 89.0% | 87.3% |
| Improved pipeline, original CNN | 95.9% | 95.5% |
| Improved pipeline, new CNN (`train.py`) | **96.8%** | **96.6%** |

The app uses the new CNN (`models/emnist_cnn.keras`) when it is present. To use the original one
(`AZaz09weights`), set `HWR_MODEL=original`, e.g. `HWR_MODEL=original .venv/bin/python frontend.py`
or `.venv/bin/python evaluate.py --model original`.

How the pipeline gives the CNN cleaner input (see the docstring in `app.py`):
- background normalisation before thresholding, so shadows and show-through from the back of the page are dropped
- line grouping that follows slanted lines, i/j dots attached to their stems, broken strokes merged, and full stops/hyphens detected
- characters rendered like EMNIST samples: aspect ratio kept, normalised stroke width, anti-aliased and blurred
- predictions averaged over a few renderings of each character
- context-aware decoding: a word is read as all letters or all digits unless a switch is strongly supported, and a dotted vs tall bar decides i / l

## Training the new CNN

`train.py` trains a deeper CNN (6 conv layers with batch norm, ~594k parameters) from scratch on
EMNIST "bymerge" (~678k images, same 47 classes as the original model). It runs in PyTorch on the
fastest device available (Apple GPU via MPS on Apple Silicon, CUDA, or CPU), keeps the dataset and the
augmentation (rotation, slant, size, shift, stroke thickness, blur) on the device, and exports the
weights to `models/emnist_cnn.keras` so the app needs no PyTorch.

```bash
uv pip install --python .venv/bin/python -r requirements-train.txt
# download https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip (~560 MB) and extract
# the emnist-bymerge-* and emnist-balanced-* files into data/emnist/gzip/
.venv/bin/python train.py --epochs 10
```

On an M3 MacBook Air, 10 epochs take about 40 minutes. EMNIST balanced test accuracy: original
model 86.7%, new model 89.2%.
