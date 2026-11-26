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
| Original pipeline | 89.0% | 87.3% |
| Current pipeline (same CNN and weights) | 95.9% | 95.5% |

How the pipeline gives the CNN cleaner input (see the docstring in `app.py`):
- background normalisation before thresholding, so shadows and show-through from the back of the page are dropped
- line grouping that follows slanted lines, i/j dots attached to their stems, broken strokes merged, and full stops/hyphens detected
- characters rendered like EMNIST samples: aspect ratio kept, normalised stroke width, anti-aliased and blurred
- predictions averaged over a few renderings of each character
- context-aware decoding: a word is read as all letters or all digits unless a switch is strongly supported, and a dotted vs tall bar decides i / l
