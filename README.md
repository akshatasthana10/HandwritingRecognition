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
