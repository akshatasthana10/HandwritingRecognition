"""Local web app for handwriting recognition.

Run:  .venv/bin/python server.py      (opens http://127.0.0.1:8765 in your browser)

Everything runs on this computer: images are sent only to this local server.
"""
import argparse
import os
import re
import threading
import time
import webbrowser

import cv2 as cv
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import app as engine

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
DATASET = os.path.join(HERE, "dataset")
MAX_UPLOAD = 25 * 1024 * 1024
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

api = FastAPI(title="Handwriting recognition", docs_url=None, redoc_url=None)
_thumbs = {}


@api.on_event("startup")
def warm_up():
    # load the models in the background so the page is usable straight away
    def load():
        for name in engine.available_models():
            engine.get_model(name)
    threading.Thread(target=load, daemon=True).start()


@api.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@api.get("/api/info")
def info():
    return {"models": engine.available_models(), "default": engine.model_name()}


def _samples():
    if not os.path.isdir(DATASET):
        return []
    files = [f for f in os.listdir(DATASET) if f.lower().endswith(IMAGE_TYPES)]
    # natural order: Test2 before Test10
    return sorted(files, key=lambda f: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", f)])


@api.get("/api/samples")
def samples():
    return [{"name": f, "label": os.path.splitext(f)[0].split(" ")[0]} for f in _samples()]


def _sample_path(name):
    if name not in _samples():                      # only files listed above, never a path
        raise HTTPException(404, "No such sample page.")
    return os.path.join(DATASET, name)


@api.get("/api/samples/{name}")
def sample(name: str):
    return FileResponse(_sample_path(name))


@api.get("/api/samples/{name}/thumb")
def sample_thumb(name: str):
    if name not in _thumbs:
        img = cv.imread(_sample_path(name))
        scale = 360 / max(img.shape[:2])
        img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        _thumbs[name] = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return Response(_thumbs[name], media_type="image/jpeg")


@api.post("/api/recognize")
def recognize(image: UploadFile = File(...), model: str = Form(None)):
    data = image.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "This image is larger than 25 MB. Use a smaller photo.")
    try:
        gray = engine.read_image(data)
    except ValueError:
        raise HTTPException(415, "This file isn't an image the app can read. Use a JPG or PNG photo.")
    if model not in (None, "", *engine.available_models()):
        raise HTTPException(400, f"Unknown model: {model}")
    start = time.time()
    result = engine.recognize(gray, model or None)
    result["seconds"] = round(time.time() - start, 3)
    return result


api.mount("/static", StaticFiles(directory=WEB), name="static")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Handwriting recognition running at {url}  (Ctrl-C to stop)")
    uvicorn.run(api, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
