"""Handwritten text recognition: OpenCV segmentation + the EMNIST-trained CNN in AZaz09weights.

Pipeline
1. Binarise: divide out the uneven paper background, then Otsu threshold. Faint
   bleed-through from the back of the page falls below the threshold.
2. Segment: connected components; remove specks; group components into text lines
   left-to-right (so slanted lines are followed); attach i/j dots to their stems and
   merge broken pieces of one character; detect full stops and hyphens.
3. Normalise each character the way EMNIST images were made: keep the aspect ratio,
   fill the 28x28 frame, normalise stroke thickness, anti-alias and blur.
4. Classify with the CNN (the original one, or the deeper one from train.py), averaging
   predictions over a few thickness/scale variants. Rendering settings are per model (PREP).
   Wide characters the model is unsure about (touching letters such as "th") are cut at
   their thinnest column when the model is confident about both halves.
5. Decode with context: each word is read as letters or digits as a whole, unless a
   switch is strongly supported, so "a1ong" becomes "along" and "l0" becomes "10". Plain
   vertical bars, which the model cannot tell apart, become i / I / l from the dot, the
   word's case and the local x-height.
"""
import os

import cv2 as cv
import h5py
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "AZaz09weights")                 # original CNN (2020)
NEW_MODEL_PATH = os.path.join(HERE, "models", "emnist_cnn.keras")   # deeper CNN from train.py

OUTPUT_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
                 'f', 'g', 'h', 'n', 'q', 'r', 't']
DIGITS = np.arange(10)
LETTERS = np.arange(10, 47)
LOWER_ONLY = {'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'}
# EMNIST merges these letters' cases into the uppercase class.
UPPER_HAS_LOWER = set('ABDEFGHNQRT')

# Tunables (character sizes are relative to the median character height of the page).
# How characters are rendered for each model (tuned with evaluate.py):
#   fill   - size of the character's longest side inside the 28x28 frame
#   stroke - stroke width in 28x28 pixels to normalise to, or None to keep the natural width
#   blur   - Gaussian blur sigma after resizing (0 = none)
#   tta    - average predictions over a few size/thickness variants
PREP = {
    "original": dict(fill=22, stroke=3.5, blur=0.8, tta=True),
    "new": dict(fill=20, stroke=None, blur=0, tta=False),
}
SPACE_GAP = 0.35      # smallest gap (x median height) that may count as a word break
SWITCH_PENALTY = 4.0  # log-prob cost of switching between letters and digits inside a word


def _larger_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(30, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(47, activation='softmax'))
    return model


def _load_legacy_weights(model, path):
    """Load weights saved by old Keras (HDF5, by layer name) into the model's layers in order."""
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
        weighted = [l for l in model.layers if l.weights]
        saved = []
        for n in names:
            lg = g[n]
            wn = [w.decode() if isinstance(w, bytes) else w for w in lg.attrs["weight_names"]]
            if wn:
                saved.append([lg[w][()] for w in wn])
        assert len(saved) == len(weighted), (len(saved), len(weighted))
        for layer, ws in zip(weighted, saved):
            layer.set_weights(ws)


_MODELS = {}


def model_name():
    """Which classifier to use: HWR_MODEL=original|new, default "new" when it has been trained."""
    name = os.environ.get("HWR_MODEL", "").lower()
    if name in ("original", "new"):
        return name
    return "new" if os.path.exists(NEW_MODEL_PATH) else "original"


def get_model(name=None):
    name = name or model_name()
    if name not in _MODELS:
        if name == "new":
            import keras
            _MODELS[name] = keras.models.load_model(NEW_MODEL_PATH)
        else:
            model = _larger_model()
            _load_legacy_weights(model, WEIGHTS_PATH)
            _MODELS[name] = model
    return _MODELS[name]


# ----------------------------------------------------------------------------- 1. binarise

def estimate_background(gray, factor=4):
    """Paper brightness everywhere, with the ink removed (morphological closing with a disc larger
    than any stroke). The background varies slowly, so it is computed on a copy shrunk by
    `factor` (taking the brightest pixel of each block, so thin ink does not darken it) and
    scaled back up: the same result at a fraction of the cost of closing the full image."""
    h, w = gray.shape
    hs, ws = h // factor, w // factor
    if hs < 32 or ws < 32:
        factor, hs, ws = 1, h, w
    small = gray[:hs * factor, :ws * factor].reshape(hs, factor, ws, factor).max(axis=(1, 3))
    k = max(31, (min(gray.shape) // 20) | 1) // factor | 1
    small = cv.morphologyEx(small, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    return cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR)


def binarise(gray):
    """Return a uint8 mask (255 = ink) robust to shadows, uneven lighting and show-through."""
    gray = cv.medianBlur(gray, 3)
    background = estimate_background(gray)
    norm = cv.divide(gray, background, scale=255)          # paper ~255, ink darker
    ink = cv.GaussianBlur(255 - norm, (3, 3), 0)
    _, mask = cv.threshold(ink, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Faint show-through shapes produce weak ink; Otsu on a page of mostly paper can set the
    # threshold too low, so require a minimum contrast as well.
    mask[ink < 40] = 0
    return mask


# ----------------------------------------------------------------------------- 2. segment

class Box:
    def __init__(self, x0, y0, x1, y1, labels):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.labels = list(labels)
        self.kind = 'char'  # or 'dot', 'dash'
        self.has_dot = False
        self.band = None    # (top, bottom) of the local x-height band, set for bar-shaped boxes

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    def absorb(self, other):
        self.x0, self.y0 = min(self.x0, other.x0), min(self.y0, other.y0)
        self.x1, self.y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        self.labels += other.labels


def segment(mask):
    """Return (lines, label_image, median_height); lines is a list of lists of Box sorted by x."""
    # Join hairline breaks within a stroke before finding components.
    joined = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv.connectedComponentsWithStats(joined, connectivity=8)
    boxes = [Box(x, y, x + w, y + h, [i]) for i, (x, y, w, h, a) in enumerate(stats) if i > 0 and a > 0]
    if not boxes:
        return [], labels, 0

    heights = np.array([b.h for b in boxes])
    areas = np.array([stats[b.labels[0], cv.CC_STAT_AREA] for b in boxes])
    big = heights > np.percentile(heights, 50) * 0.5
    med_h = float(np.median(heights[big]))
    med_area = float(np.median(areas[big]))

    chars, smalls = [], []
    for b, a in zip(boxes, areas):
        if a < med_area * 0.01 or max(b.w, b.h) < med_h * 0.06:
            continue                                       # speck
        if b.h < med_h * 0.4 and b.w < med_h * 0.4:
            b.kind = 'dot'
            smalls.append(b)
        elif b.h < med_h * 0.3 and b.w > b.h * 2.5:
            b.kind = 'dash'
            smalls.append(b)
        else:
            chars.append(b)

    # Merge pieces of one character that overlap horizontally (broken strokes, "i" serifs...).
    chars.sort(key=lambda b: b.x0)
    merged = []
    for b in chars:
        for m in merged[-4:]:
            ov = min(m.x1, b.x1) - max(m.x0, b.x0)
            vgap = max(m.y0, b.y0) - min(m.y1, b.y1)
            if ov > 0.6 * min(m.w, b.w) and vgap < med_h * 0.25 and max(m.y1, b.y1) - min(m.y0, b.y0) < med_h * 1.6:
                m.absorb(b)
                break
        else:
            merged.append(b)
    chars = merged

    # Group characters into lines, scanning left to right so a sloping line is followed.
    # A character joins the line whose recent text band (median top..bottom of its last few
    # characters) it overlaps most; ascenders and descenders still cover that band.
    lines = []
    for b in sorted(chars, key=lambda b: b.x0):
        best, best_ov = None, 0.0
        for line in lines:
            recent = line[-4:]
            t, btm = np.median([c.y0 for c in recent]), np.median([c.y1 for c in recent])
            ov = (min(b.y1, btm) - max(b.y0, t)) / max(1.0, min(b.h, btm - t))
            if ov > 0.5 and ov > best_ov:
                best, best_ov = line, ov
        if best is None:
            lines.append([b])
        else:
            best.append(b)
    # merge line fragments that share a band and do not overlap horizontally (a line that
    # started with a tall or descending character can split in two)
    lines.sort(key=lambda l: l[0].x0)
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines):
            a, c = lines[i], lines[j]
            at, ab = np.median([x.y0 for x in a]), np.median([x.y1 for x in a])
            ct, cb = np.median([x.y0 for x in c]), np.median([x.y1 for x in c])
            ov = (min(ab, cb) - max(at, ct)) / max(1.0, min(ab - at, cb - ct))
            # local check: the end of one fragment against the start of the other, for slanted lines
            if c[0].x0 >= a[-1].x1 - med_h * 0.2:
                left, right = a[-4:], c[:4]
            elif a[0].x0 >= c[-1].x1 - med_h * 0.2:
                left, right = c[-4:], a[:4]
            else:
                left = right = None
            local = 0.0
            if left is not None and right[0].x0 - left[-1].x1 < med_h * 4:
                lt, lb = np.median([x.y0 for x in left]), np.median([x.y1 for x in left])
                rt, rb = np.median([x.y0 for x in right]), np.median([x.y1 for x in right])
                local = (min(lb, rb) - max(lt, rt)) / max(1.0, min(lb - lt, rb - rt))
            if ov > 0.5 or local > 0.3:
                lines[i] = sorted(a + c, key=lambda x: x.x0)
                del lines[j]
            else:
                j += 1
        i += 1
    lines.sort(key=lambda l: np.median([c.cy for c in l]))

    # A dot directly above a stem belongs to it (i, j); otherwise it is punctuation.
    for s in smalls:
        stem = None
        if s.kind == 'dot':
            for line in lines:
                for c in line:
                    if (c.x0 - med_h * 0.15 <= s.cx <= c.x1 + med_h * 0.15 and s.y1 <= c.y0 + med_h * 0.15
                            and c.y0 - s.y1 < med_h * 0.6 and c.w < med_h * 0.6):
                        if stem is None or abs(c.cx - s.cx) < abs(stem.cx - s.cx):
                            stem = c
        if stem is not None:
            stem.absorb(s)
            stem.has_dot = True
            continue
        # a piece that lies inside a character's box is a broken stroke of it (e.g. an arm of "E")
        host = next((c for line in lines for c in line
                     if c.x0 <= s.cx <= c.x1 and c.y0 - med_h * 0.1 <= s.cy <= c.y1 + med_h * 0.1), None)
        if host is not None:
            host.absorb(s)
            continue
        # punctuation: compare with the characters of each line nearest to it horizontally
        def local(line):
            near = sorted(line, key=lambda c: abs(c.cx - s.cx))[:3]
            return np.median([c.cy for c in near]), np.median([c.h for c in near])
        if not lines:
            continue
        line = min(lines, key=lambda l: abs(local(l)[0] - s.cy))
        lcy, lh = local(line)
        if s.kind == 'dot' and s.cy > lcy and s.cy < lcy + lh:   # sits on the baseline: full stop
            line.append(s)
        elif s.kind == 'dash' and abs(s.cy - lcy) < lh * 0.4:
            line.append(s)
    for line in lines:
        line.sort(key=lambda b: b.cx)
    return lines, labels, med_h


# ----------------------------------------------------------------------------- 3. normalise

def _stroke_width(mask):
    dist = cv.distanceTransform(mask, cv.DIST_L2, 3)
    vals = dist[dist > 0]
    return 4.0 * float(vals.mean()) if vals.size else 1.0


def to_emnist(mask, labels, box, fill=22, stroke=3.5, blur=0.8):
    """Crop one character and render it as a 28x28 float image (aspect ratio kept)."""
    pad = 4
    y0, y1, x0, x1 = box.y0, box.y1, box.x0, box.x1
    crop = np.isin(labels[y0:y1, x0:x1], box.labels) & (mask[y0:y1, x0:x1] > 0)
    crop = np.pad(crop.astype(np.uint8) * 255, pad)
    scale = fill / max(box.w, box.h)
    delta = 0
    if stroke is not None:
        sw = _stroke_width(crop)
        delta = int(round((stroke / scale - sw) / 2))       # full-res pixels to add per side
        delta = max(-int(sw / 3), min(delta, int(4 / scale) + 1))
    if delta > 0:
        crop = np.pad(crop, delta)
        crop = cv.dilate(crop, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * delta + 1, 2 * delta + 1)))
    elif delta < 0:
        crop = cv.erode(crop, cv.getStructuringElement(cv.MORPH_ELLIPSE, (-2 * delta + 1, -2 * delta + 1)))
    ys, xs = np.nonzero(crop)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    small = cv.resize(square, (fill, fill), interpolation=cv.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    o = (28 - fill) // 2
    out[o:o + fill, o:o + fill] = small
    if blur:
        out = cv.GaussianBlur(out, (3, 3), blur)
    return out / max(out.max(), 1e-6)


# ----------------------------------------------------------------------------- 4. classify

def classify(mask, labels, boxes):
    """Return an (n, 47) array of class log-probabilities, averaged over a few renderings."""
    p = PREP[model_name()]
    fill, stroke = p["fill"], p["stroke"]
    variants = [(fill, stroke)]
    if p["tta"]:
        if stroke is None:
            variants += [(fill + 2, None), (fill - 2, None)]
        else:
            variants += [(fill + 2, stroke), (fill, stroke * 1.3), (fill - 2, stroke * 0.8)]
    batch = []
    for b in boxes:
        for f, s in variants:
            batch.append(to_emnist(mask, labels, b, f, s, p["blur"]).T)   # EMNIST images are transposed
    probs = get_model().predict(np.asarray(batch)[..., None], verbose=0)
    probs = probs.reshape(len(boxes), len(variants), 47).mean(axis=1)
    return np.log(probs + 1e-9)


# ----------------------------------------------------------------------------- 5. decode

SPLIT_MAX_CONF = 0.65   # only try to split characters the model is unsure about
SPLIT_MIN_CONF = 0.85   # ...and accept a cut only if both halves are recognised at least this well


def _sub_box(mask, labels, box, x0, x1):
    """The part of `box` between columns x0..x1, with its vertical extent tightened to the ink."""
    region = np.isin(labels[box.y0:box.y1, x0:x1], box.labels) & (mask[box.y0:box.y1, x0:x1] > 0)
    rows = np.nonzero(region.any(axis=1))[0]
    if len(rows) == 0:
        return None
    sub = Box(x0, box.y0 + rows[0], x1, box.y0 + rows[-1] + 1, box.labels)
    return sub


def split_touching(mask, labels, lines, logp, med_h):
    """Touching letters ("to", "th") form one connected shape that the model cannot read. For a
    wide character the model is unsure about, try cutting it at the thinnest columns and keep
    the cut when the model is confident about both halves. Modifies lines and logp in place."""
    candidates = []                      # (line, box, [(left, right), ...])
    for line in lines:
        for b in line:
            if b.kind != 'char' or b.w < b.h * 1.05 or b.w < med_h * 0.8:
                continue
            if np.exp(logp[id(b)].max()) > SPLIT_MAX_CONF:
                continue
            region = np.isin(labels[b.y0:b.y1, b.x0:b.x1], b.labels) & (mask[b.y0:b.y1, b.x0:b.x1] > 0)
            ink = region.sum(axis=0).astype(float)
            lo, hi = int(b.w * 0.25), int(b.w * 0.75)
            if hi <= lo:
                continue
            cuts = []
            for x in np.argsort(ink[lo:hi]) + lo:            # thinnest columns first
                if all(abs(x - c) > b.w * 0.1 for c in cuts):
                    cuts.append(int(x))
                if len(cuts) == 3:
                    break
            pairs = []
            for c in cuts:
                left = _sub_box(mask, labels, b, b.x0, b.x0 + c)
                right = _sub_box(mask, labels, b, b.x0 + c, b.x1)
                if left is not None and right is not None and min(left.w, right.w) > med_h * 0.2:
                    pairs.append((left, right))
            if pairs:
                candidates.append((line, b, pairs))
    if not candidates:
        return
    pieces = [p for _, _, pairs in candidates for pair in pairs for p in pair]
    piece_logp = classify(mask, labels, pieces)
    k = 0
    for line, b, pairs in candidates:
        best, best_score = None, np.exp(logp[id(b)].max())
        for left, right in pairs:
            conf = np.exp(min(piece_logp[k].max(), piece_logp[k + 1].max()))
            if conf > SPLIT_MIN_CONF and conf > best_score:
                best, best_score = (left, right, piece_logp[k], piece_logp[k + 1]), conf
            k += 2
        if best is not None:
            left, right, lp_l, lp_r = best
            i = line.index(b)
            line[i:i + 1] = [left, right]
            logp[id(left)], logp[id(right)] = lp_l, lp_r


def decode_word(logp, prior=(0.0, 0.0)):
    """Choose letters or digits for each character with a penalty for switching mid-word.

    Returns (labels, states) where state 0 = letter, 1 = digit. `prior` is a log-prior on the
    first character's state, taken from how the rest of the page reads.
    """
    n = len(logp)
    best = np.stack([logp[:, LETTERS].max(1), logp[:, DIGITS].max(1)], 1)   # state 0 letter, 1 digit
    arg = np.stack([LETTERS[logp[:, LETTERS].argmax(1)], DIGITS[logp[:, DIGITS].argmax(1)]], 1)
    score = best[0] + np.asarray(prior)
    back = np.zeros((n, 2), int)
    for i in range(1, n):
        new = np.empty(2)
        for s in range(2):
            stay, switch = score[s], score[1 - s] - SWITCH_PENALTY
            back[i, s] = s if stay >= switch else 1 - s
            new[s] = max(stay, switch) + best[i, s]
        score = new
    s = int(score.argmax())
    out, states = [], []
    for i in range(n - 1, -1, -1):
        out.append(OUTPUT_LABELS[arg[i, s]])
        states.append(s)
        s = back[i, s]
    return out[::-1], states[::-1]


def _is_bar(box):
    return box.w <= box.h * 0.45


def word_is_upper(labels):
    """True when most letters whose case the model can tell apart are capitals."""
    distinct = [c for c in labels if c in LOWER_ONLY or c in UPPER_HAS_LOWER]
    return bool(distinct) and sum(c.isupper() for c in distinct) > len(distinct) / 2


def fix_bars(labels, boxes):
    """A plain vertical bar is ambiguous to the model: its "I" class means a seriffed capital I or
    a dotted i, so a bare bar comes out as "1" or "l". Decide from what segmentation saw:
    - a bar with a dot attached is "i";
    - in a word written in capitals it is "I";
    - otherwise compare with the line's x-height band around it: rising clearly above it is
      "l", staying within it is an "i" whose dot was not found.
    """
    upper = word_is_upper([lab for lab, b in zip(labels, boxes) if not _is_bar(b)])
    out = list(labels)
    for k, (lab, b) in enumerate(zip(labels, boxes)):
        if lab not in ('I', 'L') or not _is_bar(b):
            continue
        if b.has_dot:
            out[k] = 'i'
        elif upper:
            out[k] = 'I'
        elif b.band is not None:
            top, bottom = b.band
            out[k] = 'l' if b.y0 < top - 0.2 * (bottom - top) else 'i'
    return out


def set_bands(line):
    """For each bar in the line, estimate the x-height band from its nearest other characters:
    the median top and bottom of the shorter half of them (letters like a, e, n, o)."""
    others = [c for c in line if c.kind == 'char' and not _is_bar(c)]
    if len(others) < 3:
        return
    for b in line:
        if b.kind == 'char' and _is_bar(b):
            near = sorted(others, key=lambda c: abs(c.cx - b.cx))[:8]
            h = np.median([c.h for c in near])
            short = [c for c in near if c.h <= h]           # drop ascenders/descenders
            b.band = (float(np.median([c.y0 for c in short])), float(np.median([c.y1 for c in short])))


def format_word(chars, capitalise):
    word = ''.join(chars)
    letters = [c for c in chars if c.isalpha()]
    if len(letters) > 1 and word_is_upper(letters):
        return word.upper()
    word = word.lower()
    if capitalise and chars and chars[0].isupper():
        word = word[0].upper() + word[1:]
    return word


def word_gap_threshold(lines, med_h):
    """Split the page's gaps between neighbouring characters into letter gaps and word gaps (1-D Otsu)."""
    gaps = np.array([b.x0 - a.x1 for line in lines for a, b in zip(line, line[1:])
                     if a.kind == 'char' and b.kind == 'char'], float) / med_h
    lo, hi = SPACE_GAP, 1.6
    if len(gaps) < 4:
        return SPACE_GAP * 1.5 * med_h
    best_t, best_var = None, -1.0
    for t in np.linspace(lo, hi, 60):
        a, b = gaps[gaps <= t], gaps[gaps > t]
        if len(a) == 0 or len(b) == 0:
            continue
        var = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if var > best_var:
            best_t, best_var = t, var
    return (best_t if best_t is not None else SPACE_GAP * 1.5) * med_h


def image_method(path):
    gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")
    mask = binarise(gray)
    lines, labels, med_h = segment(mask)
    chars = [b for line in lines for b in line if b.kind == 'char']
    if not chars:
        return ""
    logp = dict(zip(map(id, chars), classify(mask, labels, chars)))
    split_touching(mask, labels, lines, logp, med_h)
    for line in lines:
        set_bands(line)

    gap_threshold = word_gap_threshold(lines, med_h)
    # split every line into words at large gaps
    page = []
    for line in lines:
        words, cur, prev = [], [], None
        for b in line:
            if prev is not None and b.kind == 'char' and b.x0 - prev.x1 > gap_threshold:
                words.append(cur)
                cur = []
            cur.append(b)
            prev = b
        words.append(cur)
        page.append(words)

    # page-level prior: how much of the page reads as letters vs digits (decides 1-char words)
    states = []
    for words in page:
        for word in words:
            cs = [b for b in word if b.kind == 'char']
            if len(cs) > 1:
                states += decode_word(np.array([logp[id(b)] for b in cs]))[1]
    frac_letters = float(np.clip(1 - np.mean(states), 0.25, 0.75)) if states else 0.5
    prior = (np.log(frac_letters), np.log(1 - frac_letters))

    out_lines = []
    sentence_start = True
    for words in page:
        texts = []
        for word in words:
            cs = [b for b in word if b.kind == 'char']
            letters = fix_bars(decode_word(np.array([logp[id(b)] for b in cs]), prior)[0], cs) if cs else []
            text, k = '', 0
            for b in word:
                if b.kind == 'char':
                    text += letters[k]
                    k += 1
                elif b.kind == 'dot':
                    text += '.'
                else:
                    text += '-'
            core = text.rstrip('.')
            body = format_word(list(core), sentence_start) if core else ''
            texts.append(body + text[len(core):])
            if core:
                sentence_start = False
            if text.endswith('.'):
                sentence_start = True
        out_lines.append(' '.join(t for t in texts if t))
    return '\n'.join(out_lines)
