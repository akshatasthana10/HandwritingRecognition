"""Measure recognition accuracy on the sample images in dataset/ against dataset/ground_truth.json.

Character accuracy = 1 - (edit distance / reference length), compared case-insensitively
(the model's 47 EMNIST classes merge upper/lower case for many letters) with all
whitespace collapsed to single spaces. Character accuracy ignoring spaces is also reported.

Usage: .venv/bin/python evaluate.py [-v] [--model original|new]
"""
import contextlib
import io
import json
import os
import re
import sys

if "--model" in sys.argv:
    os.environ["HWR_MODEL"] = sys.argv[sys.argv.index("--model") + 1]

from app import image_method, model_name

HERE = os.path.dirname(os.path.abspath(__file__))


def normalise(text, keep_spaces=True):
    text = re.sub(r"\s+", " " if keep_spaces else "", text.lower()).strip()
    return text


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    verbose = "-v" in sys.argv
    truth = json.load(open(os.path.join(HERE, "dataset", "ground_truth.json")))
    tot = {"err": 0, "len": 0, "err_ns": 0, "len_ns": 0}
    print(f"model: {model_name()}")
    print(f"{'image':<18}{'char acc':>10}{'no-space':>10}")
    for name, ref in truth.items():
        with contextlib.redirect_stdout(io.StringIO()):
            out = image_method(os.path.join(HERE, "dataset", name))
        r, o = normalise(ref), normalise(out)
        rn, on = normalise(ref, False), normalise(out, False)
        e, en = edit_distance(o, r), edit_distance(on, rn)
        tot["err"] += e; tot["len"] += len(r); tot["err_ns"] += en; tot["len_ns"] += len(rn)
        print(f"{name:<18}{100 * (1 - e / len(r)):>9.1f}%{100 * (1 - en / len(rn)):>9.1f}%")
        if verbose:
            print("   ", out.replace("\n", " / "))
    print(f"{'OVERALL':<18}{100 * (1 - tot['err'] / tot['len']):>9.1f}%{100 * (1 - tot['err_ns'] / tot['len_ns']):>9.1f}%")


if __name__ == "__main__":
    main()
