"""Train a deeper character CNN on EMNIST for the recognition pipeline in app.py.

Training runs in PyTorch on the fastest available device (Apple GPU via MPS on Apple
Silicon, CUDA elsewhere, CPU as a fallback). The whole dataset lives on the device and
augmentation runs there too, so the GPU is never waiting on the CPU. The trained weights
are then exported to a Keras model (models/emnist_cnn.keras), which app.py loads, so
the app itself does not need PyTorch.

Data: EMNIST "bymerge" (about 700k training images) has the same 47 classes as the
original model. Accuracy is reported on the standard EMNIST "balanced" test set for both
the original model and the new one. Download
https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip and extract the emnist-bymerge-*
and emnist-balanced-* files into data/emnist/gzip/.

Augmentation imitates what the page pipeline produces: characters slightly smaller than
the frame, small rotations, handwriting slant, shifts, thicker or thinner strokes, blur.

Usage: .venv/bin/python train.py [--epochs 10] [--batch 512]
"""
import argparse
import gzip
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "emnist", "gzip")
OUT = os.path.join(HERE, "models", "emnist_cnn.keras")
CKPT = os.path.join(HERE, "data", "emnist_cnn.pt")
LOG = os.path.join(HERE, "data", "train.log")
BN_EPS = 1e-3   # Keras' default, so the exported model computes the same thing


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- data

def load_idx(split, kind):
    name = f"emnist-{split}-{kind}-{'idx3' if kind.endswith('images') else 'idx1'}-ubyte.gz"
    with gzip.open(os.path.join(DATA, name), "rb") as f:
        data = f.read()
    if kind.endswith("images"):
        n = int.from_bytes(data[4:8], "big")
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def load(split, part):
    return load_idx(split, f"{part}-images"), load_idx(split, f"{part}-labels")


def augment(x):
    """x: (B,1,28,28) float in [0,1] on the device. Raw EMNIST images are transposed, so
    the handwriting's horizontal axis is the tensor's height axis here."""
    b, dev = x.shape[0], x.device
    glyph = torch.empty(b, device=dev).uniform_(0.77, 1.05)          # <1: smaller character
    angle = torch.empty(b, device=dev).uniform_(-0.19, 0.19)         # about +-11 degrees
    slant = torch.empty(b, device=dev).uniform_(-0.15, 0.15)
    shift = torch.empty(b, 2, device=dev).uniform_(-0.16, 0.16)
    cos, sin = torch.cos(angle) / glyph, torch.sin(angle) / glyph
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin + slant * cos, cos - slant * sin
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # stroke thickness: a third dilated, a third eroded, a third unchanged
    r = torch.rand(b, 1, 1, 1, device=dev)
    dil = F.max_pool2d(F.pad(x, (1, 0, 1, 0)), 2, 1)                 # about half a pixel per side
    ero = -F.max_pool2d(F.pad(-x, (0, 1, 0, 1), value=-1.0), 2, 1)
    x = torch.where(r < 1 / 3, dil, torch.where(r < 2 / 3, ero, x))
    # mild blur on half of the samples
    k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=dev).view(1, 1, 3, 3) / 16
    x = torch.where(torch.rand(b, 1, 1, 1, device=dev) < 0.5, F.conv2d(x, k, padding=1), x)
    # normalise like app.to_emnist (max = 1), keep strokes solid after erosion/blur
    x = x / x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x * 1.3).clamp(0, 1)


# ----------------------------------------------------------------------------- model

class Net(nn.Module):
    """Same architecture as build_keras_model below."""

    def __init__(self, num_classes=47):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o, eps=BN_EPS), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o, eps=BN_EPS), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))   # 28 -> 14 -> 7 -> 3
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128 * 9, 256), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_keras_model(num_classes=47):
    import keras
    from keras import layers

    def block(x, filters):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(epsilon=BN_EPS)(x)
            x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x)

    inputs = keras.Input((28, 28, 1))
    x = block(block(block(inputs, 32), 64), 128)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="emnist_cnn")


def export_to_keras(net):
    """Copy the PyTorch weights into the Keras model (NCHW -> NHWC) and save it."""
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    model = build_keras_model()
    convs = [l for l in model.layers if l.__class__.__name__ == "Conv2D"]
    bns = [l for l in model.layers if l.__class__.__name__ == "BatchNormalization"]
    denses = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    t_convs = [k[:-len(".weight")] for k in sd if k.endswith(".weight") and sd[k].ndim == 4]
    t_bns = [k[:-len(".running_mean")] for k in sd if k.endswith(".running_mean")]
    for layer, k in zip(convs, t_convs):
        layer.set_weights([sd[k + ".weight"].transpose(2, 3, 1, 0)])
    for layer, k in zip(bns, t_bns):
        layer.set_weights([sd[k + ".weight"], sd[k + ".bias"], sd[k + ".running_mean"], sd[k + ".running_var"]])
    w1 = sd["head.2.weight"].reshape(256, 128, 3, 3).transpose(2, 3, 1, 0).reshape(9 * 128, 256)
    denses[0].set_weights([w1, sd["head.2.bias"]])
    denses[1].set_weights([sd["head.5.weight"].T, sd["head.5.bias"]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model.save(OUT)
    return model


# ----------------------------------------------------------------------------- training

def accuracy(net, x, y, dev, batch=4096):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(dev).float().unsqueeze(1) / 255
            correct += (net(xb).argmax(1).cpu() == y[i:i + batch]).sum().item()
    return correct / len(x)


def progress(done, total, epoch, epochs, acc, start):
    frac = done / total
    elapsed = time.time() - start
    eta = elapsed / frac - elapsed
    bar = "#" * int(40 * frac) + "-" * (40 - int(40 * frac))
    sys.stdout.write(f"\r[{bar}] {100 * frac:5.1f}%  epoch {epoch}/{epochs}  acc {acc:.3f}  "
                     f"elapsed {elapsed / 60:4.1f} min  ETA {eta / 60:4.1f} min ")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = best_device()
    open(LOG, "w").close()

    x, y = load("bymerge", "train")
    x_test, y_test = load("balanced", "test")
    # bymerge and balanced share their 47-class label mapping (checked in the mapping files)
    idx = np.random.default_rng(0).permutation(len(x))
    val, tr = idx[:20_000], idx[20_000:]
    x_tr = torch.from_numpy(x[tr]).to(dev)                 # uint8 on the device (~0.5 GB)
    y_tr = torch.from_numpy(y[tr]).to(dev)
    x_val, y_val = torch.from_numpy(x[val]), torch.from_numpy(y[val])
    log(f"device: {dev}, train {len(tr)}, val {len(val)}, test {len(x_test)}")

    net = Net().to(dev)
    log(f"parameters: {sum(p.numel() for p in net.parameters()):,}")
    steps_per_epoch = math.ceil(len(tr) / args.batch)
    total = args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    start, best = time.time(), 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(tr), device=dev)
        correct = seen = 0
        for step in range(steps_per_epoch):
            bi = perm[step * args.batch:(step + 1) * args.batch]
            xb = augment(x_tr[bi].float().unsqueeze(1) / 255)
            yb = y_tr[bi]
            out = net(xb)
            loss = lossf(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 25 == 0 or step == steps_per_epoch - 1:
                correct += (out.argmax(1) == yb).sum().item()
                seen += len(yb)
                progress((epoch - 1) * steps_per_epoch + step + 1, total, epoch, args.epochs, correct / seen, start)
        val_acc = accuracy(net, x_val, y_val, dev)
        sys.stdout.write("\n")
        log(f"epoch {epoch}/{args.epochs}: train acc {correct / seen:.4f}  val acc {val_acc:.4f}  "
            f"({(time.time() - start) / 60:.1f} min elapsed)")
        if val_acc > best:
            best = val_acc
            torch.save(net.state_dict(), CKPT)
    log(f"training took {(time.time() - start) / 60:.1f} min, best val acc {best:.4f}")

    net.load_state_dict(torch.load(CKPT, map_location=dev))
    new_acc = accuracy(net, torch.from_numpy(x_test), torch.from_numpy(y_test), dev)

    # export to Keras and check it gives the same predictions as the PyTorch model
    keras_model = export_to_keras(net.cpu().eval())
    sample = x_test[:2000].astype(np.float32) / 255
    with torch.no_grad():
        p_torch = torch.softmax(net(torch.from_numpy(sample).unsqueeze(1)), 1).numpy()
    p_keras = keras_model.predict(sample[..., None], verbose=0)
    log(f"exported {OUT}: max prob difference vs PyTorch {np.abs(p_torch - p_keras).max():.2e}, "
        f"same top class {np.mean(p_torch.argmax(1) == p_keras.argmax(1)):.4f}")

    import app
    original = app.get_model("original")
    old_pred = original.predict(x_test.astype(np.float32)[..., None] / 255, batch_size=2048, verbose=0).argmax(1)
    log(f"EMNIST balanced test accuracy: original model {np.mean(old_pred == y_test):.4f}, new model {new_acc:.4f}")


if __name__ == "__main__":
    main()
