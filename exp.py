# -*- coding: utf-8 -*-
"""EXP - repeated-run experiment driver for Table 1 of the paper.

Runs the baseline and the optimized (ASP) model n times, measures
processing time, memory and task quality, and prints ready LaTeX rows
in the form   127.4 $\\pm$ 2.1 & 73.8 $\\pm$ 1.4 & $-42.1$ \\\\

Default mode is a synthetic smoke test (verifies that the pipeline
works end to end on any machine).  On the experimental cluster point
the driver at the real datasets:

    python exp.py --task iot --data iot.npz  --reps 10 --latex
    python exp.py --task txt --data txt.npz  --reps 10 --latex
    python exp.py --task fin --data fin.npz  --reps 10 --latex

Each .npz must contain arrays x (features) and y (targets/labels).

    n    - repetitions           x/y   - data
    l    - latency, ms           mem   - RSS, MB
    q    - quality (mae/f1/auc)  d     - relative change, %
"""

import argparse
import os
import time

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras import layers, Sequential

from asp import ASP

try:
    from sklearn.metrics import f1_score, roc_auc_score
    OK_SK = True
except Exception:
    OK_SK = False


# ------------------------------------------------------------------ data
def synth(task, n=2048):
    """Synthetic stream for the smoke test."""
    rng = np.random.default_rng(0)
    if task == "iot":                                # regression -> mae
        x = rng.normal(size=(n, 64, 1)).astype("float32")
        y = x.mean(axis=(1, 2)) + 0.1 * rng.normal(size=n)
        return x, y.astype("float32"), "mae"
    x = rng.normal(size=(n, 16, 16, 1)).astype("float32")
    y = (x.mean(axis=(1, 2, 3)) > 0).astype("int64")
    if task == "fin":                                # anomaly -> auc
        return x, y, "auc"
    return x, y, "f1"                                # txt -> f1


def load(task, p):
    k = {"iot": "mae", "txt": "f1", "fin": "auc"}[task]
    if p:
        if not os.path.exists(p):
            raise FileNotFoundError(
                "--data file not found: %s (refusing to fall back to "
                "synthetic data)" % p)
        z = np.load(p)
        if "x" not in z or "y" not in z:
            raise KeyError("%s must contain arrays 'x' and 'y'" % p)
        return z["x"].astype("float32"), z["y"], k, True
    return synth(task) + (False,)


# ---------------------------------------------------------------- models
def net(task, ish, ncl):
    if task == "iot":
        return Sequential([layers.Input(ish),
                           layers.Conv1D(32, 5, activation="relu"),
                           layers.Conv1D(32, 5, activation="relu"),
                           layers.Conv1D(16, 3, activation="relu"),
                           layers.GlobalAveragePooling1D(),
                           layers.Dense(1)])
    return Sequential([layers.Input(ish),
                       layers.Conv2D(32, 3, activation="relu"),
                       layers.Conv2D(32, 3, activation="relu"),
                       layers.Conv2D(16, 3, activation="relu"),
                       layers.GlobalAveragePooling2D(),
                       layers.Dense(ncl, activation="softmax")])


def qual(kind, y, z):
    if kind == "mae":
        return float(np.mean(np.abs(y - z.ravel())))
    zz = np.argmax(z, axis=1) if z.ndim > 1 else (z > 0.5).astype(int)
    if kind == "f1" and OK_SK:
        return float(f1_score(y, zz, average="macro"))
    if kind == "auc" and OK_SK and z.ndim > 1:
        return float(roc_auc_score(y, z[:, 1]))
    return float(np.mean(zz == y))                   # fallback: accuracy


# ------------------------------------------------------------------- run
def one(task, x, y, kind, opt, b=64, ep=2):
    """One full run: train, stream-evaluate; return (l, mem, q)."""
    ncl = int(y.max()) + 1 if kind != "mae" else 1
    m = net(task, x.shape[1:], ncl)
    if kind == "mae":
        m.compile("adam", "mae")
        m.fit(x, y, epochs=ep, batch_size=b, verbose=0)
        f = m
    elif opt:
        a = ASP(m, ncl, b=b)
        a.m.fit(x, [y] * len(a.m.outputs),
                epochs=ep, batch_size=b, verbose=0)
        f = a
    else:
        m.compile("adam", "sparse_categorical_crossentropy")
        m.fit(x, y, epochs=ep, batch_size=b, verbose=0)
        f = m
    if kind == "mae" and opt:                        # prune only, no exits
        try:
            from tensorflow_model_optimization.sparsity import keras as sp
            f = sp.prune_low_magnitude(m)
        except Exception:
            f = m
        f.compile("adam", "mae")
    tt, zz = [], []
    for i in range(0, len(x), b):                    # stream emulation
        xb, t0 = x[i:i + b], time.time()
        if isinstance(f, ASP):
            z = f.run_batch(xb)["y"]
        else:
            z = f.predict(xb, verbose=0)
        tt.append((time.time() - t0) * 1000.0)
        zz.append(z)
    l = float(np.mean(tt))
    mem = psutil.Process().memory_info().rss / 2 ** 20
    q = qual(kind, y, np.vstack(zz))
    return l, mem, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="txt", choices=["iot", "txt", "fin"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--quick", action="store_true")
    v = ap.parse_args()

    x, y, kind, real = load(v.task, v.data)
    a1 = ASP(net("txt", (16, 16, 1), 2), 2)          # env probe
    okp, okq = getattr(a1, "ok_p", False), getattr(a1, "ok_q", False)
    print("env: TF %s | tfmot pruning: %s | tfmot int8: %s | sklearn: %s "
          "| data: %s"
          % (tf.__version__, okp, okq, OK_SK,
             "REAL (%s)" % v.data if real else "SYNTHETIC"))
    if not real:
        print("*** SYNTHETIC smoke test: numbers verify the pipeline "
              "only and MUST NOT be used in the paper. ***")
    if not (okp and okq):
        print("*** tfmot inactive (needs tensorflow<2.16 / Keras 2): "
              "no pruning/quantization -> no memory or speed gain "
              "is expected. ***")
    if v.quick:
        x, y = x[:512], y[:512]

    rr = {0: [], 1: []}                              # 0 baseline, 1 optimized
    for opt in (0, 1):
        for i in range(v.reps):
            tf.keras.utils.set_random_seed(i)
            rr[opt].append(one(v.task, x, y, kind, opt))
            print("opt=%d rep=%d  l=%.1f ms  mem=%.0f MB  %s=%.4f"
                  % ((opt, i) + rr[opt][-1][:2] + (kind, rr[opt][-1][2])))

    a, b_ = np.array(rr[0]), np.array(rr[1])
    ms, ss = a.mean(0), a.std(0, ddof=1)
    mo, so = b_.mean(0), b_.std(0, ddof=1)
    d = (mo - ms) / ms * 100.0
    nm = ["Processing time (ms)", "Memory (MB)",
          {"mae": "MAE", "f1": "F1-score", "auc": "AUC-ROC"}[kind]]
    if v.latex:
        print("\n%% rows for Table 1 (mean $\\pm$ std, n=%d):" % v.reps)
        if not real:
            print("%% !!! SYNTHETIC DATA - verification only, "
                  "do NOT insert into the paper !!!")
        if not (okp and okq):
            print("%% !!! pruning/quantization were skipped in this "
                  "run (tfmot inactive) !!!")
        ff = ["%.1f", "%.0f", "%.3f"]                # time, mem, quality
        for k in range(3):
            f = ff[k]
            print(("  & %s & " + f + " $\\pm$ " + f + " & "
                   + f + " $\\pm$ " + f + " & $%+.1f$ \\\\")
                  % (nm[k], ms[k], ss[k], mo[k], so[k], d[k]))
    else:
        for k in range(3):
            print("%-22s base %.3f+-%.3f  opt %.3f+-%.3f  d=%+.1f%%"
                  % (nm[k], ms[k], ss[k], mo[k], so[k], d[k]))


if __name__ == "__main__":
    main()
