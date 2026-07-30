# -*- coding: utf-8 -*-
"""AdaptiveStreamProcessor (ASP).

Combines pruning, 8-bit quantization, early exits and run-time adaptation
for real-time stream processing.  See Sec. 3.4 of the paper.

Naming (short form used across the repo):
    m0  - base (unoptimized) keras model
    m   - optimized model with auxiliary exits
    b   - batch size            s   - target sparsity
    qb  - quantization bits     th  - early-exit confidence threshold
    dt  - monitor period, s     ncl - number of classes
    c   - cpu load, %           r   - ram load, %
"""

import time
import threading

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

try:
    from tensorflow_model_optimization.sparsity import keras as sp
    from tensorflow_model_optimization.python.core.sparsity.keras import (
        prune as pr)
    OK_SP = True
except Exception:                                    # tfmot is optional
    OK_SP = False


def inner(l):
    """Return the wrapped layer if l is a pruning wrapper, else l."""
    return getattr(l, "layer", l)


class ASP:
    """Adaptive stream processor: build once, then feed batches."""

    def __init__(self, m0, ncl, b=32, s=0.5, qb=8, th=0.8, dt=5.0):
        self.ncl = ncl
        self.b = b
        self.s = s
        self.qb = qb
        self.th = th
        self.dt = dt
        self.mets = {"lat": [], "tput": [], "acc": [], "mem": []}
        self.cs = {"cpu": 0.0, "mem": 0.0, "lat": 0.0, "b": b, "eer": 0.0}
        self.stop = False
        self.nb = 0                                  # processed batches
        self.tfl = None                              # int8 tflite bytes
        self.m = self._build(m0)
        self.t = threading.Thread(target=self._mon, daemon=True)

    # ------------------------------------------------------------------ build
    def _build(self, m0):
        m1 = self._prune(m0)
        self.tfl = self._quant(m1)
        m2 = self._exits(m1)
        losses = ["sparse_categorical_crossentropy"] * len(m2.outputs)
        m2.compile(optimizer="adam", loss=losses,
                   metrics=[["accuracy"]] * len(m2.outputs))
        return m2

    def _prune(self, m0):
        """Polynomial-decay magnitude pruning up to sparsity s."""
        self.ok_p = False
        if not OK_SP:
            return m0
        try:
            pp = {"pruning_schedule": sp.PolynomialDecay(
                initial_sparsity=0.0, final_sparsity=self.s,
                begin_step=0, end_step=1000, frequency=100)}
            m1 = sp.prune_low_magnitude(m0, **pp)
            self.ok_p = True
            return m1
        except Exception as e:
            print("prune skipped:", e)
            return m0

    def _quant(self, m):
        """Export an int8 TFLite graph for serving; keep float for adaptation."""
        self.ok_q = False
        if self.qb != 8:
            return None
        try:
            mm = sp.strip_pruning(m) if OK_SP else m
            cv = tf.lite.TFLiteConverter.from_keras_model(mm)
            cv.optimizations = [tf.lite.Optimize.DEFAULT]
            t = cv.convert()
            self.ok_q = True
            return t
        except Exception as e:
            print("quant skipped:", e)
            return None

    def _exits(self, m):
        """Attach aux heads after ~1/3 and ~2/3 of the conv stack."""
        ll = [l for l in m.layers
              if not isinstance(l, layers.InputLayer)]
        ii = [i for i, l in enumerate(ll)
              if isinstance(inner(l), (layers.Conv2D, layers.Conv1D))]
        ee = []
        if len(ii) >= 3:
            ee = [ii[len(ii) // 3], ii[2 * len(ii) // 3]]
        x0 = tf.keras.Input(shape=m.input_shape[1:])
        x = x0
        yy = []
        for i, l in enumerate(ll):
            x = l(x)
            if i in ee:
                z = layers.GlobalAveragePooling2D()(x) if len(x.shape) == 4 \
                    else layers.GlobalAveragePooling1D()(x)
                z = layers.Dense(64, activation="relu")(z)
                z = layers.Dropout(0.5)(z)
                y1 = layers.Dense(self.ncl, activation="softmax",
                                  name="ex_%d" % i)(z)
                yy.append(y1)
        return Model(x0, yy + [x])

    # -------------------------------------------------------------- run time
    def _mon(self):
        while not self.stop:
            c = psutil.cpu_percent(interval=1)
            r = psutil.virtual_memory().percent
            self.cs["cpu"], self.cs["mem"] = c, r
            self._adapt(c, r)
            time.sleep(self.dt)

    def _adapt(self, c, r):
        """Adaptation rules, Eq. (1) of the paper."""
        if c > 80 or r > 80:
            self.b = max(1, self.b // 2)
        elif c < 30 and r < 50:
            self.b = min(128, self.b * 2)
        if c > 70:
            self.th = max(0.6, self.th - 0.05)
        else:
            self.th = min(0.9, self.th + 0.02)
        self.cs["b"] = self.b

    def run_batch(self, x, y=None):
        """Process one batch with per-sample early exit."""
        t0 = time.time()
        pp = self.m.predict(x, batch_size=self.b, verbose=0)
        if isinstance(pp, list):
            zz, kk = [], []
            for i in range(len(x)):
                z, k = pp[-1][i], len(pp) - 1
                for j in range(len(pp) - 1):         # auxiliary exits
                    if np.max(pp[j][i]) >= self.th:
                        z, k = pp[j][i], j
                        break
                zz.append(z)
                kk.append(k)
            zz = np.array(zz)
            eer = sum(k < len(pp) - 1 for k in kk) / len(x)
        else:
            zz, eer = pp, 0.0
        t1 = time.time()
        lat = (t1 - t0) * 1000.0                     # ms per batch
        tput = len(x) / (t1 - t0)                    # samples / s
        a = None
        if y is not None:
            a = float(np.mean(np.argmax(zz, axis=1) == y))
            self.mets["acc"].append(a)
        self.mets["lat"].append(lat)
        self.mets["tput"].append(tput)
        self.nb += 1
        self.cs["eer"] = eer
        self.cs["lat"] = float(np.mean(self.mets["lat"][-10:]))
        return {"y": zz, "lat": lat, "tput": tput, "acc": a,
                "b": self.b, "eer": eer}

    def start(self):
        self.stop = False
        self.t.start()

    def stop_proc(self):
        self.stop = True
        if self.t.is_alive():
            self.t.join(timeout=1.0)

    def stats(self):
        g = lambda k: float(np.mean(self.mets[k])) if self.mets[k] else 0.0
        return {"lat": g("lat"), "tput": g("tput"), "acc": g("acc"),
                "nb": self.nb, "cs": dict(self.cs)}

    def save_tflite(self, p="model_int8.tflite"):
        if self.tfl:
            open(p, "wb").write(self.tfl)
            return p
        return None
