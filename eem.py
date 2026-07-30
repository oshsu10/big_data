# -*- coding: utf-8 -*-
"""EEM - adaptive early-exit model (Sec. 4.2 of the paper).

The backbone is split into segments; an auxiliary head follows every k-th
layer.  During inference a sample leaves at the first head whose maximum
softmax probability reaches the head threshold, so easy samples never
touch the deeper segments (real compute saving, not a simulation).

    bm  - sequential backbone       ths - per-head thresholds
    k   - attach a head every k layers
    ncl - number of classes
    x   - input batch, y - output probabilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential


def head(ncl, rank):
    gp = layers.GlobalAveragePooling2D() if rank == 4 \
        else layers.GlobalAveragePooling1D()
    return Sequential([gp,
                       layers.Dense(64, activation="relu"),
                       layers.Dropout(0.2),
                       layers.Dense(ncl, activation="softmax")])


class EEM:
    def __init__(self, bm, ncl, ths=(0.8, 0.9, 0.95), k=3):
        self.ncl = ncl
        self.ths = ths
        ll = [l for l in bm.layers
              if not isinstance(l, layers.InputLayer)]
        cut = [i for i in range(1, len(ll) - 1) if i % k == 0]
        cut = cut[:len(ths)]
        self.segs, self.hh, a = [], [], 0
        x = tf.keras.Input(bm.input_shape[1:])
        z = x
        for i, l in enumerate(ll):
            z = l(z)
            if i in cut:
                self.segs.append(Sequential(ll[a:i + 1]))
                self.hh.append(head(ncl, len(z.shape)))
                a = i + 1
        self.fin = Sequential(ll[a:])                # tail of the backbone

    def predict(self, x):
        """Cascade inference with per-sample early termination."""
        n = len(x)
        y = np.zeros((n, self.ncl), dtype=np.float32)
        lv = np.full(n, len(self.segs), dtype=int)   # exit level taken
        ii = np.arange(n)
        z = tf.convert_to_tensor(x)
        for j, (sg, h) in enumerate(zip(self.segs, self.hh)):
            z = sg(z, training=False)
            p = h(z, training=False).numpy()
            t = self.ths[min(j, len(self.ths) - 1)]
            mm = p.max(axis=1) >= t                  # confident -> exit
            y[ii[mm]] = p[mm]
            lv[ii[mm]] = j
            ii = ii[~mm]
            if ii.size == 0:
                return y, lv
            z = tf.boolean_mask(z, ~mm)
        y[ii] = self.fin(z, training=False).numpy()  # full depth
        return y, lv
