# -*- coding: utf-8 -*-
"""BAL - dynamic load balancing (Sec. 4.3 of the paper).

rw()  - multi-criteria reward, Eq. (2)
QB    - tabular Q-learning controller over discretized system state;
        actions change the batch size and the early-exit threshold.

    l  - latency, ms       t  - throughput, samples/s
    a  - accuracy [0,1]    e  - energy, J
    w  - weights (w1..w4)  lm/tt/em - bounds L_max, T_tgt, E_max
    c  - cpu load %        u  - stream intensity, samples/s
"""

import random
from collections import defaultdict


def rw(l, t, a, e, w=(0.4, 0.3, 0.2, 0.1),
       lm=200.0, tt=2000.0, em=100.0):
    """Reward of Eq. (2)."""
    x1 = max(0.0, 1.0 - l / lm)
    x2 = min(1.0, t / tt)
    x3 = a
    x4 = max(0.0, 1.0 - e / em)
    return w[0] * x1 + w[1] * x2 + w[2] * x3 + w[3] * x4


# actions: (batch multiplier, threshold delta)
AA = [(0.5, -0.05), (0.5, 0.0), (1.0, -0.05), (1.0, 0.0),
      (1.0, +0.02), (2.0, 0.0), (2.0, +0.02)]


class QB:
    """eps-greedy tabular Q-learning over (cpu bucket, load bucket)."""

    def __init__(self, g=0.9, lr=0.1, eps=0.1):
        self.q = defaultdict(lambda: [0.0] * len(AA))
        self.g, self.lr, self.eps = g, lr, eps

    @staticmethod
    def st(c, u):
        """Discretize state: cpu into 5 buckets, intensity into 6."""
        return (min(4, int(c) // 20), min(5, int(u) // 1000))

    def act(self, s):
        if random.random() < self.eps:
            return random.randrange(len(AA))
        v = self.q[s]
        return v.index(max(v))

    def step(self, s, a, r, s2):
        v = self.q[s]
        v[a] += self.lr * (r + self.g * max(self.q[s2]) - v[a])

    def apply(self, a, b, th, bmin=1, bmax=128):
        """Map action a onto new (batch, threshold)."""
        mb, dt = AA[a]
        b = int(min(bmax, max(bmin, round(b * mb))))
        th = min(0.9, max(0.6, th + dt))
        return b, th
