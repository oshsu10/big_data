# -*- coding: utf-8 -*-
"""KSTREAM - asynchronous stream consumer with adaptive batching
(Sec. 4.5 of the paper).

Uses aiokafka when available; otherwise falls back to a mock consumer
that generates a synthetic message stream, so the module runs anywhere.

    b    - batch size          l    - observed latency, ms
    tl   - target latency      bmin/bmax - batch bounds
"""

import asyncio
import random
import time

import numpy as np

TL = 80.0          # target latency, ms
BMIN, BMAX = 4, 128

try:
    from aiokafka import AIOKafkaConsumer
    OK_KAFKA = True
except Exception:
    OK_KAFKA = False


class MockCons:
    """Synthetic message source used when Kafka is not available."""

    def __init__(self, u=500.0):
        self.u = u                                    # msgs per second

    async def get_message(self):
        await asyncio.sleep(random.expovariate(self.u))
        return np.random.rand(32).astype("float32")


async def mk_cons(src):
    if OK_KAFKA and src.startswith("kafka://"):
        c = AIOKafkaConsumer(src.split("://", 1)[1],
                             bootstrap_servers="localhost:9092")
        await c.start()
        return c
    return MockCons()


def prep(msg):
    return np.asarray(msg, dtype="float32")


async def proc_stream(src, mdl, b=32, nmax=None):
    """Consume src, batch adaptively, run mdl on every batch."""
    cons = await mk_cons(src)
    bb, n, ll = [], 0, 50.0
    while nmax is None or n < nmax:
        try:
            m = await asyncio.wait_for(cons.get_message(), timeout=0.1)
            bb.append(prep(m))
        except asyncio.TimeoutError:
            pass
        # adaptive batching, same rule as Eq. (1)
        if ll > TL and b > BMIN:
            b = max(BMIN, b // 2)
        elif ll < TL / 2 and b < BMAX:
            b = min(BMAX, b * 2)
        if len(bb) >= b or (bb and n % 7 == 0):
            t0 = time.time()
            mdl(np.stack(bb))
            ll = (time.time() - t0) * 1000.0
            n += len(bb)
            bb = []
    return n


if __name__ == "__main__":
    f = lambda x: x.sum(axis=1)                      # trivial demo model
    k = asyncio.run(proc_stream("mock://demo", f, nmax=500))
    print("processed:", k, "messages")
