"""N-Queens workload proxy: measured local time, modelled edge time."""
import time
import numpy as np
from . import config as C


def count_solutions(n):
    """Count all N-Queens solutions with a bitmask backtracker (exponential, CPU-bound)."""
    full = (1 << n) - 1

    def go(cols, d1, d2):
        if cols == full:
            return 1
        total, free = 0, full & ~(cols | d1 | d2)
        while free:
            bit = free & -free
            free ^= bit
            total += go(cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1)
        return total
    return go(0, 0, 0)


def benchmark(ns=range(6, 14), trials=3):
    """Median wall time of counting all N-Queens solutions on this machine (not a phone)."""
    med = []
    for n in ns:
        ts = []
        for _ in range(trials):
            t0 = time.perf_counter()
            count_solutions(n)
            ts.append(time.perf_counter() - t0)
        med.append(float(np.median(ts)))
    return list(ns), med


def modelled_edge_time(t_local, rate, data_bits=8e6):
    """Edge time = local time scaled by the CPU ratio, plus setup delay and transfer of 8 Mb."""
    return t_local * (C.F_LOCAL / C.F_REMOTE) + C.SETUP_DELAY + data_bits / rate
