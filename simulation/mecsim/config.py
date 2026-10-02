"""Physical model and constants shared by every experiment.

Task size is parameterised by a complexity c in (0, 1]:
    cycles n = c * CYCLES_MAX,  input data m = c * DATA_MAX.
At c = 0.5 this is the paper's reference task (1e9 cycles, 8 Mb). Because n and m scale
together, the analytical break-even bandwidth does not depend on c.

All power and battery numbers are assumptions chosen to be plausible, not measurements.
"""
import numpy as np

F_LOCAL = 1.2e9          # device CPU, Hz
F_REMOTE = 4.0e9         # edge CPU, Hz
SETUP_DELAY = 0.05       # fixed connection setup per offloaded task, s
CYCLES_MAX = 2.0e9
DATA_MAX = 16.0e6        # bits
BW_MIN, BW_MAX = 1e6, 100e6   # bits/s

P_CPU = 0.9              # W, local computation
P_TX = 1.3               # W, transmitting
P_IDLE = 0.15            # W, waiting for the edge result
BATTERY_J = 60.0         # usable battery energy budget, J

# security buckets (ODG score -> penalty for offloading)
VULN_MED, VULN_HIGH = 0.33, 0.66
PEN_MED, PEN_HIGH = 1.5, 3.0
ODG_GATE = 0.4           # hard gate: objects scoring >= this never leave the device
LOAD_GATE = 0.85         # queue gate: no offloading when edge load is at or above this


def breakeven_bandwidth(data_bits=8e6, cycles=1e9, f_local=F_LOCAL, f_remote=F_REMOTE, setup=0.0):
    """Bandwidth (bit/s) above which offloading is faster than local execution."""
    gain = cycles / f_local - cycles / f_remote - setup
    return np.inf if gain <= 0 else data_bits / gain


R_STAR = breakeven_bandwidth()                          # 13.71 Mbps (paper headline)
R_STAR_SETUP = breakeven_bandwidth(setup=SETUP_DELAY)   # includes the 50 ms setup delay


def task_size(c):
    return c * CYCLES_MAX, c * DATA_MAX


def queue_delay(load):
    return 0.8 * np.asarray(load) ** 2


def t_local(c):
    n, _ = task_size(c)
    return n / F_LOCAL


def t_offload(c, rate, load):
    n, m = task_size(c)
    return SETUP_DELAY + m / rate + n / F_REMOTE + queue_delay(load)


def e_local(c):
    return P_CPU * t_local(c)


def e_offload(c, rate, load):
    n, m = task_size(c)
    return P_TX * (m / rate) + P_IDLE * (SETUP_DELAY + n / F_REMOTE + queue_delay(load))


def security_penalty(v):
    v = np.asarray(v)
    return np.where(v >= VULN_HIGH, PEN_HIGH, np.where(v >= VULN_MED, PEN_MED, 0.0))


def energy_weight(battery):
    """Energy matters more as the battery drains."""
    return 0.5 * (1.0 + 3.0 * (1.0 - np.asarray(battery)))
