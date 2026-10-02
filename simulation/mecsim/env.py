"""Sequential MEC offloading environment with a physics-based reward.

State (observation, all in [0, 1]): [battery, complexity, vulnerability, bandwidth, edge load]
  - battery drains by the energy each task uses; the episode ends at 0 (with a penalty)
  - complexity c ~ U(0.05, 1) and vulnerability v ~ Beta(1, 2) are drawn fresh for every task
  - bandwidth follows a log-random walk between 1 and 100 Mbps (observed on a log scale)
  - edge load rises with offloaded work and decays otherwise, so offloading has a cost to later tasks
Reward = -(latency + w(battery) * energy) - security penalty if the task is offloaded.
"""
import numpy as np
from . import config as C

LN_MIN, LN_MAX = np.log(C.BW_MIN), np.log(C.BW_MAX)
DEPLETION_PENALTY = 5.0


def reward_terms(battery, c, v, rate, load, action):
    """Vectorised reward, latency and energy for `action` (0 = local, 1 = offload)."""
    action = np.asarray(action)
    T = np.where(action == 1, C.t_offload(c, rate, load), C.t_local(c))
    E = np.where(action == 1, C.e_offload(c, rate, load), C.e_local(c))
    r = -(T + C.energy_weight(battery) * E) - action * C.security_penalty(v)
    return r, T, E


class MECEnv:
    obs_dim, n_actions = 5, 2

    def __init__(self, seed=0, max_steps=60):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps

    # ------------------------------------------------------------------ state
    def reset(self):
        r = self.rng
        self.battery = r.uniform(0.3, 1.0)
        self.load = r.uniform(0.0, 0.6)
        self.log_rate = r.uniform(LN_MIN, LN_MAX)
        self.t = 0
        self._new_task()
        return self.obs()

    def _new_task(self):
        self.c = self.rng.uniform(0.05, 1.0)
        self.v = self.rng.beta(1.0, 2.0)

    @property
    def rate(self):
        return float(np.exp(self.log_rate))

    def obs(self):
        w = (self.log_rate - LN_MIN) / (LN_MAX - LN_MIN)
        return np.array([self.battery, self.c, self.v, w, self.load], dtype=np.float32)

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = int(action)
        r, T, E = reward_terms(self.battery, self.c, self.v, self.rate, self.load, a)
        info = dict(latency=float(T), energy=float(E), offload=a, v=self.v, load=self.load)
        self.battery -= float(E) / C.BATTERY_J
        self.load = float(np.clip(0.9 * self.load + 0.15 * a * (0.4 + self.c)
                                  + self.rng.normal(0, 0.02), 0.0, 1.0))
        self.log_rate = float(np.clip(self.log_rate + self.rng.normal(0, 0.25), LN_MIN, LN_MAX))
        self.t += 1
        depleted = self.battery <= 0.0
        r = float(r) - (DEPLETION_PENALTY if depleted else 0.0)
        info["depleted"] = depleted
        self._new_task()
        done = depleted or self.t >= self.max_steps
        return self.obs(), r, done, info
