"""Learning agents (tabular Q-learning, NumPy DQN), rule baselines, gates, and evaluation."""
import numpy as np
from . import config as C
from .env import MECEnv, reward_terms

# --------------------------------------------------------------------------- tabular Q-learning
B_BINS, C_BINS, W_EDGES, L_BINS = 5, 4, np.array([0.35, 0.5, 0.6, 0.7, 0.85]), 4


def discretise(obs):
    b, c, v, w, l = obs
    return (min(int(b * B_BINS), B_BINS - 1), min(int(c * C_BINS), C_BINS - 1),
            0 if v < C.VULN_MED else (1 if v < C.VULN_HIGH else 2),
            int(np.searchsorted(W_EDGES, w)), min(int(l * L_BINS), L_BINS - 1))


class TabularQ:
    """Q-table over a discretised version of the same observation the DQN sees."""
    name = "Q-table"

    def __init__(self, rng, alpha=0.1, gamma=0.95):
        self.rng, self.alpha, self.gamma = rng, alpha, gamma
        self.q = np.zeros((B_BINS, C_BINS, 3, len(W_EDGES) + 1, L_BINS, 2))
        self.eps = 1.0

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        return int(np.argmax(self.q[discretise(obs)]))

    def update(self, s, a, r, s2, done):
        i, j = discretise(s), discretise(s2)
        target = r if done else r + self.gamma * self.q[j].max()
        self.q[i + (a,)] += self.alpha * (target - self.q[i + (a,)])


# ---------------------------------------------------------------------------------- NumPy DQN
class DQN:
    """5 -> 64 -> 64 -> 2 ReLU network, experience replay, target network, Adam (all in NumPy)."""
    name = "DQN"

    def __init__(self, rng, hidden=64, lr=1e-3, gamma=0.95, batch=64, capacity=20000,
                 target_every=200, warmup=500):
        self.rng, self.lr, self.gamma, self.batch = rng, lr, gamma, batch
        self.target_every, self.warmup = target_every, warmup
        sizes = [5, hidden, hidden, 2]
        self.p = {}
        for k in range(3):
            self.p[f"W{k}"] = rng.normal(0, np.sqrt(2.0 / sizes[k]), (sizes[k], sizes[k + 1]))
            self.p[f"b{k}"] = np.zeros(sizes[k + 1])
        self.tp = {k: v.copy() for k, v in self.p.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.n_params = int(sum(v.size for v in self.p.values()))
        self.S = np.zeros((capacity, 5), np.float32)
        self.S2 = np.zeros((capacity, 5), np.float32)
        self.A = np.zeros(capacity, np.int64)
        self.R = np.zeros(capacity, np.float32)
        self.D = np.zeros(capacity, np.float32)
        self.cap, self.ptr, self.size, self.steps, self.t_adam = capacity, 0, 0, 0, 0
        self.eps = 1.0

    @staticmethod
    def _forward(p, X):
        z0 = X @ p["W0"] + p["b0"]
        a0 = np.maximum(z0, 0)
        z1 = a0 @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        return a1 @ p["W2"] + p["b2"], (X, z0, a0, z1, a1)

    def qvalues(self, obs):
        return self._forward(self.p, np.atleast_2d(obs))[0]

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        return int(np.argmax(self.qvalues(obs)[0]))

    def update(self, s, a, r, s2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, float(done)
        self.ptr, self.size = (i + 1) % self.cap, min(self.size + 1, self.cap)
        self.steps += 1
        if self.size >= self.warmup:
            self._learn()
        if self.steps % self.target_every == 0:
            self.tp = {k: v.copy() for k, v in self.p.items()}

    def _learn(self):
        idx = self.rng.integers(0, self.size, self.batch)
        X, A, R, X2, D = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]
        y = R + self.gamma * (1 - D) * self._forward(self.tp, X2)[0].max(axis=1)
        q, (X, z0, a0, z1, a1) = self._forward(self.p, X)
        rows = np.arange(self.batch)
        dq = np.zeros_like(q)
        dq[rows, A] = np.clip(q[rows, A] - y, -2.0, 2.0) / self.batch      # Huber-style gradient
        g = {}
        g["W2"], g["b2"] = a1.T @ dq, dq.sum(0)
        d1 = (dq @ self.p["W2"].T) * (z1 > 0)
        g["W1"], g["b1"] = a0.T @ d1, d1.sum(0)
        d0 = (d1 @ self.p["W1"].T) * (z0 > 0)
        g["W0"], g["b0"] = X.T @ d0, d0.sum(0)
        self.t_adam += 1
        b1c, b2c = 1 - 0.9 ** self.t_adam, 1 - 0.999 ** self.t_adam
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            self.p[k] -= self.lr * (self.m[k] / b1c) / (np.sqrt(self.v[k] / b2c) + 1e-8)


# ------------------------------------------------------------------------------ rule baselines
class Policy:
    """Wraps a function (env -> action) with a display name."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __call__(self, env):
        return self.fn(env)


def always_local():
    return Policy("Always local", lambda e: 0)


def always_offload():
    return Policy("Always offload", lambda e: 1)


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return Policy("Random", lambda e: int(rng.integers(2)))


def myopic_oracle():
    """Picks the action with the higher immediate reward using the true model (reference only)."""
    def fn(e):
        r0 = reward_terms(e.battery, e.c, e.v, e.rate, e.load, 0)[0]
        r1 = reward_terms(e.battery, e.c, e.v, e.rate, e.load, 1)[0]
        return int(r1 > r0)
    return Policy("Myopic oracle", fn)


def learned(agent, name=None):
    return Policy(name or agent.name, lambda e: agent.act(e.obs(), greedy=True))


def gated(base, name, bandwidth=True, odg=True, load=True):
    """Hard gates applied before `base`: bandwidth below R*, ODG score >= 0.4, edge load >= 0.85."""
    def fn(e):
        if bandwidth and e.rate < C.R_STAR:
            return 0
        if odg and e.v >= C.ODG_GATE:
            return 0
        if load and e.load >= C.LOAD_GATE:
            return 0
        return base(e)
    return Policy(name, fn)


# ------------------------------------------------------------------------------------ training
def evaluate(policy, env_seed, episodes=200):
    """Roll out `policy` on fresh episodes (same env seed -> same random streams across policies)."""
    env = MECEnv(env_seed)
    tot = dict(steps=0, reward=0.0, latency=0.0, energy=0.0, offload=0, viol_high=0, viol_odg=0,
               high=0, depleted=0, load=0.0)
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            a = policy(env)
            _, r, done, info = env.step(a)
            tot["steps"] += 1
            tot["reward"] += r
            tot["latency"] += info["latency"]
            tot["energy"] += info["energy"]
            tot["offload"] += info["offload"]
            tot["load"] += info["load"]
            if info["v"] >= C.VULN_HIGH:
                tot["high"] += 1
                tot["viol_high"] += info["offload"]
            tot["viol_odg"] += info["offload"] * (info["v"] >= C.ODG_GATE)
        tot["depleted"] += int(info["depleted"])
    n = tot["steps"]
    return dict(reward_per_step=tot["reward"] / n, latency=tot["latency"] / n,
                energy=tot["energy"] / n, offload_rate=tot["offload"] / n,
                viol_high_rate=tot["viol_high"] / max(tot["high"], 1),
                viol_odg_per_task=tot["viol_odg"] / n,
                depleted_frac=tot["depleted"] / episodes, mean_load=tot["load"] / n)


def train(agent, seed, episodes=1500, eval_every=100, eval_eps=20):
    """Train on the environment; return learning curve [(env_steps, greedy reward/step), ...]."""
    env = MECEnv(1000 + seed)
    curve, steps = [], 0
    for ep in range(episodes):
        agent.eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s, steps = s2, steps + 1
        if ep % eval_every == 0 or ep == episodes - 1:
            curve.append((steps, evaluate(learned(agent), 77_000 + seed, eval_eps)["reward_per_step"]))
    return curve


def probe_states(seed, n=20000):
    """States visited by a random policy: the on-distribution test set for security probing."""
    env, pol, out = MECEnv(9000 + seed), random_policy(seed), []
    env.reset()
    while len(out) < n:
        out.append(env.obs())
        _, _, done, _ = env.step(pol(env))
        if done:
            env.reset()
    return np.array(out)


def security_probe(agent, states):
    """Offload rate of the learned policy (no gates) on high-vulnerability states."""
    acts = np.array([agent.act(s, greedy=True) for s in states])
    v, w, c, b, l = states[:, 2], states[:, 3], states[:, 1], states[:, 0], states[:, 4]
    high = v >= C.VULN_HIGH
    favourable = high & (w > 0.6) & (c > 0.5) & (b > 0.2) & (l < 0.5)
    return dict(n_high=int(high.sum()), offload_high=float(acts[high].mean()),
                n_fav=int(favourable.sum()),
                offload_fav=float(acts[favourable].mean()) if favourable.any() else float("nan"),
                offload_all=float(acts.mean()))
