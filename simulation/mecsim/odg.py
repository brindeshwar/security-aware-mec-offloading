"""Object Dependency Graph (ODG) vulnerability scoring."""
from collections import defaultdict
import numpy as np


class ODG:
    def __init__(self):
        self.sensitive = {}
        self.edges = defaultdict(set)

    def add(self, name, sensitive=False):
        self.sensitive[name] = sensitive

    def depend(self, src, dst):
        self.edges[src].add(dst)

    def reachable(self, name):
        seen, stack = set(), [name]
        while stack:
            for nb in self.edges.get(stack.pop(), ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        seen.discard(name)
        return seen

    def ratio_score(self, name):
        """Paper's score: sensitive reachable objects / all reachable objects (1.0 if sensitive)."""
        if self.sensitive[name]:
            return 1.0
        r = self.reachable(name)
        return sum(self.sensitive[x] for x in r) / len(r) if r else 0.0

    def taint_score(self, name):
        """Conservative variant: 1.0 if the object is sensitive or can reach any sensitive object."""
        if self.sensitive[name]:
            return 1.0
        return 1.0 if any(self.sensitive[x] for x in self.reachable(name)) else 0.0

    def scores(self, mode="ratio"):
        f = self.ratio_score if mode == "ratio" else self.taint_score
        return {n: f(n) for n in self.sensitive}


def example_app():
    """Hand-built healthcare/fitness app used in the paper (12 objects)."""
    g = ODG()
    for n in ("getUserID", "getHeartRate", "getMedicalRecord", "getLocation"):
        g.add(n, True)
    for n in ("calculateBMI", "sortLeaderboard", "renderChart", "encryptData",
              "analyzeHealth", "buildReport", "syncToCloud", "runAIInference"):
        g.add(n, False)
    for a, b in [("analyzeHealth", "getHeartRate"), ("analyzeHealth", "calculateBMI"),
                 ("buildReport", "analyzeHealth"), ("buildReport", "getUserID"),
                 ("syncToCloud", "buildReport"), ("syncToCloud", "sortLeaderboard"),
                 ("syncToCloud", "encryptData"), ("runAIInference", "analyzeHealth"),
                 ("runAIInference", "calculateBMI"), ("renderChart", "sortLeaderboard")]:
        g.depend(a, b)
    return g


def random_app(rng, n=30, p=0.12, frac_sensitive=0.2):
    """Random dependency DAG for stress-testing the gate (objects depend only on later ones)."""
    g = ODG()
    names = [f"o{i}" for i in range(n)]
    sens = rng.random(n) < frac_sensitive
    for nm, s in zip(names, sens):
        g.add(nm, bool(s))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.depend(names[i], names[j])
    return g


def leakage_sweep(rng, n_graphs=300, thresholds=None, **kw):
    """Sweep the ratio-gate threshold on random graphs.

    leak rate        = share of objects that reach sensitive data yet are allowed to offload
    blocked fraction = share of non-sensitive objects forced to stay local
    The taint gate corresponds to leak rate 0 at the blocked fraction `taint_blocked_fraction`.
    """
    thresholds = np.linspace(0.02, 1.0, 50) if thresholds is None else np.asarray(thresholds)
    leaked = np.zeros(len(thresholds))
    blocked = np.zeros(len(thresholds))
    tainted_total, objs_total = 0, 0
    for _ in range(n_graphs):
        g = random_app(rng, **kw)
        ns = [x for x in g.sensitive if not g.sensitive[x]]
        if not ns:
            continue
        rs = np.array([g.ratio_score(x) for x in ns])
        tainted = np.array([g.taint_score(x) for x in ns]) > 0
        tainted_total += tainted.sum()
        objs_total += len(ns)
        for k, tau in enumerate(thresholds):
            offload = rs < tau
            leaked[k] += (offload & tainted).sum()
            blocked[k] += (~offload).sum()
    return {"thresholds": thresholds.tolist(),
            "leak_rate": (leaked / max(tainted_total, 1)).tolist(),
            "blocked_fraction": (blocked / max(objs_total, 1)).tolist(),
            "taint_blocked_fraction": float(tainted_total / max(objs_total, 1))}
