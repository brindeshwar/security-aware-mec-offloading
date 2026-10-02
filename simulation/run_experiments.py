"""Run every experiment and write results/data/results.json.

    python run_experiments.py            # full run (about 10-15 minutes on a laptop)
    python run_experiments.py --quick    # 3 seeds, shorter training (about 2 minutes)
"""
import argparse
import json
import os
import platform
import time

import numpy as np

from mecsim import config as C
from mecsim import odg, queueing, workload
from mecsim.agents import (DQN, TabularQ, always_local, always_offload, evaluate, gated, learned,
                           myopic_oracle, probe_states, random_policy, security_probe, train)

T95 = {2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 19: 2.09, 20: 2.09}


def summarise(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = T95.get(n - 1, 1.96) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(x.mean()), "std": sd, "ci95": float(half), "n": n}


def exp_breakeven():
    rates = np.linspace(1, 100, 400) * 1e6
    c = 0.5
    speedups = np.linspace(1.05, 12, 120)
    return {
        "r_star_mbps": C.R_STAR / 1e6,
        "r_star_with_setup_mbps": C.R_STAR_SETUP / 1e6,
        "rates_mbps": (rates / 1e6).tolist(),
        "t_local": float(C.t_local(c)),
        "t_offload": [float(C.t_offload(c, r, 0.0)) for r in rates],
        "t_offload_no_setup": [float(C.t_offload(c, r, 0.0) - C.SETUP_DELAY) for r in rates],
        "speedups": speedups.tolist(),
        "r_star_vs_speedup": {str(s): [min(C.breakeven_bandwidth(f_remote=C.F_LOCAL * k, setup=s) / 1e6, 400)
                                       for k in speedups] for s in (0.0, 0.05, 0.1)},
    }


def exp_nqueens(quick):
    ns, med = workload.benchmark(range(6, 12 if quick else 14), trials=3)
    out = {"n": ns, "t_local": med, "edge": {}, "crossover": {}}
    for mbps in (10, 50):
        e = [workload.modelled_edge_time(t, mbps * 1e6) for t in med]
        out["edge"][str(mbps)] = e
        out["crossover"][str(mbps)] = next((n for n, t, x in zip(ns, med, e) if x < t), None)
    return out


def exp_odg():
    g = odg.example_app()
    names = list(g.sensitive)
    ratio, taint = g.scores("ratio"), g.scores("taint")
    sweep = odg.leakage_sweep(np.random.default_rng(0), n_graphs=300)
    # leak rate of the paper's gate (ratio score, threshold 0.4) on random graphs
    k = int(np.argmin(np.abs(np.array(sweep["thresholds"]) - C.ODG_GATE)))
    sweep["leak_at_gate"] = sweep["leak_rate"][k]
    sweep["blocked_at_gate"] = sweep["blocked_fraction"][k]
    return {"objects": names, "sensitive": [g.sensitive[n] for n in names],
            "ratio": [ratio[n] for n in names], "taint": [taint[n] for n in names],
            "gate": C.ODG_GATE, "random_graphs": sweep}


def exp_queue(quick):
    det = queueing.deterministic()
    seeds = range(5 if quick else 20)
    slots = 1000 if quick else 2000
    res = {"deterministic": det, "stochastic": {}}
    cfgs = [("greedy", {}), ("threshold", {"theta": 20})] + \
           [(f"dpp_V{V}", {"V": V}) for V in (5, 10, 20, 50, 100)]
    for name, kw in cfgs:
        runs = [queueing.simulate("dpp" if name.startswith("dpp") else name,
                                  np.random.default_rng(s), slots=slots, **kw) for s in seeds]
        res["stochastic"][name] = {
            "mean_q": summarise([r["mean_q"] for r in runs]),
            "max_q": summarise([r["max_q"] for r in runs]),
            "admit_frac": summarise([r["admit_frac"] for r in runs]),
            "benefit_per_slot": summarise([r["benefit_per_slot"] for r in runs]),
            "trace": runs[0]["trace"], "V": kw.get("V")}
    return res



def policy_surfaces(dqn, qtab, n=41):
    """Q(offload) - Q(local) of the DQN on two slices of the state space, plus Q-table argmax maps."""
    g = np.linspace(0, 1, n)

    def diff(state):
        q = dqn.qvalues(np.array(state, np.float32))[0]
        return float(q[1] - q[0])
    # slice A: bandwidth (x) vs vulnerability (y); battery 0.7, complexity 0.8, load 0.3
    A = [[diff([0.7, 0.8, v, w, 0.3]) for w in g] for v in g]
    # slice B: complexity (x) vs battery (y); vulnerability 0.1, bandwidth 0.8, load 0.3
    B = [[diff([b, c, 0.1, 0.8, 0.3]) for c in g] for b in g]
    # slice C: bandwidth (x) vs load (y); battery 0.7, complexity 0.8, vulnerability 0.1
    Cc = [[diff([0.7, 0.8, 0.1, w, l]) for w in g] for l in g]
    return {"grid": g.tolist(), "A": A, "B": B, "C": Cc}

def exp_learning(quick, log):
    seeds = list(range(3 if quick else 10))
    episodes = 400 if quick else 1500
    out = {"episodes": episodes, "seeds": seeds, "curves": {"Q-table": [], "DQN": []},
           "policies": {}, "probe": {"Q-table": [], "DQN": []}, "dqn_params": None,
           "gamma_sweep": {}}
    per_policy = {}
    for s in seeds:
        t0 = time.time()
        q = TabularQ(np.random.default_rng(s))
        d = DQN(np.random.default_rng(s))
        out["dqn_params"] = d.n_params
        out["curves"]["Q-table"].append(train(q, s, episodes, eval_every=max(episodes // 15, 1)))
        out["curves"]["DQN"].append(train(d, s, episodes, eval_every=max(episodes // 15, 1)))
        base = always_offload()
        pols = [always_local(), always_offload(), random_policy(s),
                gated(base, "Threshold only", odg=False, load=False),
                gated(base, "Threshold + ODG gate", load=False),
                gated(base, "Rule pipeline (3 gates)"),
                learned(q, "Q-table"), gated(learned(q), "Q-table + gates"),
                learned(d, "DQN"), gated(learned(d), "DQN + gates (full pipeline)"),
                myopic_oracle()]
        for p in pols:
            per_policy.setdefault(p.name, []).append(evaluate(p, 5000 + s, 60 if quick else 200))
        if s == seeds[0]:
            out["surfaces"] = policy_surfaces(d, q)
        states = probe_states(s, 5000 if quick else 20000)
        out["probe"]["Q-table"].append(security_probe(q, states))
        out["probe"]["DQN"].append(security_probe(d, states))
        log(f"  seed {s}: done in {time.time() - t0:.0f}s")
    for name, rows in per_policy.items():
        out["policies"][name] = {k: summarise([r[k] for r in rows]) for k in rows[0]}
    # discount-factor sensitivity (fewer seeds): the Q-table is much more sensitive to gamma
    gseeds = seeds[:3] if quick else seeds[:5]
    for g in (0.0, 0.5, 0.95):
        rows = {"Q-table": [], "DQN": []}
        for s in gseeds:
            for cls, nm in ((TabularQ, "Q-table"), (DQN, "DQN")):
                a = cls(np.random.default_rng(s), gamma=g)
                train(a, s, episodes, eval_every=episodes)
                rows[nm].append(evaluate(learned(a), 5000 + s, 60 if quick else 200)["reward_per_step"])
        out["gamma_sweep"][str(g)] = {k: summarise(v) for k, v in rows.items()}
        log(f"  gamma {g}: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in rows.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    res = {"meta": {"quick": args.quick, "python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()}}
    log("[1/5] break-even analysis")
    res["breakeven"] = exp_breakeven()
    log("[2/5] N-Queens workload")
    res["nqueens"] = exp_nqueens(args.quick)
    log("[3/5] ODG scoring")
    res["odg"] = exp_odg()
    log("[4/5] queue control")
    res["queue"] = exp_queue(args.quick)
    log("[5/5] learning agents and policy comparison")
    res["learning"] = exp_learning(args.quick, log)
    res["meta"]["runtime_s"] = round(time.time() - t0)
    path = os.path.join(args.out, "results_quick.json" if args.quick else "results.json")
    with open(path, "w") as f:
        json.dump(res, f)
    log(f"wrote {os.path.abspath(path)} in {res['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
