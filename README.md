# Security-Aware Adaptive Computation Offloading in Mobile Edge Computing

**Author:** Brindeshwar Sharma (Department of Information Technology, Manipal University Jaipur)

**Status:** research project / preprint draft. It has not been submitted to a journal or conference and has not been peer reviewed. All results come from a simulator written in Python and NumPy; there is no real-device deployment.

## What this is

A simulation of an offloading decision pipeline for Mobile Edge Computing (MEC) in which a mobile device decides, per task, whether to run locally or on an edge server. The pipeline combines:

1. **An analytical bandwidth threshold.** The break-even bandwidth below which local execution is faster (13.71 Mbps for the reference task; 15.0 Mbps once a 50 ms connection setup delay is included).
2. **Q-Learning extended to a DQN.** A tabular Q-agent and a hand-written NumPy DQN (5 inputs: battery, task complexity, vulnerability score, bandwidth, edge load) are trained on the same environment and compared with rule-based baselines over 10 random seeds.
3. **Lyapunov drift-plus-penalty queue control.** An admission rule that admits a task only if `V * benefit > queue length`, with a simple proved bound `Q <= V*b_max + A_max`, compared with greedy admission and a fixed threshold under stochastic multi-user arrivals.
4. **Object Dependency Graph (ODG) vulnerability scoring.** A hard gate that keeps objects with a high sensitive-dependency score on the device, plus a measurement of how often a ratio-based score lets transitively dependent objects leak compared with a stricter "taint" score.

## Key results

Everything below is produced by the code in this repository (`results/data/results.json`). The reward comes from the simulator's own latency, energy and security model, so these numbers show how the policies compare *inside that model*, not on real networks.

| Quantity | Result |
|---|---|
| Break-even bandwidth | 13.71 Mbps (15.0 Mbps with the 50 ms setup delay) |
| Deterministic queue example (arrival 10, capacity 7 per slot, 100 slots) | greedy reaches **297** tasks (3 per slot x 99 slots); the throttled queue stays at most **21** |
| Stochastic queue (Poisson arrivals, 20 seeds, 2,000 slots) | greedy mean queue ~3,025; fixed threshold mean 17.6 (max 32); drift-plus-penalty `V=20` mean 8.7 (max 19) with higher offloading benefit (4.92 vs 4.20 per slot) |
| ODG gate on random graphs (300 graphs) | the ratio gate (0.4) lets 74% of objects that reach sensitive data offload, blocking 11% of non-sensitive objects; a taint gate leaks 0% but blocks 44% |
| Reward per step, always local / rule pipeline | -2.07 / -1.97 |
| Reward per step, DQN with gates / DQN alone / myopic oracle | -1.91 / -1.90 / -1.89 |
| Reward per step, Q-table alone (discount 0.95) | -2.42 (-1.93 at discount 0.0) |
| High-vulnerability tasks offloaded without any gate | Q-table 33%, DQN 0% (all 10 seeds) |

Figures are in [`results/figures/`](results/figures/) (PDF and PNG), tables in [`results/tables/`](results/tables/).

## How to run

Requires Python 3.10+.

```bash
git clone https://github.com/brindeshwar/security-aware-mec-offloading
cd security-aware-mec-offloading
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd simulation
python -m pytest -q                     # unit tests for the quantities stated above (seconds)
python run_experiments.py --quick       # 3 seeds, short training (about 2 minutes)
python run_experiments.py               # full run: 10 seeds (about 15 minutes)
python make_figures.py                  # regenerates results/figures/*.pdf and *.png
python export_latex.py                  # regenerates results/tables/*.tex (numbers used in the paper)
```

`run_experiments.py --quick` writes `results/data/results_quick.json`; point the figure and table scripts at it with `--data`.

## Repository layout

```
simulation/
  mecsim/            config (physical model), env, agents (Q-table, DQN), odg, queueing, workload
  run_experiments.py runs every experiment and writes results/data/results.json
  make_figures.py    draws all figures from the JSON (no numbers typed in)
  export_latex.py    turns the JSON into LaTeX tables/macros
  tests/             pytest checks (13.71 Mbps, 297 vs 21, DPP bound, ODG scores, gates)
results/             data, figures, tables
```

The first version of this repository (a single-script pipeline) is in the git history (commit `9a3794c`). This version replaces it with a modular, tested implementation, multiple seeds and baselines.

## Limitations

- **Simulation only.** No real-device or real-network evaluation. Power and battery values are assumptions, and the reward the agents optimise is defined by the simulator's own cost model.
- **Workload proxy.** N-Queens stands in for CPU-bound tasks; local times are measured on the host CPU and edge times are modelled.
- **Single edge server.** The edge load is a single scalar.
- **Weakly sequential problem.** A decision affects later tasks mainly through the edge load, so the learned policy largely matches a myopic optimum; margins over the rule pipeline are small (about 3% in reward).
- **Soft security constraint.** The learned agents respect the security penalty in these experiments, but only the hard gate guarantees it.
- **ODG.** The example application graph is hand-built, and ratio scores can understate transitive risk.
- **Statistics.** 10 seeds (5 for the discount-factor study); no comparison with algorithms from the literature or real traces.

## License

Code is released under the [MIT License](LICENSE). The paper is a draft preprint; all rights reserved by the author.
