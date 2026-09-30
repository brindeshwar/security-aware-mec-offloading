# Security-Aware Adaptive Computation Offloading in Mobile Edge Computing Using Reinforcement Learning and Deep Q-Networks

**Author:** Brindeshwar Sharma (Department of Information Technology, Manipal University Jaipur)

**Status:** research project / preprint draft, formatted in the IEEE Access template. It has not been submitted to any journal or conference and has not been peer reviewed. The evaluation is simulation-only (Python, NumPy, Matplotlib).

## Abstract

Mobile Edge Computing (MEC) lets resource-constrained devices offload heavy computation to nearby edge servers. Most offloading work optimises latency, energy or resource allocation and treats security and multi-user queue behaviour separately. This project builds a simulated decision pipeline that combines an analytical bandwidth threshold, N-Queens-based workload classification, Q-Learning, a Lyapunov-inspired queue controller, Object Dependency Graph (ODG) vulnerability scoring, and a Deep Q-Network (DQN) over a continuous five-dimensional state (battery, task complexity, vulnerability score, bandwidth, server load). Everything is evaluated in a simplified simulator; there is no real-device deployment.

Preprint link: _to be added._

## Contributions

1. **Analytical bandwidth threshold.** A closed-form break-even bandwidth below which local execution beats offloading (13.71 Mbps for the simulated parameters).
2. **Q-Learning extended to a DQN.** A tabular Q-Learning agent (battery x task type) is extended to a hand-written NumPy DQN (5 inputs, two hidden layers of 64 ReLU units, 2 actions, experience replay, target network) that works on continuous states.
3. **Queue stability control.** A Lyapunov-inspired admission controller that throttles arrivals when the edge queue passes a threshold, compared against a greedy "accept everything" policy in a multi-user queue simulation.
4. **ODG vulnerability scoring.** Each application object gets a score from its sensitive dependencies; objects at or above a threshold are forced to run locally, which keeps location, medical-record, biometric and user-ID objects on the device.

## Results summary

All numbers below come from the code in this repo. Raw output is in [`results/`](results/).

| Item | Result | Notes |
|---|---|---|
| Break-even bandwidth | **13.71 Mbps** | `R* = D * f_local * f_remote / (C * (f_remote - f_local))` with C = 1e9 cycles, D = 8 Mb, f_local = 1.2 GHz, f_remote = 4.0 GHz. The 50 ms network setup delay is not part of this formula. |
| Edge queue length | greedy **297** vs. controlled **21** (max) | 100 time slots, arrival 10/slot, capacity 7/slot. The greedy queue grows by 3 per slot, so 297 is 3 x 99 and is a property of the chosen parameters and horizon. The controller admits 4 tasks/slot whenever the queue is at or above 20. |
| Sensitive objects kept local | **100%** of the sensitive data objects (user ID, medical record, location, heart rate) | Enforced by a hard rule (vulnerability >= 0.4 in code), so this is by construction. |
| DQN offload rate on high-vulnerability states (> 0.66) | **1.1%** (n = 6,722 random states, seed 42) | The DQN learns this through a reward penalty and is not a guarantee. It is 5.3% when the other conditions favour offloading. See [`results/dqn_blocking.txt`](results/dqn_blocking.txt). |
| Q-Learning convergence | about 350 episodes | Read off the reward curve (`results/figures/fig03_phase3_convergence.png`); the pipeline's summary prints it as a fixed string, not a computed value. |

Figures are in [`results/figures/`](results/figures/), and the combined dashboard is [`results/adaptive_offloading_full_pipeline.png`](results/adaptive_offloading_full_pipeline.png).

## How to run the simulation

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd simulation
python adaptive_offloading_full_pipeline.py      # all phases, about 10-15 s; saves a PNG dashboard
python evaluate_dqn_blocking.py                  # measures how often the DQN offloads high-vulnerability tasks
```

On a headless machine, set `MPLBACKEND=Agg` first to skip the plot window. Tested on Windows 11 with Python 3.12, NumPy 2.4 and Matplotlib 3.10.

## Repository layout

```
simulation/   full pipeline (phases 1-5) and the DQN blocking evaluation
results/      pipeline output, DQN blocking measurement, figures
```

## Limitations / Future work

**Limitations**

- **Simulation only.** No real-device or real-network deployment; all timings, energy and rewards come from a simplified model.
- **N-Queens is a workload proxy.** It stands in for CPU-bound tasks. In the Phase 2 benchmark the edge time is *modelled* as local time / 10 plus the 50 ms network setup delay. It is not measured on a remote machine.
- **Single edge server.** No multi-server orchestration or load balancing.
- **Rule-based reward.** The Q-Learning and DQN agents are trained on a hand-written reward function in a synthetic environment, so they largely learn to reproduce those rules. There is no comparison against baseline policies or real traces.
- **Queue controller.** It is a threshold-based admission controller motivated by Lyapunov drift-plus-penalty ideas. It does not compute the drift-plus-penalty objective or prove stability.
- **ODG is hand-built.** The dependency graph is a small healthcare-app example written by hand, not extracted from real application code. Blocking depends on that labelling and on the chosen threshold.
- **No statistical analysis.** Results are from single seeded runs without confidence intervals.

**Future work**

- Real-device deployment (Android/iOS) and measurement on real edge hardware.
- Multi-server edge orchestration.
- Baselines and ablations (e.g. threshold-only, greedy, tabular vs. DQN) on real workload traces.
- A proper drift-plus-penalty controller with analysis.
- Runtime ODG extraction from real applications.
- Federated RL and explainable RL for security-critical settings.

## License

Code is released under the [MIT License](LICENSE). The paper is a draft preprint; all rights reserved by the author.
