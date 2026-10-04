# Security-Aware Adaptive Computation Offloading in Mobile Edge Computing

**Author:** Brindeshwar Sharma (Department of Information Technology, Manipal University Jaipur)

**Status:** research project / preprint draft. It has not been submitted to a journal or conference and has not been peer reviewed. All results come from a simulator written in Python and NumPy; there is no real-device deployment.

## What this is

A simulation of an offloading decision pipeline for Mobile Edge Computing (MEC) in which a mobile device decides, per task, whether to run locally or on an edge server. The pipeline combines:

1. **An analytical bandwidth threshold.** The break-even bandwidth below which local execution is faster (13.71 Mbps for the reference task; 15.0 Mbps once a 50 ms connection setup delay is included).
2. **Q-Learning extended to a DQN.** A tabular Q-agent and a hand-written NumPy DQN (5 inputs: battery, task complexity, vulnerability score, bandwidth, edge load) are trained on the same environment and compared with rule-based baselines over 10 random seeds.
3. **Lyapunov drift-plus-penalty queue control.** An admission rule that admits a task only if `V * benefit > queue length`, with a simple proved bound `Q <= V*b_max + A_max`, compared with greedy admission and a fixed threshold under stochastic multi-user arrivals.
4. **Object Dependency Graph (ODG) vulnerability scoring.** A hard gate that keeps objects with a high sensitive-dependency score on the device, plus a measurement of how often a ratio-based score lets transitively dependent objects leak compared with a stricter "taint" score.

**Preprint (paper + code archive):** [doi:10.5281/zenodo.23143615](https://doi.org/10.5281/zenodo.23143615) (Zenodo; always resolves to the latest version, v1.0.0 is [10.5281/zenodo.23143616](https://doi.org/10.5281/zenodo.23143616)). Not peer reviewed. The PDF and LaTeX source are also in [`paper/`](paper/); build with `python paper/build.py`.

## Cite

Sharma, B. *Security-Aware Adaptive Computation Offloading in Mobile Edge Computing Using Reinforcement Learning and Deep Q-Networks* (preprint, not peer reviewed). Zenodo, 2026. https://doi.org/10.5281/zenodo.23143615 (see also `CITATION.cff`).

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
results/
  data/              results.json (every number in the paper and README)
  figures/           PDF + PNG, one per figure
  tables/            LaTeX tables and number macros generated from results.json
references/          the cited papers: open-licence PDFs + index of all 29 (DOI, licence)
paper/               LaTeX source and PDF of the preprint draft (IEEE Access class in paper/ieee)
```

The first version of this repository (a single-script pipeline) is in the git history (commit `9a3794c`). This version replaces it with a modular, tested implementation, multiple seeds and baselines.

## Cited papers

The 29 papers cited in the manuscript, in citation order. Papers with an open licence are stored in this repository; the others are owned by their publishers, so only the publisher link is given. Licence and DOI details are in [`references/INDEX.md`](references/INDEX.md).

<!-- REFS:START -->
| # | Paper | Where to read it |
|---|---|---|
| 1 | K. Akherfi, M. Gerndt, H. Harroud, "Mobile Cloud Computing for Computation Offloading: Issues and Challenges", Appl. Comput. Informat., 2018. | [PDF in this repository](references/open_access/01_Mobile_Cloud_Computing_for_Computation_Offloading_Issues_and_Challenges.pdf) (CC-BY-NC-ND) |
| 2 | Y. Mao, C. You, J. Zhang et al., "A Survey on Mobile Edge Computing: The Communication Perspective", IEEE Commun. Surveys Tuts., 2017. | [Publisher page](https://doi.org/10.1109/COMST.2017.2745201) |
| 3 | E. Cuervo et al., "MAUI: Making Smartphones Last Longer with Code Offload", Proc. MobiSys, 2010. | [Publisher page](https://doi.org/10.1145/1814433.1814441) |
| 4 | B.-G. Chun et al., "CloneCloud: Elastic Execution between Mobile Device and Cloud", Proc. EuroSys, 2011. | [Publisher page](https://doi.org/10.1145/1966445.1966473) |
| 5 | L. Lin, X. Liao, H. Jin et al., "Computation Offloading Toward Edge Computing", Proc. IEEE, 2019. | [Publisher page](https://doi.org/10.1109/JPROC.2019.2922285) |
| 6 | J. Zhang, X. Hu, Z. Ning et al., "Energy-Latency Tradeoff for Energy-Aware Offloading in Mobile Edge Computing Networks", IEEE Internet Things J., 2018. | [Publisher page](https://doi.org/10.1109/JIOT.2017.2786343) |
| 7 | F. Wang, J. Xu, X. Wang et al., "Joint Offloading and Computing Optimization in Wireless Powered Mobile-Edge Computing Systems", IEEE Trans. Wireless Commun., 2018. | [Publisher page](https://doi.org/10.1109/TWC.2017.2785305) |
| 8 | S. Barbarossa, S. Sardellitti, P. Di Lorenzo, "Computation Offloading for Mobile Cloud Computing Based on Wide Cross-Layer Optimization", Proc. Future Netw. Mobile Summit, 2013. | [ResearchGate](https://www.researchgate.net/publication/261053174_Computation_offloading_for_mobile_cloud_computing_based_on_wide_cross-layer_optimization) |
| 9 | X. Chen, L. Jiao, W. Li et al., "Efficient Multi-User Computation Offloading for Mobile-Edge Cloud Computing", IEEE/ACM Trans. Netw., 2016. | [Publisher page](https://doi.org/10.1109/TNET.2015.2487344) |
| 10 | H. Zhu, C. Huang, J. Yan, "Vulnerability Evaluation for Securely Offloading Mobile Apps in the Cloud", Proc. IEEE 2nd Int. Conf. Cloud Netw. (CloudNet), 2013. | [Publisher page](https://doi.org/10.1109/CloudNet.2013.6710564) |
| 11 | S. Gan, M. Siew, C. Xu et al., "Differentially Private Deep Q-Learning for Pattern Privacy Preservation in MEC Offloading", Proc. IEEE Int. Conf. Commun. (ICC), 2023. | [Publisher page](https://doi.org/10.1109/ICC45041.2023.10278840) |
| 12 | N. Kiran, C. Pan, S. Wang et al., "Joint Resource Allocation and Computation Offloading in Mobile Edge Computing for SDN Based Wireless Networks", J. Commun. Netw., 2020. | [Publisher page](https://doi.org/10.1109/JCN.2019.000046) |
| 13 | M. Tang, V. W. S. Wong, "Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing Systems", IEEE Trans. Mobile Comput., 2022. | [Publisher page](https://doi.org/10.1109/TMC.2020.3036871) |
| 14 | L. Huang et al., "Deep Reinforcement Learning-Based Joint Task Offloading and Bandwidth Allocation for Multi-User Mobile Edge Computing", Digit. Commun. Netw., 2019. | [PDF in this repository](references/open_access/14_Deep_Reinforcement_Learning-Based_Joint_Task_Offloading_and_Bandwidth_Allocation_for_Multi-User_Mobile_Edge.pdf) (CC-BY-NC-ND) |
| 15 | D. Kovachev, R. Klamma, "Framework for Computation Offloading in Mobile Cloud Computing", Int. J. Interact. Multimedia Artif. Intell., 2012. | [PDF in this repository](references/open_access/15_Framework_for_Computation_Offloading_in_Mobile_Cloud_Computing.pdf) (CC-BY) |
| 16 | H. He, X. Yang, X. Mi et al., "Multi-Agent Deep Reinforcement Learning Based Dynamic Task Offloading in a Device-to-Device Mobile-Edge Computing Network to Minimize Average Task Delay with Deadline Constraints", Sensors, 2024. | [PDF in this repository](references/open_access/16_Multi-Agent_Deep_Reinforcement_Learning_Based_Dynamic_Task_Offloading_in_a_Device-to-Device_Mobile-Edge.pdf) (CC-BY) |
| 17 | S. Bi et al., "Lyapunov-Guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks", IEEE Trans. Wireless Commun., 2021. | [Publisher page](https://doi.org/10.1109/TWC.2021.3085319) |
| 18 | S. Bae, S. Han, Y. Sung, "A Reinforcement Learning Formulation of the Lyapunov Optimization: Application to Edge Computing Systems with Queue Stability", arXiv:2012.07279, 2020. | [arXiv](https://arxiv.org/abs/2012.07279) |
| 19 | X. Zhao, G. Huang, J. Jiang et al., "Task Offloading of Cooperative Intrusion Detection System Based on Deep Q Network in Mobile Edge Computing", Expert Syst. Appl., 2022. | [Publisher page](https://doi.org/10.1016/j.eswa.2022.117860) |
| 20 | C. Yang, X. Xu, X. Zhou et al., "Deep Q Network--Driven Task Offloading for Efficient Multimedia Data Analysis in Edge Computing--Assisted IoV", ACM Trans. Multimedia Comput. Commun. Appl., 2022. | [Publisher page](https://doi.org/10.1145/3548687) |
| 21 | J. Anand, B. Karthikeyan, "Adaptive and Intelligent Customized Deep Q-Network for Energy-Efficient Task Offloading in Mobile Edge Computing Environments", Sci. Rep., 2026. | [PDF in this repository](references/open_access/21_Adaptive_and_Intelligent_Customized_Deep_Q-Network_for_Energy-Efficient_Task_Offloading_in_Mobile_Edge.pdf) (CC-BY-NC-ND) |
| 22 | C. Liu, H. Wang, M. Zhao et al., "Dependency-Aware Online Task Offloading Based on Deep Reinforcement Learning for IoV", J. Cloud Comput., 2024. | [PDF in this repository](references/open_access/22_Dependency-Aware_Online_Task_Offloading_Based_on_Deep_Reinforcement_Learning_for_IoV.pdf) (CC-BY) |
| 23 | J. Fang, D. Qu, H. Chen et al., "Dependency-Aware Dynamic Task Offloading Based on Deep Reinforcement Learning in Mobile-Edge Computing", IEEE Trans. Netw. Service Manag., 2024. | [PDF in this repository](references/open_access/23_Dependency-Aware_Dynamic_Task_Offloading_Based_on_Deep_Reinforcement_Learning_in_Mobile-Edge_Computing.pdf) (CC-BY-NC-ND) |
| 24 | Y. Qin, J. Chen, L. Jin et al., "Task Offloading Optimization in Mobile Edge Computing Based on a Deep Reinforcement Learning Algorithm Using Density Clustering and Ensemble Learning", Sci. Rep., 2025. | [PDF in this repository](references/open_access/24_Task_Offloading_Optimization_in_Mobile_Edge_Computing_Based_on_a_Deep_Reinforcement_Learning_Algorithm.pdf) (CC-BY-NC-ND) |
| 25 | J. Chen, L. Jin, R. Yao et al., "Deep Reinforcement Learning Method for Task Offloading in Mobile Edge Computing Networks Based on Parallel Exploration with Asynchronous Training", Mobile Netw. Appl., 2025. | [Publisher page](https://doi.org/10.1007/s11036-024-02397-7) |
| 26 | X. Zhang, C. Fang, Z. Bai et al., "Temporal Dependency Task Offloading via Deep Reinforcement Learning for Mobile Edge Computing", Peer-to-Peer Netw. Appl., 2025. | [Publisher page](https://doi.org/10.1007/s12083-025-02101-w) |
| 27 | J. Sun, W. Zhang, M. Han et al., "Deep Reinforcement Learning-Based Adaptive Task Offloading for Mobile Edge Computing", J. Supercomput., 2026. | [Publisher page](https://doi.org/10.1007/s11227-026-08492-8) |
| 28 | A. Jalal, U. Farooq, I. Rabbi et al., "Towards Intelligent Edge Computing Through Reinforcement Learning Based Offloading in Public Edge as a Service", Sci. Rep., 2026. | [PDF in this repository](references/open_access/28_Towards_Intelligent_Edge_Computing_Through_Reinforcement_Learning_Based_Offloading_in_Public_Edge_as_a.pdf) (CC-BY-NC-ND) |
| 29 | J. Wang, J. Hu, G. Min et al., "Dependent Task Offloading for Edge Computing Based on Deep Reinforcement Learning", IEEE Trans. Comput., 2022. | [Publisher page](https://doi.org/10.1109/TC.2021.3131040) |
<!-- REFS:END -->


## Limitations

- **Simulation only.** No real-device or real-network evaluation. Power and battery values are assumptions, and the reward the agents optimise is defined by the simulator's own cost model.
- **Workload proxy.** N-Queens stands in for CPU-bound tasks; local times are measured on the host CPU and edge times are modelled.
- **Single edge server.** The edge load is a single scalar.
- **Weakly sequential problem.** A decision affects later tasks mainly through the edge load, so the learned policy largely matches a myopic optimum; margins over the rule pipeline are small (about 3% in reward).
- **Soft security constraint.** The learned agents respect the security penalty in these experiments, but only the hard gate guarantees it.
- **ODG.** The example application graph is hand-built, and ratio scores can understate transitive risk.
- **Statistics.** 10 seeds (5 for the discount-factor study); no comparison with algorithms from the literature or real traces.

## License

Code is released under the [MIT License](LICENSE). The paper (PDF) is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as a preprint draft; the cited open-access papers in `references/open_access/` keep their own licences (see `references/INDEX.md`).
