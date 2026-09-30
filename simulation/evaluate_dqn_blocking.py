"""Measure how often the trained DQN offloads high-vulnerability tasks.

The pipeline's hard ODG gate blocks sensitive objects by construction. The DQN,
however, only *learns* to avoid offloading them through the reward penalty, so
this script measures that rate instead of assuming it.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import adaptive_offloading_full_pipeline as P

agent, *_ = P.phase5c_train()
S = np.random.default_rng(0).uniform(0, 1, (20000, 5)).astype(np.float32)
acts = np.array([agent.decide(s)[0] for s in S])  # 1 = offload
vuln, bw, cx, bat = S[:, 2], S[:, 3], S[:, 1], S[:, 0]

print("\nDQN offload rate on 20,000 uniformly sampled states "
      "(state = [battery, complexity, vulnerability, bandwidth, server_load])")
for label, mask in [
    ("vulnerability > 0.66 (HIGH)", vuln > 0.66),
    ("vulnerability > 0.33 (MED+HIGH)", vuln > 0.33),
    ("HIGH vuln AND otherwise favourable to offload", (vuln > 0.66) & (bw > 0.5) & (cx > 0.5) & (bat > 0.2)),
]:
    print(f"  {label:<48} n={mask.sum():>6}  offloaded: {acts[mask].mean():.1%}")
