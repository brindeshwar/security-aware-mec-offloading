"""Regenerate the figures that summarise measured results (no hand-written claims).

Writes into ../results/figures:
  fig02_nqueens_benchmark.png   N-Queens local time vs. *modelled* edge time
  fig05_queue_control.png       greedy admission vs. threshold admission controller
  fig13_agent_comparison.png    Q-table vs. DQN size/inputs (computed from the code)
  fig03_qlearning_progress.png, fig04_tabular_policies.png, fig12_dqn_policy_surfaces.png
  fig14_pipeline_gates.png      decision pipeline (gate order as implemented)
and ../results/nqueens_benchmark.txt.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import adaptive_offloading_full_pipeline as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def fig_nqueens():
    ns = list(range(1, 13))
    local, remote, cross = P.phase2_benchmark(ns, repetitions=300)
    with open(os.path.join(OUT, "nqueens_benchmark.txt"), "w", encoding="utf-8") as f:
        f.write("N-Queens, 30 repetitions per N. Local time is measured; edge time is MODELLED as\n"
                "local_time / 10 + 0.05 s network setup (not measured on a remote machine).\n\n")
        for n, l, r in zip(ns, local, remote):
            f.write(f"N={n}  local={l:.4f}s  modelled_edge={r:.4f}s\n")
        f.write(f"\nFirst N where modelled edge time < local time: {cross}\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    x, w = np.arange(len(ns)), 0.38
    ax.bar(x - w / 2, local, w, label="Local (measured)", color="#d1495b")
    ax.bar(x + w / 2, remote, w, label="Edge (modelled: local/10 + 50 ms)", color="#2e86ab")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_xlabel("N-Queens problem size")
    ax.set_ylabel("Time (s, log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig02_nqueens_benchmark.png"), dpi=200)
    plt.close(fig)


def fig_queue():
    greedy, ctrl, throttle, forced = P.phase4_simulate()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(greedy, "--", color="#d1495b", lw=2, label=f"Greedy admission (final {greedy[-1]})")
    ax.plot(ctrl, "-", color="#2e86ab", lw=2, label=f"Threshold controller (max {max(ctrl)})")
    ax.axhline(P.QUEUE_THRESHOLD, color="k", ls=":", lw=1.2, label=f"Threshold = {P.QUEUE_THRESHOLD}")
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Queued tasks")
    ax.set_title(f"Arrival {P.ARRIVAL_RATE}/slot, capacity {P.SERVER_CAPACITY}/slot, "
                 f"throttled admission {P.THROTTLE_RATE}/slot, {P.TIME_SLOTS} slots", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig05_queue_control.png"), dpi=200)
    plt.close(fig)


def fig_agents():
    agent = P.DQNAgent().main
    n_params = sum(a.size for a in (agent.W1, agent.b1, agent.W2, agent.b2, agent.W3, agent.b3))
    q3 = P.BATTERY_STATES * 2
    q5 = P.BATTERY_STATES * 2 * P.VULN_BUCKETS
    rows = [
        ("State representation", "Discrete", "Discrete", "Continuous"),
        ("State inputs", "battery, task type", "battery, task type,\nvulnerability bucket",
         "battery, complexity, vulnerability,\nbandwidth, server load"),
        ("Distinct states", str(q3), str(q5), "n/a (continuous)"),
        ("Learned parameters", f"{q3 * 2} Q-values", f"{q5 * 2} Q-values", f"{n_params:,} weights"),
        ("Training", f"{P.EPISODES_P3} updates", f"{P.EPISODES_P5} updates", f"{P.EPISODES_DQN} episodes"),
        ("Evaluation", "simulation only", "simulation only", "simulation only"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["", "Q-table (Phase 3)", "Q-table (Phase 5b)", "DQN (Phase 5c)"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_agent_comparison.png"), dpi=200)
    plt.close(fig)
    return n_params


def fig_pipeline():
    gates = [
        ("Task\narrives", "#555555"),
        ("Gate 1\nbandwidth >\n13.71 Mbps?", "#1f4e79"),
        ("Gate 2\ntask heavy\n(N-Queens)?", "#1f4e79"),
        ("Gate 3\nODG score <\nthreshold?", "#8b0000"),
        ("Gate 4\nQ-Learning /\nDQN policy", "#5b3f7d"),
        ("Gate 5\nedge queue\nbelow limit?", "#1a5e1a"),
        ("Offload\nto edge", "#2a9d2a"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, len(gates))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (t, c) in enumerate(gates):
        ax.text(i + 0.5, 0.6, t, ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="none"))
        if i < len(gates) - 1:
            ax.annotate("", xy=(i + 1.08, 0.6), xytext=(i + 0.92, 0.6), arrowprops=dict(arrowstyle="->"))
        if 1 <= i <= 5:
            ax.text(i + 0.5, 0.08, "otherwise: run locally", ha="center", fontsize=7, color="#a00000")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig14_pipeline_gates.png"), dpi=200)
    plt.close(fig)


def fig_tabular(thr):
    q3, r3 = P.phase3_train(thr)
    q5, r5 = P.phase5b_train(thr)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(r3, color="#2e86ab", label=f"Q-table, battery x task type ({P.EPISODES_P3} updates)")
    ax.plot(r5, color="#8b0000", label=f"Q-table + vulnerability bucket ({P.EPISODES_P5} updates)")
    ax.set_xlabel("Training update (one transition each)")
    ax.set_ylabel("Sum of all Q-table entries")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig03_qlearning_progress.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    panels = [("Phase 3 (no security)", np.argmax(q3, axis=2))] +              [(f"Phase 5b, {n} vulnerability", np.argmax(q5[:, :, b, :], axis=2))
              for b, n in enumerate(["LOW", "MED", "HIGH"])]
    for ax, (t, pol) in zip(axes, panels):
        ax.imshow(pol, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_title(t, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["light", "heavy"], fontsize=7)
        ax.set_yticks(range(P.BATTERY_STATES))
        ax.set_yticklabels(range(P.BATTERY_STATES), fontsize=6)
        ax.set_ylabel("battery level", fontsize=7)
    fig.suptitle("argmax policy: green = offload, red = local (ties resolve to local)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig04_tabular_policies.png"), dpi=200)
    plt.close(fig)


def fig_dqn_surfaces(agent):
    g = np.linspace(0, 1, 41)

    def diff(state):
        q = agent.main.forward(state)[0]
        return q[1] - q[0]
    A = np.array([[diff(np.array([0.7, 0.8, v, b, 0.3], dtype=np.float32)) for b in g] for v in g])
    B = np.array([[diff(np.array([bat, c, 0.1, 0.8, 0.3], dtype=np.float32)) for c in g] for bat in g])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, M, xl, yl, t in [
        (axes[0], A, "bandwidth (normalised)", "vulnerability score",
         "battery 0.7, complexity 0.8, server load 0.3"),
        (axes[1], B, "task complexity", "battery level",
         "vulnerability 0.1, bandwidth 0.8, server load 0.3")]:
        im = ax.imshow(M, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=8)
        fig.colorbar(im, ax=ax, label="Q(offload) - Q(local)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig12_dqn_policy_surfaces.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_nqueens()
    fig_queue()
    print("DQN parameters:", fig_agents())
    fig_pipeline()
    thr = P.phase1_compute_threshold()[0]
    fig_tabular(thr)
    agent, *_ = P.phase5c_train()
    fig_dqn_surfaces(agent)
    print("figures written to", os.path.abspath(FIG))
