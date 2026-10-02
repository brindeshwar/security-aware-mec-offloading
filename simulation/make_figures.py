"""Draw every figure in the paper from results/data/results.json (no numbers are typed in here).

    python make_figures.py [--data ../results/data/results.json]

Writes vector PDF + 300-dpi PNG per figure into ../results/figures.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from mecsim import config as C

# Reference categorical palette (slots 1-4) + neutrals
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    "legend.frameon": False,
})
W1, W2 = 3.5, 7.16     # IEEE single / double column width, inches
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def panel(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


def fig_breakeven(R):
    b = R["breakeven"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"wspace": 0.28})
    x = np.array(b["rates_mbps"])
    a.plot(x, b["t_offload"], color=BLUE, label="Offload (with 50 ms setup)")
    a.plot(x, b["t_offload_no_setup"], color=AQUA, ls="--", label="Offload (no setup delay)")
    a.axhline(b["t_local"], color=ORANGE, label="Local execution")
    for xv, col in ((b["r_star_mbps"], AQUA), (b["r_star_with_setup_mbps"], BLUE)):
        a.axvline(xv, color=col, ls=":", lw=1)
    a.annotate(f"{b['r_star_mbps']:.2f} Mbps", (b["r_star_mbps"], 0.12), xytext=(-4, 0),
               textcoords="offset points", ha="right", color=INK2, fontsize=7.5)
    a.annotate(f"{b['r_star_with_setup_mbps']:.1f} Mbps", (b["r_star_with_setup_mbps"], 0.12), xytext=(4, 0),
               textcoords="offset points", ha="left", color=INK2, fontsize=7.5)
    a.set(xlabel="Uplink bandwidth (Mbps)", ylabel="Task completion time (s)", xlim=(0, 60), ylim=(0, 1.6))
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
    panel(a, "(a)")
    sp = np.array(b["speedups"])
    for key, col, ls, lab in (("0.0", AQUA, "--", "no setup delay"), ("0.05", BLUE, "-", "50 ms setup"),
                              ("0.1", VIOLET, "-.", "100 ms setup")):
        c.plot(sp, b["r_star_vs_speedup"][key], color=col, ls=ls, label=lab)
    c.axvline(C.F_REMOTE / C.F_LOCAL, color=INK2, ls=":", lw=1)
    c.annotate("simulated edge\n(4.0 / 1.2 GHz)", (C.F_REMOTE / C.F_LOCAL, 85), xytext=(4, 0),
               textcoords="offset points", fontsize=7.5, color=INK2)
    c.set(xlabel="Edge speed-up $f_{remote}/f_{local}$", ylabel="Break-even bandwidth $R^*$ (Mbps)",
          xlim=(1, 12), ylim=(0, 100))
    c.legend(loc="upper right")
    panel(c, "(b)")
    save(fig, "fig01_breakeven")


def fig_nqueens(R):
    q = R["nqueens"]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    n = q["n"]
    ax.semilogy(n, q["t_local"], "o-", color=ORANGE, ms=3.5, label="Local (measured)")
    ax.semilogy(n, q["edge"]["50"], "s-", color=BLUE, ms=3.5, label="Edge, 50 Mbps (modelled)")
    ax.semilogy(n, q["edge"]["10"], "^--", color=VIOLET, ms=3.5, label="Edge, 10 Mbps (modelled)")
    for key, col, y0 in (("50", BLUE, 3e-4), ("10", VIOLET, 3e-3)):
        cx = q["crossover"][key]
        if cx:
            ax.axvline(cx, color=col, ls=":", lw=1)
            ax.annotate(f"N = {cx}", (cx, y0), xytext=(-3, 0), textcoords="offset points", ha="right",
                        fontsize=7, color=col)
    ax.set(xlabel="N-Queens board size N", ylabel="Time to count all solutions (s)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, "fig02_nqueens")


def fig_pipeline(R):
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    nodes = [("Task\narrives", INK2), ("Gate 1\nbandwidth\n$\\geq R^*$ ?", BLUE),
             ("Gate 2\nODG score\n$< 0.4$ ?", RED), ("Gate 3\nlearned policy\n(Q-table / DQN)", VIOLET),
             ("Gate 4\nedge load\n$< 0.85$ ?", AQUA), ("Offload\nto edge", INK2)]
    xs = np.linspace(8, 92, len(nodes))
    for i, ((txt, col), x) in enumerate(zip(nodes, xs)):
        ax.add_patch(FancyBboxPatch((x - 6.0, 18), 12.0, 14, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=col, ec="none"))
        ax.text(x, 25, txt, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
        if i < len(nodes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 7.4, 25), xytext=(x + 7.4, 25),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
        if 1 <= i <= 4:
            ax.annotate("", xy=(x, 8), xytext=(x, 16.2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, ls="--"))
            ax.text(x, 4, "chooses local" if i == 3 else "fails: run locally", ha="center", va="center",
                    fontsize=7, color=INK2)
    save(fig, "fig03_pipeline")


def fig_odg(R):
    o = R["odg"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.5})
    order = np.argsort([-(r + 0.001 * t) for r, t in zip(o["ratio"], o["taint"])], kind="stable")
    names = [o["objects"][i] for i in order]
    y = np.arange(len(names))
    a.barh(y - 0.18, [o["ratio"][i] for i in order], 0.34, color=BLUE, label="Ratio score (paper)")
    a.barh(y + 0.18, [o["taint"][i] for i in order], 0.34, color=ORANGE, label="Taint score")
    a.axvline(o["gate"], color=INK, ls="--", lw=1)
    a.text(o["gate"] + 0.01, -0.9, "gate = 0.4", fontsize=7.5, va="bottom")
    a.set_yticks(y)
    a.set_yticklabels(names, fontsize=7, style="italic")
    a.invert_yaxis()
    a.set(xlabel="Vulnerability score", xlim=(0, 1.08))
    a.legend(loc="lower right")
    a.grid(axis="y", visible=False)
    panel(a, "(a)")
    s = o["random_graphs"]
    c.plot(np.array(s["blocked_fraction"]) * 100, np.array(s["leak_rate"]) * 100, color=BLUE,
           label="Ratio gate, threshold swept")
    k = int(np.argmin(np.abs(np.array(s["thresholds"]) - o["gate"])))
    c.plot(s["blocked_fraction"][k] * 100, s["leak_rate"][k] * 100, "o", color=BLUE, ms=6, mec="white", mew=1.2,
           zorder=5)
    c.annotate("paper gate (0.4):\n%.0f%% leak" % (s["leak_rate"][k] * 100),
               (s["blocked_fraction"][k] * 100, s["leak_rate"][k] * 100), xytext=(8, 4),
               textcoords="offset points", fontsize=7.5, color=INK2)
    c.plot(s["taint_blocked_fraction"] * 100, 0, "s", color=ORANGE, ms=6, mec="white", mew=1.2, zorder=5)
    c.annotate("taint gate: 0% leak", (s["taint_blocked_fraction"] * 100, 0), xytext=(8, 8),
               textcoords="offset points", ha="left", fontsize=7.5, color=INK2)
    c.set(xlabel="Non-sensitive objects forced local (%)",
          ylabel="Objects reaching sensitive data\nthat may still offload (%)", xlim=(0, 100), ylim=(-3, 100))
    c.legend(loc="upper right")
    panel(c, "(b)")
    save(fig, "fig04_odg")


def fig_queue(R):
    q = R["queue"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"wspace": 0.38})
    d = q["deterministic"]
    ax[0].plot(d["greedy"], color=ORANGE, label=f"Greedy ({d['greedy'][-1]} tasks)")
    ax[0].plot(d["controlled"], color=BLUE, label=f"Throttled (max {max(d['controlled'])})")
    ax[0].axhline(d["params"]["threshold"], color=INK2, ls=":", lw=1)
    ax[0].set(xlabel="Time slot", ylabel="Queued tasks", ylim=(0, 320))
    ax[0].legend(loc="upper left")
    panel(ax[0], "(a)")
    s = q["stochastic"]
    for name, col, lab in (("threshold", INK2, "Fixed threshold ($\\Theta$=20)"),
                           ("dpp_V20", BLUE, "Drift-plus-penalty ($V$=20)")):
        ax[1].plot(s[name]["trace"], color=col, lw=1.0, label=lab)
    ax[1].set(xlabel="Time slot", ylabel="Queued tasks", ylim=(0, 60), xlim=(0, 300))
    ax[1].text(0.97, 0.97, "greedy: off scale\n(mean %.0f tasks)" % s["greedy"]["mean_q"]["mean"],
               transform=ax[1].transAxes, ha="right", va="top", fontsize=7, color=ORANGE)
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.8))
    panel(ax[1], "(b)")
    Vs = [5, 10, 20, 50, 100]
    mq = [s[f"dpp_V{V}"]["mean_q"]["mean"] for V in Vs]
    bn = [s[f"dpp_V{V}"]["benefit_per_slot"]["mean"] for V in Vs]
    ax[2].plot(mq, bn, "o-", color=BLUE, ms=4, label="Drift-plus-penalty")
    for V, x, y in zip(Vs, mq, bn):
        ax[2].annotate(f"V={V}", (x, y), xytext=(3, -11), textcoords="offset points", fontsize=7, color=INK2)
    ax[2].plot(s["threshold"]["mean_q"]["mean"], s["threshold"]["benefit_per_slot"]["mean"], "s", color=INK2, ms=5,
               label="Fixed threshold")
    ax[2].set(xlabel="Mean queue length (tasks)", ylabel="Offloading benefit per slot", xscale="log",
              ylim=(3.85, 5.15))
    ax[2].legend(loc="lower right")
    panel(ax[2], "(c)")
    save(fig, "fig05_queue")


def fig_learning(R):
    L = R["learning"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    grid = None
    for name, col in (("Q-table", ORANGE), ("DQN", BLUE)):
        curves = L["curves"][name]
        top = min(cv[-1][0] for cv in curves)
        grid = np.linspace(0, top, 60)
        ys = np.array([np.interp(grid, [p[0] for p in cv], [p[1] for p in cv]) for cv in curves])
        m = ys.mean(0)
        h = 2.26 * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else 0
        a.plot(grid / 1000, m, color=col, label=name)
        a.fill_between(grid / 1000, m - h, m + h, color=col, alpha=0.18, lw=0)
    P = L["policies"]
    for nm, col, lab, dy in (("Always local", INK2, "always local", 0.0),
                             ("Rule pipeline (3 gates)", AQUA, "rule pipeline", 0.0),
                             ("Myopic oracle", VIOLET, "myopic oracle", 0.0)):
        v = P[nm]["reward_per_step"]["mean"]
        a.axhline(v, color=col, ls="--", lw=1)
        a.text(1.01, v, lab, va="center", ha="left", fontsize=7, color=col, transform=a.get_yaxis_transform(),
               clip_on=False)
    a.set(xlabel="Environment steps of training (thousands)", ylabel="Reward per step (greedy policy)",
          ylim=(-2.8, -1.7), xlim=(0, grid[-1] / 1000))
    a.legend(loc="lower right")
    panel(a, "(a)")
    gs = L["gamma_sweep"]
    gam = ["0.0", "0.5", "0.95"]
    x = np.arange(len(gam))
    for k, (nm, col) in enumerate((("Q-table", ORANGE), ("DQN", BLUE))):
        c.errorbar(x + (k - 0.5) * 0.12, [gs[g][nm]["mean"] for g in gam], yerr=[gs[g][nm]["ci95"] for g in gam],
                   fmt="o-", color=col, label=nm, ms=5, lw=1.4, capsize=2, mec="white", mew=0.8)
    c.set_xticks(x)
    c.set_xticklabels(gam)
    c.set(xlabel="Discount factor $\\gamma$", ylabel="Reward per step", ylim=(-2.8, -1.7), xlim=(-0.4, 2.4))
    c.legend(loc="lower left")
    c.grid(axis="x", visible=False)
    panel(c, "(b)")
    save(fig, "fig06_learning")


def fig_policies(R):
    P = R["learning"]["policies"]
    order = ["Always local", "Always offload", "Random", "Threshold only", "Threshold + ODG gate",
             "Rule pipeline (3 gates)", "Q-table", "Q-table + gates", "DQN", "DQN + gates (full pipeline)",
             "Myopic oracle"]
    cols = {"Always local": INK2, "Always offload": INK2, "Random": INK2, "Threshold only": INK2,
            "Threshold + ODG gate": AQUA, "Rule pipeline (3 gates)": AQUA, "Q-table": ORANGE,
            "Q-table + gates": ORANGE, "DQN": BLUE, "DQN + gates (full pipeline)": BLUE, "Myopic oracle": VIOLET}
    panels = [("reward_per_step", "Reward per step", 1), ("latency", "Latency per task (s)", 1),
              ("energy", "Energy per task (J)", 1),
              ("viol_high_rate", "High-vulnerability tasks\noffloaded (%)", 100)]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 3.0), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(order))[::-1]
    for ax, (key, lab, scale) in zip(axes, panels):
        for yi, nm in zip(y, order):
            m = P[nm][key]
            if key == "reward_per_step" and m["mean"] < -2.6:
                ax.annotate(f"{m['mean']:.2f}  ◀", (-2.6, yi), ha="left", va="center", fontsize=7, color=INK2)
                continue
            ax.errorbar(m["mean"] * scale, yi, xerr=m["ci95"] * scale, fmt="o", color=cols[nm], ms=4.5, lw=1,
                        capsize=2, mec="white", mew=0.8)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        if key == "reward_per_step":
            ax.set_xlim(-2.6, -1.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    save(fig, "fig07_policies")


def fig_surfaces(R):
    S = R["learning"]["surfaces"]
    g = np.array(S["grid"])
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    ln_min, ln_max = np.log(C.BW_MIN), np.log(C.BW_MAX)
    r_star_w = (np.log(C.R_STAR) - ln_min) / (ln_max - ln_min)
    ticks = [1, 3, 10, 30, 100]
    tpos = [(np.log(t * 1e6) - ln_min) / (ln_max - ln_min) for t in ticks]
    specs = [("A", "Bandwidth (Mbps)", "Vulnerability score", "battery 0.7, complexity 0.8, load 0.3", True),
             ("B", "Task complexity", "Battery level", "vulnerability 0.1, bandwidth 0.8, load 0.3", False),
             ("C", "Bandwidth (Mbps)", "Edge load", "battery 0.7, complexity 0.8, vulnerability 0.1", True)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={"wspace": 0.75})
    vmax = 1.5      # colour scale clipped so the structure near the decision boundary is visible
    for ax, (k, xl, yl, ttl, bw) in zip(axes, specs):
        M = np.array(S[k])
        im = ax.imshow(M, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.contour(g, g, M, levels=[0], colors=INK, linewidths=1)
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(ttl, fontsize=7)
        ax.grid(False)
        if bw:
            ax.set_xticks(tpos)
            ax.set_xticklabels(ticks)
            ax.axvline(r_star_w, color=INK2, ls=":", lw=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, extend="both")
        cb.set_label("$Q$(offload) $-$ $Q$(local)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    save(fig, "fig08_dqn_surfaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "results", "data", "results.json"))
    R = json.load(open(ap.parse_args().data))
    for f in (fig_breakeven, fig_nqueens, fig_pipeline, fig_odg, fig_queue, fig_learning, fig_policies, fig_surfaces):
        f(R)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
