"""Turn results/data/results.json into LaTeX macros and tables so the paper never contains typed numbers.

    python export_latex.py [--data ../results/data/results.json]

Writes ../results/tables/numbers.tex, tab_policies.tex, tab_probe.tex, tab_queue.tex.
"""
import argparse
import json
import os

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", "tables")
WORD = {5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred"}
POL = {"Always local": "Local", "Always offload": "Offload", "Random": "Random", "Threshold only": "ThrOnly",
       "Threshold + ODG gate": "ThrOdg", "Rule pipeline (3 gates)": "Rules", "Q-table": "Qtab",
       "Q-table + gates": "QtabG", "DQN": "Dqn", "DQN + gates (full pipeline)": "DqnG", "Myopic oracle": "Oracle"}


def pm(m, digits=2, scale=1.0):
    return f"{m['mean'] * scale:.{digits}f} $\\pm$ {m['ci95'] * scale:.{digits}f}"

def write_table(name, colspec, header, rows):
    """Write a complete tabular environment (an \\input of bare rows inside a tabular is fragile)."""
    body = ["\\begin{tabular}{" + colspec + "}", "\\toprule", header + " \\\\", "\\midrule"] + rows + \
           ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, name), "w", newline="\n") as f:
        f.write("\n".join(body) + "\n")


def main():
    bs = chr(92)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "results", "data", "results.json"))
    R = json.load(open(ap.parse_args().data))
    os.makedirs(OUT, exist_ok=True)
    mac = []

    def M(name, val):
        mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    b, q, o, nq, L = R["breakeven"], R["queue"], R["odg"], R["nqueens"], R["learning"]
    M("RstarNoSetup", f"{b['r_star_mbps']:.2f}")
    M("RstarSetup", f"{b['r_star_with_setup_mbps']:.1f}")
    d = q["deterministic"]
    M("QGreedyDet", d["greedy"][-1])
    M("QCtrlDet", max(d["controlled"]))
    M("QRedirDet", d["redirected"])
    s = q["stochastic"]
    M("QStochGreedyMean", f"{s['greedy']['mean_q']['mean']:.0f}")
    M("QStochGreedyMax", f"{s['greedy']['max_q']['mean']:.0f}")
    M("QStochThrMean", f"{s['threshold']['mean_q']['mean']:.1f}")
    M("QStochThrMax", f"{s['threshold']['max_q']['mean']:.0f}")
    M("QStochThrBenefit", f"{s['threshold']['benefit_per_slot']['mean']:.2f}")
    for V, w in WORD.items():
        e = s[f"dpp_V{V}"]
        M(f"QDppV{w}Mean", f"{e['mean_q']['mean']:.1f}")
        M(f"QDppV{w}Max", f"{e['max_q']['mean']:.0f}")
        M(f"QDppV{w}Benefit", f"{e['benefit_per_slot']['mean']:.2f}")
    g = o["random_graphs"]
    M("OdgLeakGate", f"{g['leak_at_gate'] * 100:.0f}")
    M("OdgBlockedGate", f"{g['blocked_at_gate'] * 100:.0f}")
    M("OdgTaintBlocked", f"{g['taint_blocked_fraction'] * 100:.0f}")
    M("NqCrossFifty", nq["crossover"]["50"] if nq["crossover"]["50"] else "none")
    M("NqCrossTen", nq["crossover"]["10"] if nq["crossover"]["10"] else "none")
    M("NSeeds", len(L["seeds"]))
    M("NEpisodes", L["episodes"])
    M("DqnParams", f"{L['dqn_params']:,}".replace(",", "{,}"))
    for full, key in POL.items():
        p = L["policies"][full]
        M(f"Rew{key}", f"{p['reward_per_step']['mean']:.2f}")
        M(f"Lat{key}", f"{p['latency']['mean']:.2f}")
        M(f"ViolHigh{key}", f"{p['viol_high_rate']['mean'] * 100:.1f}")
    for agent, key in (("Q-table", "Q"), ("DQN", "Dqn")):
        rows = L["probe"][agent]
        hi = [r["offload_high"] for r in rows]
        fav = [r["offload_fav"] for r in rows if r["offload_fav"] == r["offload_fav"]]
        M(f"Probe{key}High", f"{100 * sum(hi) / len(hi):.1f}")
        M(f"Probe{key}HighMax", f"{100 * max(hi):.1f}")
        M(f"Probe{key}Fav", f"{100 * sum(fav) / max(len(fav), 1):.1f}")
    for gm, w in (("0.0", "Zero"), ("0.5", "Half"), ("0.95", "Ninety")):
        for agent, key in (("Q-table", "Q"), ("DQN", "Dqn")):
            M(f"Gam{w}{key}", f"{L['gamma_sweep'][gm][agent]['mean']:.2f}")
    with open(os.path.join(OUT, "numbers.tex"), "w", newline="\n") as f:
        f.write("\n".join(mac) + "\n")

    order = list(POL)
    rows = []
    for nm in order:
        p = L["policies"][nm]
        rows.append(f"{nm} & {pm(p['reward_per_step'])} & {pm(p['latency'])} & {pm(p['energy'])} & "
                    f"{p['offload_rate']['mean'] * 100:.0f} & {p['viol_high_rate']['mean'] * 100:.1f} & "
                    f"{p['depleted_frac']['mean'] * 100:.0f} \\\\")
    write_table("tab_policies.tex", "@{}lcccccc@{}",
                "Policy & Reward/step & Latency (s) & Energy (J) & Offloaded (\\%) & High-vuln.\\ offl.\\ (\\%) "
                "& Depleted (\\%)", rows)

    rows = []
    for agent in ("Q-table", "DQN"):
        pr = L["probe"][agent]
        hi = [r["offload_high"] * 100 for r in pr]
        fav = [r["offload_fav"] * 100 for r in pr if r["offload_fav"] == r["offload_fav"]]
        al = [r["offload_all"] * 100 for r in pr]
        mean = lambda x: sum(x) / len(x)
        rows.append(f"{agent} & {mean(al):.1f} & {mean(hi):.1f} ({min(hi):.1f}--{max(hi):.1f}) & "
                    f"{mean(fav):.1f} ({min(fav):.1f}--{max(fav):.1f}) \\\\")
    write_table("tab_probe.tex", "@{}lccc@{}",
                "Agent & All states (\\%) & High vuln.\\ (\\%) & High vuln., favourable (\\%)", rows)

    rows = []
    labels = [("greedy", "Greedy (admit all)"), ("threshold", "Fixed threshold ($\\Theta=20$)")] + \
             [(f"dpp_V{V}", f"Drift-plus-penalty, $V={V}$") for V in WORD]
    for k, lab in labels:
        e = s[k]
        rows.append(f"{lab} & {e['mean_q']['mean']:.1f} & {e['max_q']['mean']:.0f} & "
                    f"{e['admit_frac']['mean'] * 100:.0f} & {pm(e['benefit_per_slot'])} \\\\")
    write_table("tab_queue.tex", "@{}lrrrr@{}",
                "Policy & Mean $Q$ & Max $Q$ & Admitted (\\%) & Benefit/slot", rows)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
