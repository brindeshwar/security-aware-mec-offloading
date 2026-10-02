"""
=============================================================================
  Adaptive Computation Offloading — Full Integrated Pipeline (Phases 1–5)
=============================================================================

  PIPELINE FLOW:
  ─────────────────────────────────────────────────────────────────────────
  Phase 1          Phase 2          Phase 5a         Phase 3          Phase 4
  Bandwidth   ──►  Task         ──► ODG Security ──► Secure      ──►  Lyapunov
  Threshold        Complexity       Scorer           Q-Learning       Queue
  (Can we          (Is it heavy     (Is it safe      Agent            Stability
  offload?)        enough?)         to offload?)     (DQN-ready)      (Multi-user)

  Phase 5b: Security-aware Q-Learning (extended from Phase 3)
  Phase 5c: Deep Q-Network replacing Q-table for continuous states
  ─────────────────────────────────────────────────────────────────────────

  HOW PHASES CONNECT:
  Phase 1  → threshold_bps  ────────────────────► Phase 3 reward + Phase 5b
  Phase 2  → task_type (LIGHT/HEAVY) ──────────► Phase 3 state  + Phase 5b
  Phase 5a → vuln_score (ODG) ─────────────────► Phase 5b state + Phase 5c
  Phase 5b → offload decision ─────────────────► Phase 4 queue
  Phase 4  → admission control ────────────────► forced_local override
  Phase 5c → DQN continuous policy (replaces 5b Q-table for demo)
=============================================================================
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import time
import random
from collections import defaultdict, deque

# ═════════════════════════════════════════════════════════════════════════════
#  SHARED SYSTEM PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

# Device & Server
F_LOCAL           = 1.2e9      # 1.2 GHz smartphone CPU
F_REMOTE          = 4.0e9      # 4.0 GHz edge server CPU

# Task defaults
TASK_CYCLES       = 1e9        # 1 Billion CPU cycles
TASK_DATA_BITS    = 8e6        # 1 MB payload

# Network
NETWORK_LAG       = 0.05       # 50 ms fixed network overhead

# Q-Learning (Phase 3 / Phase 5b)
EPISODES_P3       = 500
EPISODES_P5       = 600
LEARNING_RATE     = 0.05
GAMMA             = 0.95
EPSILON_START     = 1.0
EPSILON_MIN       = 0.01
EPSILON_DECAY     = 0.99
BATTERY_STATES    = 10
LOW_BATT_THRESH   = 2

# Phase 5 security
VULN_THRESHOLD    = 0.4        # ODG: score >= this → must stay local
SECURITY_PENALTY  = -3.0       # reward penalty for offloading sensitive task
VULN_BUCKETS      = 3          # LOW / MEDIUM / HIGH

# Lyapunov Queue (Phase 4)
TIME_SLOTS        = 100
ARRIVAL_RATE      = 10
SERVER_CAPACITY   = 7
QUEUE_THRESHOLD   = 20
THROTTLE_RATE     = 4

# DQN (Phase 5c)
STATE_DIM         = 5
ACTION_DIM        = 2
HIDDEN_SIZE       = 64
DQN_LR            = 0.001
EPISODES_DQN      = 800
REPLAY_BUFFER_SIZE= 2000
BATCH_SIZE        = 32
TARGET_UPDATE_FREQ= 20
PHASE1_NORM_THRESH= 0.5        # normalised bandwidth threshold for DQN env


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — BINARY DECISION THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════

def phase1_compute_threshold():
    """
    Compute break-even bandwidth R* analytically.
    Offload only when: data_bits/R + cycles/f_remote < cycles/f_local
    Solving → R* = data_bits * f_local * f_remote / (cycles * (f_remote - f_local))
    """
    bandwidths = np.linspace(1, 100, 500) * 1e6
    t_local    = np.full(len(bandwidths), TASK_CYCLES / F_LOCAL)
    t_offload  = (TASK_DATA_BITS / bandwidths) + (TASK_CYCLES / F_REMOTE)
    threshold_bps = (TASK_DATA_BITS * F_LOCAL * F_REMOTE) / \
                    (TASK_CYCLES * (F_REMOTE - F_LOCAL))
    return threshold_bps, bandwidths, t_local, t_offload


def phase1_plot(ax, threshold_bps, bandwidths, t_local, t_offload):
    tmb = threshold_bps / 1e6
    ax.plot(bandwidths / 1e6, t_local,   'r--', lw=2, label='Local Execution Time')
    ax.plot(bandwidths / 1e6, t_offload, 'b-',  lw=2, label='Edge Offloading Time')
    ax.axvline(tmb, color='green', ls=':', lw=2,
               label=f'Break-even ≈ {tmb:.1f} Mbps')
    ax.fill_betweenx([0, max(t_local)], 0, tmb,
                     alpha=0.07, color='red',   label='Stay Local Zone')
    ax.fill_betweenx([0, max(t_local)], tmb, 100,
                     alpha=0.07, color='green', label='Offload Zone')
    ax.set_title('Phase 1 — Binary Decision Threshold', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bandwidth (Mbps)'); ax.set_ylabel('Execution Time (s)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, max(t_local) * 1.2)


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — N-QUEENS WORKLOAD BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def _is_safe(board, row, col, n):
    for i in range(row):
        if board[i, col] == 1: return False
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i, j] == 1: return False
        i -= 1; j -= 1
    i, j = row, col
    while i >= 0 and j < n:
        if board[i, j] == 1: return False
        i -= 1; j += 1
    return True

def _solve_nqueens(board, row, n):
    if row >= n: return True
    for col in range(n):
        if _is_safe(board, row, col, n):
            board[row, col] = 1
            if _solve_nqueens(board, row + 1, n): return True
            board[row, col] = 0
    return False

def phase2_benchmark(n_values=None, repetitions=300):
    if n_values is None: n_values = [1, 2, 3, 4, 5, 6]
    local_times, remote_times = [], []
    for n in n_values:
        board = np.zeros((n, n), dtype=int)
        start = time.perf_counter()
        for _ in range(repetitions):
            board[:] = 0
            _solve_nqueens(board, 0, n)
        elapsed = time.perf_counter() - start
        local_times.append(elapsed)
        remote_times.append((elapsed / 10.0) + NETWORK_LAG)
    crossover_n = next((n for i, n in enumerate(n_values)
                        if remote_times[i] < local_times[i]), None)
    return local_times, remote_times, crossover_n

def phase2_classify_task(local_time, remote_time):
    return 'HEAVY' if remote_time < local_time else 'LIGHT'

def phase2_plot(ax, n_values, local_times, remote_times, crossover_n):
    x, w = np.arange(len(n_values)), 0.35
    ax.bar(x - w/2, local_times,  w, label='Local (Phone)', color='red',  alpha=0.82)
    ax.bar(x + w/2, remote_times, w, label='Edge (Offload)', color='blue', alpha=0.82)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in n_values])
    ax.set_title('Phase 2 — N-Queens Benchmark (log scale)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Problem Size (N)'); ax.set_ylabel('Time (s, log scale)')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)
    if crossover_n:
        idx = n_values.index(crossover_n)
        ax.annotate(f'Crossover N={crossover_n}',
                    xy=(idx + w/2, remote_times[idx]),
                    xytext=(idx + 1.0, remote_times[idx] * 3),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 5a — ODG VULNERABILITY SCORER
# ═════════════════════════════════════════════════════════════════════════════

class AppObject:
    def __init__(self, name, is_sensitive=False, description=""):
        self.name         = name
        self.is_sensitive = is_sensitive
        self.description  = description
        self.vuln_score   = 1.0 if is_sensitive else 0.0

class ObjectDependencyGraph:
    """
    Directed graph: nodes = code objects, edges = dependencies.
    DFS propagates vulnerability from sensitive sinks to all callers.
    """
    def __init__(self):
        self.objects = {}
        self.edges   = defaultdict(set)

    def add_object(self, obj):
        self.objects[obj.name] = obj

    def add_dependency(self, from_name, to_name):
        self.edges[from_name].add(to_name)

    def compute_vulnerability_scores(self):
        """
        vuln_score = sensitive_reachable / total_reachable  (DFS from each node)
        """
        for start_name, start_obj in self.objects.items():
            if start_obj.is_sensitive:
                start_obj.vuln_score = 1.0
                continue
            visited = set()
            stack   = [start_name]
            sens_ct = total_ct = 0
            while stack:
                node = stack.pop()
                if node in visited: continue
                visited.add(node)
                obj = self.objects.get(node)
                if obj and node != start_name:
                    total_ct += 1
                    if obj.is_sensitive: sens_ct += 1
                for nb in self.edges.get(node, []):
                    if nb not in visited: stack.append(nb)
            start_obj.vuln_score = (sens_ct / total_ct) if total_ct else 0.0

    def classify_objects(self):
        safe, local = [], []
        for obj in self.objects.values():
            (local if obj.vuln_score >= VULN_THRESHOLD else safe).append(obj)
        return safe, local

    def get_score(self, name):
        return self.objects[name].vuln_score if name in self.objects else 0.0

def build_sample_app():
    """Healthcare fitness app — realistic mix of sensitive and safe objects."""
    g = ObjectDependencyGraph()
    # Sensitive sinks
    for name, desc in [("getUserID",        "User identity"),
                        ("getHeartRate",     "Biometric sensor"),
                        ("getMedicalRecord", "Health history"),
                        ("getLocation",      "GPS coordinates")]:
        g.add_object(AppObject(name, is_sensitive=True, description=desc))
    # Safe compute
    for name, desc in [("calculateBMI",    "BMI math"),
                        ("sortLeaderboard", "Public sort"),
                        ("renderChart",     "Display logic"),
                        ("encryptData",     "AES utility")]:
        g.add_object(AppObject(name, is_sensitive=False, description=desc))
    # Mixed
    for name, desc in [("analyzeHealth",  "Trend analysis"),
                        ("buildReport",   "Health report"),
                        ("syncToCloud",   "Sync package"),
                        ("runAIInference","ML inference")]:
        g.add_object(AppObject(name, is_sensitive=False, description=desc))
    # Edges
    g.add_dependency("analyzeHealth",  "getHeartRate")
    g.add_dependency("analyzeHealth",  "calculateBMI")
    g.add_dependency("buildReport",    "analyzeHealth")
    g.add_dependency("buildReport",    "getUserID")
    g.add_dependency("syncToCloud",    "buildReport")
    g.add_dependency("syncToCloud",    "sortLeaderboard")
    g.add_dependency("syncToCloud",    "encryptData")
    g.add_dependency("runAIInference", "analyzeHealth")
    g.add_dependency("runAIInference", "calculateBMI")
    g.add_dependency("renderChart",    "sortLeaderboard")
    return g

def score_to_bucket(vuln_score):
    if vuln_score < 0.33:  return 0   # LOW
    elif vuln_score < 0.66: return 1  # MEDIUM
    else:                   return 2  # HIGH

BUCKET_LABELS = {0: "LOW", 1: "MED", 2: "HIGH"}

def phase5a_plot(ax_bar, ax_graph, graph):
    names  = list(graph.objects.keys())
    scores = [graph.objects[n].vuln_score for n in names]
    colors = ['#d62728' if s >= VULN_THRESHOLD else '#2ca02c' for s in scores]

    bars = ax_bar.barh(names, scores, color=colors, edgecolor='white', height=0.6)
    ax_bar.axvline(VULN_THRESHOLD, color='black', ls='--', lw=2,
                   label=f'Threshold={VULN_THRESHOLD}')
    ax_bar.set_xlim(0, 1.35)
    ax_bar.set_xlabel('Vulnerability Score')
    ax_bar.set_title('Phase 5a — ODG Vulnerability Scores', fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=8); ax_bar.grid(True, axis='x', alpha=0.3)
    for bar, score in zip(bars, scores):
        lbl = "LOCAL" if score >= VULN_THRESHOLD else "OFFLOAD"
        ax_bar.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f} {lbl}', va='center', fontsize=7,
                    color='#d62728' if score >= VULN_THRESHOLD else '#2ca02c',
                    fontweight='bold')

    # Dependency graph layout
    ax_graph.set_xlim(0, 10); ax_graph.set_ylim(0, 10); ax_graph.axis('off')
    ax_graph.set_title('Object Dependency Graph\n(Red=Local, Green=Offload)',
                        fontsize=10, fontweight='bold')
    positions = {
        "getUserID": (1.2,9.0), "getHeartRate":(1.2,7.5),
        "getMedicalRecord":(1.2,6.0), "getLocation":(1.2,4.5),
        "calculateBMI":(1.2,3.0), "sortLeaderboard":(1.2,1.5),
        "encryptData":(1.2,0.3), "renderChart":(4.5,1.5),
        "analyzeHealth":(4.5,7.5), "buildReport":(7.2,8.2),
        "syncToCloud":(7.2,5.5), "runAIInference":(7.2,7.0),
    }
    for src, dests in graph.edges.items():
        if src not in positions: continue
        sx, sy = positions[src]
        for dst in dests:
            if dst not in positions: continue
            dx, dy = positions[dst]
            ax_graph.annotate("", xy=(dx,dy), xytext=(sx,sy),
                arrowprops=dict(arrowstyle="-|>", color='#999', lw=1.0,
                                mutation_scale=10))
    for name, (x, y) in positions.items():
        obj   = graph.objects[name]
        color = '#d62728' if obj.vuln_score >= VULN_THRESHOLD else '#2ca02c'
        ax_graph.add_patch(plt.Circle((x, y), 0.52, color=color, zorder=3, alpha=0.9))
        short = name.replace("get","").replace("calculate","calc\n")[:11]
        ax_graph.text(x, y, short, ha='center', va='center',
                      fontsize=6, fontweight='bold', color='white', zorder=4)
        ax_graph.text(x, y-0.7, f'{obj.vuln_score:.2f}', ha='center', va='top',
                      fontsize=6.5, color=color, fontweight='bold')
    ax_graph.legend(handles=[
        mpatches.Patch(color='#d62728', label='Must Stay Local'),
        mpatches.Patch(color='#2ca02c', label='Safe to Offload')
    ], loc='lower right', fontsize=8)


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Q-LEARNING (original, single-user, no security)
#  Kept as BASELINE for comparison against Phase 5b
# ═════════════════════════════════════════════════════════════════════════════

def phase3_train(threshold_bps):
    """Original Phase 3 Q-Learning — state: (battery, task_type)"""
    np.random.seed(42)
    q_table = np.zeros((BATTERY_STATES, 2, 2))
    rewards_history = []
    epsilon = EPSILON_START
    bw_mean = threshold_bps; bw_std = threshold_bps * 0.5

    for e in range(EPISODES_P3):
        battery   = np.random.randint(0, BATTERY_STATES)
        task_type = np.random.randint(0, 2)
        bandwidth = max(1e6, np.random.normal(bw_mean, bw_std))
        if np.random.uniform(0,1) < epsilon:
            action = np.random.choice([0,1])
        else:
            action = int(np.argmax(q_table[battery, task_type]))
        bw_ok = bandwidth > threshold_bps
        low_b = battery <= LOW_BATT_THRESH
        heavy = task_type == 1
        if low_b and action == 0:                         reward = +1
        elif heavy and bw_ok and not low_b and action==1: reward = +1
        elif not heavy and action == 0:                   reward = +1
        else:                                             reward = -1
        nb = max(0, battery-1); nt = np.random.randint(0,2)
        td = reward + GAMMA * np.max(q_table[nb,nt]) - q_table[battery,task_type,action]
        q_table[battery,task_type,action] += LEARNING_RATE * td
        if epsilon > EPSILON_MIN: epsilon *= EPSILON_DECAY
        rewards_history.append(float(np.sum(q_table)))

    return q_table, rewards_history

def phase3_decide(q_table, battery, task_type, bandwidth, threshold_bps):
    if bandwidth < threshold_bps:
        return 0, "BW below threshold → LOCAL"
    action = int(np.argmax(q_table[battery, task_type]))
    return action, "LOCAL" if action == 0 else "OFFLOAD"


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 5b — SECURITY-AWARE Q-LEARNING
#  Extended state: (battery, task_type, vuln_bucket)  →  60 states
# ═════════════════════════════════════════════════════════════════════════════

def compute_secure_reward(battery, task_type, vuln_bucket, action, bandwidth, threshold_bps):
    bw_ok = bandwidth > threshold_bps
    low_b = battery <= LOW_BATT_THRESH
    heavy = task_type == 1

    if vuln_bucket == 2 and action == 1: return SECURITY_PENALTY
    if vuln_bucket == 1 and action == 1: return SECURITY_PENALTY / 2
    if vuln_bucket == 2 and action == 0: return +1.0
    if low_b  and action == 0:           return +1.0
    if low_b  and action == 1:           return -1.0
    if not bw_ok and action == 1:        return -1.0
    if not bw_ok and action == 0:        return +1.0
    if not heavy and action == 0:        return +1.0
    if not heavy and action == 1:        return -1.0
    if heavy and bw_ok and not low_b and action == 1: return +1.0
    return -1.0

def phase5b_train(threshold_bps):
    """Security-aware Q-Learning — state: (battery, task_type, vuln_bucket)"""
    np.random.seed(42)
    q_table = np.zeros((BATTERY_STATES, 2, VULN_BUCKETS, 2))
    rewards_history = []
    epsilon = EPSILON_START
    bw_mean = threshold_bps; bw_std = threshold_bps * 0.5

    for e in range(EPISODES_P5):
        battery     = np.random.randint(0, BATTERY_STATES)
        task_type   = np.random.randint(0, 2)
        vuln_score  = np.random.uniform(0.0, 1.0)
        vuln_bucket = score_to_bucket(vuln_score)
        bandwidth   = max(1e6, np.random.normal(bw_mean, bw_std))

        if np.random.uniform(0,1) < epsilon:
            action = np.random.choice([0,1])
        else:
            action = int(np.argmax(q_table[battery, task_type, vuln_bucket]))

        reward = compute_secure_reward(battery, task_type, vuln_bucket,
                                       action, bandwidth, threshold_bps)
        nb  = max(0, battery-1)
        nt  = np.random.randint(0,2)
        nvb = score_to_bucket(np.random.uniform(0,1))
        td  = reward + GAMMA * np.max(q_table[nb,nt,nvb]) - \
              q_table[battery, task_type, vuln_bucket, action]
        q_table[battery, task_type, vuln_bucket, action] += LEARNING_RATE * td
        if epsilon > EPSILON_MIN: epsilon *= EPSILON_DECAY
        rewards_history.append(float(np.sum(q_table)))

    return q_table, rewards_history

def phase5b_decide(q_table, battery, task_type, vuln_score, bandwidth, threshold_bps):
    """Security-aware decision with hard overrides."""
    vb = score_to_bucket(vuln_score)
    if vb == 2:
        return 0, f"SECURITY OVERRIDE: vuln={vuln_score:.2f} (HIGH) → LOCAL"
    if bandwidth < threshold_bps:
        return 0, f"BW OVERRIDE: {bandwidth/1e6:.1f}Mbps < threshold → LOCAL"
    action = int(np.argmax(q_table[battery, task_type, vb]))
    label  = "OFFLOAD" if action == 1 else "LOCAL"
    return action, (f"Batt={battery} Task={'HEAVY' if task_type else 'LIGHT'} "
                    f"Vuln={vuln_score:.2f}({BUCKET_LABELS[vb]}) "
                    f"BW={bandwidth/1e6:.1f}Mbps → {label}")

def phase5b_plot(ax_conv, ax_pol, q_table_p3, rewards_p3, q_table_p5, rewards_p5):
    """Side-by-side convergence and policy comparison."""
    # Convergence
    ax_conv.plot(rewards_p3, color='steelblue',  lw=1.5, alpha=0.85, label='Phase 3 (no security)')
    ax_conv.plot(rewards_p5, color='darkviolet', lw=1.5, alpha=0.85, label='Phase 5b (security-aware)')
    ax_conv.set_title('Phase 3 vs 5b — Q-Learning Convergence', fontsize=11, fontweight='bold')
    ax_conv.set_xlabel('Training Episodes'); ax_conv.set_ylabel('Cumulative Q-Table Sum')
    ax_conv.legend(fontsize=8); ax_conv.grid(True, alpha=0.3)

    # Policy heatmap for Phase 5b (HEAVY tasks)
    grid = np.zeros((BATTERY_STATES, VULN_BUCKETS))
    for b in range(BATTERY_STATES):
        for v in range(VULN_BUCKETS):
            grid[b,v] = q_table_p5[b,1,v,1] - q_table_p5[b,1,v,0]
    im = ax_pol.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
    ax_pol.set_xticks([0,1,2]); ax_pol.set_xticklabels(['LOW','MED','HIGH'], fontsize=8)
    ax_pol.set_yticks(range(BATTERY_STATES))
    ax_pol.set_yticklabels([f'Batt {i}' for i in range(BATTERY_STATES)], fontsize=8)
    ax_pol.set_title('Phase 5b Policy — HEAVY Tasks\n(Green=Offload, Red=Local)',
                     fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax_pol, label='Q[off]−Q[loc]')
    for b in range(BATTERY_STATES):
        for v in range(VULN_BUCKETS):
            val = grid[b,v]
            ax_pol.text(v, b, 'OFF' if val>0 else 'LOC', ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='black' if abs(val)<0.8 else 'white')


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — LYAPUNOV QUEUE STABILITY
# ═════════════════════════════════════════════════════════════════════════════

def phase4_simulate():
    greedy_q = [0]; lyapunov_q = [0]; throttle_events = [False]; forced_local = 0
    for t in range(1, TIME_SLOTS):
        greedy_q.append(max(0, greedy_q[t-1] + ARRIVAL_RATE - SERVER_CAPACITY))
        if lyapunov_q[t-1] < QUEUE_THRESHOLD:
            admitted = ARRIVAL_RATE; throttled = False
        else:
            admitted = THROTTLE_RATE; throttled = True
            forced_local += (ARRIVAL_RATE - THROTTLE_RATE)
        lyapunov_q.append(max(0, lyapunov_q[t-1] + admitted - SERVER_CAPACITY))
        throttle_events.append(throttled)
    return greedy_q, lyapunov_q, throttle_events, forced_local

def phase4_plot(ax, greedy_q, lyapunov_q, throttle_events, forced_local):
    slots = range(len(greedy_q))
    ax.plot(slots, greedy_q,   'r--', lw=2, label=f'Greedy admission (→{greedy_q[-1]})')
    ax.plot(slots, lyapunov_q, 'b-',  lw=2, label=f'Threshold controller (max {max(lyapunov_q)})')
    ax.axhline(QUEUE_THRESHOLD, color='black', ls=':', lw=1.5,
               label=f'Stability Threshold={QUEUE_THRESHOLD}')
    shade_start = None
    for t, thr in enumerate(throttle_events):
        if thr and shade_start is None: shade_start = t
        if not thr and shade_start is not None:
            ax.axvspan(shade_start, t, alpha=0.10, color='orange')
            shade_start = None
    if shade_start:
        ax.axvspan(shade_start, len(throttle_events)-1, alpha=0.10, color='orange',
                   label='Throttle Active')
    ax.set_title('Phase 4 — Multi-User Queue Stability (Lyapunov)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Slots'); ax.set_ylabel('Pending Tasks in Queue')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95, f'Tasks forced local: {forced_local}',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 5c — DEEP Q-NETWORK (DQN)
# ═════════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """2-hidden-layer feedforward net in pure NumPy. No external ML library needed."""
    def __init__(self, in_dim, hid_dim, out_dim, lr=DQN_LR):
        self.lr = lr
        np.random.seed(42)
        s1 = np.sqrt(2.0/in_dim); s2 = np.sqrt(2.0/hid_dim)
        self.W1 = np.random.randn(in_dim,  hid_dim)*s1; self.b1 = np.zeros(hid_dim)
        self.W2 = np.random.randn(hid_dim, hid_dim)*s2; self.b2 = np.zeros(hid_dim)
        self.W3 = np.random.randn(hid_dim, out_dim)*s2; self.b3 = np.zeros(out_dim)

    def relu(self, x): return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(float)

    def forward(self, x):
        x = np.atleast_2d(x)
        self.x_in = x
        self.z1 = x  @ self.W1 + self.b1; self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def train_step(self, x, target):
        x = np.atleast_2d(x); target = np.atleast_2d(target)
        q  = self.forward(x)
        loss = np.mean((q - target)**2)
        d3 = 2*(q-target)/x.shape[0]
        dW3 = self.a2.T@d3;  db3 = d3.sum(0)
        d2  = (d3@self.W3.T)*self.relu_d(self.z2)
        dW2 = self.a1.T@d2;  db2 = d2.sum(0)
        d1  = (d2@self.W2.T)*self.relu_d(self.z1)
        dW1 = x.T@d1;        db1 = d1.sum(0)
        self.W3-=self.lr*dW3; self.b3-=self.lr*db3
        self.W2-=self.lr*dW2; self.b2-=self.lr*db2
        self.W1-=self.lr*dW1; self.b1-=self.lr*db1
        return loss

    def copy_from(self, other):
        self.W1=other.W1.copy(); self.b1=other.b1.copy()
        self.W2=other.W2.copy(); self.b2=other.b2.copy()
        self.W3=other.W3.copy(); self.b3=other.b3.copy()

class ReplayBuffer:
    def __init__(self, size=REPLAY_BUFFER_SIZE):
        self.buf = deque(maxlen=size)
    def push(self, *args): self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*batch)
        return np.array(s),np.array(a),np.array(r),np.array(ns),np.array(d)
    def __len__(self): return len(self.buf)

class MECEnvironment:
    """Continuous-state MEC environment. State = [battery, complexity, vuln, bw, server_load]"""
    def reset(self):
        self.battery  = np.random.uniform(0,1)
        self.complex  = np.random.uniform(0,1)
        self.vuln     = np.random.uniform(0,1)
        self.bw       = np.random.uniform(0,1)
        self.srvload  = np.random.uniform(0,1)
        return self._state()
    def _state(self):
        return np.array([self.battery,self.complex,self.vuln,self.bw,self.srvload],
                        dtype=np.float32)
    def step(self, action):
        r = 0.0
        if   self.vuln > 0.66 and action==1:                                r = SECURITY_PENALTY
        elif self.vuln > 0.33 and action==1:                                r = SECURITY_PENALTY/2
        elif self.vuln > 0.66 and action==0:                                r = 1.0
        elif self.battery < 0.2 and action==0:                             r = 1.0
        elif self.battery < 0.2 and action==1:                             r = -1.0
        elif self.bw < PHASE1_NORM_THRESH and action==1:                   r = -1.0
        elif self.bw < PHASE1_NORM_THRESH and action==0:                   r = 1.0
        elif self.complex < 0.3 and action==0:                             r = 1.0
        elif (self.complex>0.5 and self.bw>PHASE1_NORM_THRESH
              and self.battery>0.2 and self.vuln<0.33 and action==1):
            r = 1.5 if self.srvload < 0.7 else 0.8
        else:                                                               r = -1.0
        if action==1 and self.srvload>0.85: r -= 0.5
        self.battery = max(0, self.battery - np.random.uniform(0.01,0.05))
        self.complex = np.clip(self.complex + np.random.normal(0,0.1), 0, 1)
        self.vuln    = np.random.uniform(0,1)
        self.bw      = np.clip(self.bw + np.random.normal(0,0.05), 0, 1)
        self.srvload = np.clip(self.srvload + np.random.normal(0,0.1), 0, 1)
        done = self.battery < 0.01
        return self._state(), r, done

class DQNAgent:
    def __init__(self):
        self.main   = NeuralNetwork(STATE_DIM, HIDDEN_SIZE, ACTION_DIM)
        self.target = NeuralNetwork(STATE_DIM, HIDDEN_SIZE, ACTION_DIM)
        self.target.copy_from(self.main)
        self.replay  = ReplayBuffer()
        self.epsilon = EPSILON_START

    def act(self, state):
        if np.random.rand() < self.epsilon: return np.random.randint(ACTION_DIM)
        return int(np.argmax(self.main.forward(state)[0]))

    def store(self, *args): self.replay.push(*args)

    def train_step(self):
        if len(self.replay) < BATCH_SIZE: return 0.0
        s,a,r,ns,d = self.replay.sample(BATCH_SIZE)
        cq  = self.main.forward(s)
        nq  = self.target.forward(ns)
        tq  = cq.copy()
        for i in range(BATCH_SIZE):
            tq[i,a[i]] = r[i] if d[i] else r[i] + GAMMA*np.max(nq[i])
        return self.main.train_step(s, tq)

    def decay_eps(self):
        if self.epsilon > EPSILON_MIN: self.epsilon *= EPSILON_DECAY

    def update_target(self): self.target.copy_from(self.main)

    def decide(self, state):
        q = self.main.forward(state)[0]
        return int(np.argmax(q)), q

def phase5c_train():
    np.random.seed(42); random.seed(42)
    agent = DQNAgent(); env = MECEnvironment()
    ep_rewards, losses, eps_hist = [], [], []
    print(f"  {'Ep':>6}  {'AvgRew':>8}  {'Loss':>9}  {'Eps':>7}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}")
    for ep in range(EPISODES_DQN):
        state = env.reset(); total_r = 0.0; ep_loss = []; done = False; steps = 0
        while not done and steps < 50:
            a = agent.act(state)
            ns, r, done = env.step(a)
            agent.store(state, a, r, ns, done)
            loss = agent.train_step()
            if loss > 0: ep_loss.append(loss)
            agent.decay_eps(); total_r += r; state = ns; steps += 1
        if ep % TARGET_UPDATE_FREQ == 0: agent.update_target()
        ep_rewards.append(total_r)
        losses.append(np.mean(ep_loss) if ep_loss else 0.0)
        eps_hist.append(agent.epsilon)
        if ep % 200 == 0 or ep == EPISODES_DQN-1:
            avg = np.mean(ep_rewards[-50:]) if ep>=50 else np.mean(ep_rewards)
            print(f"  {ep:>6}  {avg:>8.2f}  "
                  f"{losses[-1]:>9.5f}  {agent.epsilon:>7.4f}")
    return agent, ep_rewards, losses, eps_hist

def phase5c_plot(ax_rew, ax_surf, ax_cmp, agent, ep_rewards, losses):
    # Reward curve
    w  = 30
    sm = np.convolve(ep_rewards, np.ones(w)/w, 'valid')
    ax_rew.plot(ep_rewards, alpha=0.2, color='blue', lw=0.8)
    ax_rew.plot(range(w-1, len(ep_rewards)), sm, 'b-', lw=2,
                label=f'{w}-ep moving avg')
    ax_rew.set_title('Phase 5c DQN — Episode Rewards', fontsize=11, fontweight='bold')
    ax_rew.set_xlabel('Episode'); ax_rew.set_ylabel('Total Reward')
    ax_rew.legend(fontsize=8); ax_rew.grid(True, alpha=0.3)

    # Policy surface: vuln_score vs bandwidth
    vr = np.linspace(0, 1, 25); br = np.linspace(0, 1, 25)
    surf = np.zeros((25, 25))
    for i, vs in enumerate(vr):
        for j, bw in enumerate(br):
            s = np.array([0.7, 0.8, vs, bw, 0.3], dtype=np.float32)
            q = agent.main.forward(s)[0]
            surf[i,j] = q[1] - q[0]
    im = ax_surf.imshow(surf, origin='lower', cmap='RdYlGn',
                        extent=[0,1,0,1], aspect='auto', vmin=-2, vmax=2)
    ax_surf.axvline(PHASE1_NORM_THRESH, color='white', ls='--', lw=1.5,
                    label=f'BW threshold={PHASE1_NORM_THRESH}')
    ax_surf.axhline(0.66, color='orange', ls='--', lw=1.5, label='HIGH vuln (0.66)')
    ax_surf.set_xlabel('Normalised Bandwidth'); ax_surf.set_ylabel('Vulnerability Score')
    ax_surf.set_title('DQN Policy Surface\n(Battery=70%, Complexity=80%)',
                      fontsize=10, fontweight='bold')
    ax_surf.legend(fontsize=7, loc='upper left')
    plt.colorbar(im, ax=ax_surf, label='Q[off]−Q[loc]')

    # Comparison table: Q-table vs DQN
    ax_cmp.axis('off')
    rows = [
        ("Feature",           "Q-Table (Ph3/5b)", "DQN (Ph5c)"),
        ("State type",        "Discrete",          "Continuous"),
        ("No. of states",     "10 / 60",           "Infinite"),
        ("Generalises",       "No",                "Yes"),
        ("Security-aware",    "Phase 5b: Yes",     "Yes"),
        ("Server load aware", "No",                "Yes"),
        ("BW awareness",      "Threshold only",    "Continuous"),
        ("Architecture",      "Array lookup",      "5→64→64→2 NN"),
    ]
    cx = [0.02, 0.38, 0.72]; ry = 0.94
    for j,h in enumerate(rows[0]):
        ax_cmp.text(cx[j], ry, h, fontsize=8.5, fontweight='bold',
                    transform=ax_cmp.transAxes, color='white',
                    bbox=dict(boxstyle='round', facecolor='#1f4e79', alpha=0.9))
    green_vals = {'Continuous','Infinite','Yes','5→64→64→2 NN'}
    red_vals   = {'Discrete','No','Array lookup'}
    for i, row in enumerate(rows[1:]):
        y = ry - (i+1)*0.10
        for j, cell in enumerate(row):
            c = '#2ca02c' if cell in green_vals else \
                ('#d62728' if cell in red_vals else 'black')
            ax_cmp.text(cx[j], y, cell, fontsize=8, transform=ax_cmp.transAxes,
                        color=c, fontweight='bold' if c!='black' else 'normal')
    ax_cmp.set_title('Q-Table vs DQN', fontsize=11, fontweight='bold')


# ═════════════════════════════════════════════════════════════════════════════
#  COMPARISON: Phase 3 vs Phase 5b decisions on same tasks
# ═════════════════════════════════════════════════════════════════════════════

def print_decision_comparison(q_p3, q_p5, threshold_bps):
    test_cases = [
        # (battery, task_type, vuln_score, bw_mbps, description)
        (9, 1, 0.05, 25.0, "Full bat, HEAVY, LOW vuln, good BW"),
        (9, 1, 0.50, 25.0, "Full bat, HEAVY, MED vuln, good BW"),
        (9, 1, 0.80, 25.0, "Full bat, HEAVY, HIGH vuln, good BW"),
        (2, 1, 0.05, 25.0, "LOW battery, HEAVY, safe task"),
        (8, 0, 0.05, 25.0, "Good bat, LIGHT task, safe"),
        (8, 1, 0.05,  5.0, "Good bat, HEAVY, safe, BAD BW"),
        (9, 1, 0.90, 40.0, "Full bat, HEAVY, HIGH vuln, excellent BW"),
    ]
    print(f"\n  {'Case':<40} {'Phase 3':>10}  {'Phase 5b':>10}  Note")
    print(f"  {'─'*40}  {'─'*10}  {'─'*10}  {'─'*25}")
    for batt, tt, vs, bw, desc in test_cases:
        p3_act, _ = phase3_decide(q_p3, batt, tt, bw*1e6, threshold_bps)
        p5_act, _ = phase5b_decide(q_p5, batt, tt, vs, bw*1e6, threshold_bps)
        p3s = "OFFLOAD" if p3_act==1 else "LOCAL"
        p5s = "OFFLOAD" if p5_act==1 else "LOCAL"
        note = "⚠ BLOCKED (security)" if p3_act==1 and p5_act==0 and vs>=0.33 \
               else ("✔ same" if p3_act==p5_act else "↻ changed")
        print(f"  {desc:<40} {p3s:>10}  {p5s:>10}  {note}")


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    print("=" * 68)
    print("  ADAPTIVE COMPUTATION OFFLOADING — Full Pipeline (Phases 1–5)")
    print("=" * 68)

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    print("\n[Phase 1] Binary decision threshold...")
    threshold_bps, bandwidths, t_local, t_offload = phase1_compute_threshold()
    print(f"          Break-even bandwidth: {threshold_bps/1e6:.2f} Mbps")

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    print("\n[Phase 2] N-Queens benchmark (~15s)...")
    n_values = [1,2,3,4,5,6]
    local_times, remote_times, crossover_n = phase2_benchmark(n_values)
    for i,n in enumerate(n_values):
        d = phase2_classify_task(local_times[i], remote_times[i])
        print(f"          N={n}  local={local_times[i]:.4f}s  "
              f"remote={remote_times[i]:.4f}s  → {d}")
    print(f"          Crossover: N={crossover_n or '>6'}")

    # ── PHASE 5a: ODG ────────────────────────────────────────────────────────
    print("\n[Phase 5a] ODG vulnerability scoring...")
    graph = build_sample_app()
    graph.compute_vulnerability_scores()
    safe_list, local_list = graph.classify_objects()
    print(f"          Safe to offload ({len(safe_list)}): "
          f"{[o.name for o in safe_list]}")
    print(f"          Must stay local ({len(local_list)}): "
          f"{[o.name for o in local_list]}")

    # ── PHASE 3: Q-Learning (baseline) ───────────────────────────────────────
    print("\n[Phase 3] Training baseline Q-Learning (500 eps)...")
    q_p3, rewards_p3 = phase3_train(threshold_bps)
    print(f"          Peak Q-value: {max(rewards_p3):.2f}")

    # ── PHASE 5b: Secure Q-Learning ──────────────────────────────────────────
    print("\n[Phase 5b] Training security-aware Q-Learning (600 eps)...")
    q_p5, rewards_p5 = phase5b_train(threshold_bps)
    print(f"          Peak Q-value: {max(rewards_p5):.2f}")

    print("\n[Phase 3 vs 5b] Decision comparison (same tasks):")
    print_decision_comparison(q_p3, q_p5, threshold_bps)

    # ── PHASE 4: Lyapunov ────────────────────────────────────────────────────
    print("\n[Phase 4] Lyapunov queue stability...")
    greedy_q, lyapunov_q, throttle_events, forced_local = phase4_simulate()
    print(f"          Greedy final queue : {greedy_q[-1]} (unbounded under arrival > capacity)")
    print(f"          Lyapunov max queue : {max(lyapunov_q)} (bounded by throttle threshold)")
    print(f"          Tasks forced local : {forced_local}")

    # ── PHASE 5c: DQN ────────────────────────────────────────────────────────
    print("\n[Phase 5c] Training DQN (800 eps)...")
    dqn_agent, ep_rewards, dqn_losses, eps_hist = phase5c_train()
    print(f"\n[Phase 5c] Final avg reward (last 50): "
          f"{np.mean(ep_rewards[-50:]):.2f}")

    print("\n[Phase 5c] Sample DQN decisions (continuous state):")
    print(f"  {'Bat':>5} {'Cmx':>5} {'Vln':>5} {'BW':>5} {'Srv':>5}  "
          f"{'Action':>9}  Q[loc]  Q[off]")
    for s in [[0.9,0.9,0.05,0.9,0.2],[0.9,0.9,0.80,0.9,0.2],
              [0.1,0.9,0.05,0.9,0.2],[0.9,0.2,0.05,0.9,0.2],
              [0.9,0.9,0.05,0.2,0.2],[0.7,0.7,0.15,0.7,0.4]]:
        st = np.array(s, dtype=np.float32)
        a, q = dqn_agent.decide(st)
        print(f"  {s[0]:>5.1f} {s[1]:>5.1f} {s[2]:>5.2f} "
              f"{s[3]:>5.1f} {s[4]:>5.1f}  "
              f"{'→ OFFLOAD' if a==1 else '→ LOCAL':>9}  "
              f"{q[0]:>6.2f}  {q[1]:>6.2f}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  FULL PIPELINE SUMMARY")
    print("="*68)
    print(f"  Phase 1   Break-even bandwidth        : {threshold_bps/1e6:.1f} Mbps")
    print(f"  Phase 2   Crossover N                 : {crossover_n or '>6'}")
    print(f"  Phase 5a  Objects safe to offload      : {len(safe_list)}/{len(graph.objects)}")
    print(f"  Phase 5a  Objects must stay local      : {len(local_list)}/{len(graph.objects)}")
    print(f"  Phase 3   Baseline Q-table training       : 500 updates (Q-sum still rising; not shown to converge)")
    print(f"  Phase 5b  Secure Q-table training         : 600 updates (Q-sum still rising; not shown to converge)")
    print(f"  Phase 4   Lyapunov queue bounded at    : ≤{max(lyapunov_q)} tasks")
    print(f"  Phase 4   Greedy queue (unstable)      : {greedy_q[-1]} tasks")
    print(f"  Phase 5c  DQN avg reward (last 50)     : {np.mean(ep_rewards[-50:]):.2f}")
    print(f"\n  ✔ ODG gate keeps sensitive objects local (by construction); run evaluate_dqn_blocking.py for the DQN rate")
    print(f"  ✔ Continuous-state DQN trained (simulation only; no real-device evaluation)")
    print("="*68)

    # ── ALL PLOTS ─────────────────────────────────────────────────────────────
    print("\n[Plots] Generating full dashboard...")

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        'Adaptive Computation Offloading — Full Integrated Pipeline (Phases 1–5)',
        fontsize=16, fontweight='bold', y=0.99
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)

    # Row 0: Phase 1, Phase 2, Phase 5a bar, Phase 5a graph
    ax_p1      = fig.add_subplot(gs[0, 0])
    ax_p2      = fig.add_subplot(gs[0, 1])
    ax_odg_bar = fig.add_subplot(gs[0, 2])
    ax_odg_grp = fig.add_subplot(gs[0, 3])

    # Row 1: Phase 3 vs 5b convergence, Phase 5b policy, Phase 4
    ax_conv    = fig.add_subplot(gs[1, 0:2])
    ax_pol     = fig.add_subplot(gs[1, 2])
    ax_p4      = fig.add_subplot(gs[1, 3])

    # Row 2-3: DQN reward, DQN surface, DQN comparison table
    ax_dqn_r   = fig.add_subplot(gs[2, 0:2])
    ax_dqn_s   = fig.add_subplot(gs[2, 2:4])
    ax_dqn_c   = fig.add_subplot(gs[3, 0:2])

    # Summary text panel
    ax_summ    = fig.add_subplot(gs[3, 2:4])

    phase1_plot(ax_p1, threshold_bps, bandwidths, t_local, t_offload)
    phase2_plot(ax_p2, n_values, local_times, remote_times, crossover_n)
    phase5a_plot(ax_odg_bar, ax_odg_grp, graph)
    phase5b_plot(ax_conv, ax_pol, q_p3, rewards_p3, q_p5, rewards_p5)
    phase4_plot(ax_p4, greedy_q, lyapunov_q, throttle_events, forced_local)
    phase5c_plot(ax_dqn_r, ax_dqn_s, ax_dqn_c, dqn_agent, ep_rewards, dqn_losses)

    # Summary panel
    ax_summ.axis('off')
    summary_lines = [
        ("PHASE 1",  f"Break-even bandwidth = {threshold_bps/1e6:.1f} Mbps"),
        ("PHASE 2",  f"N-Queens crossover at N={crossover_n or '>6'} (edge time is modelled, not measured)"),
        ("PHASE 5a", f"ODG scores {len(safe_list)} objects safe, {len(local_list)} must stay local"),
        ("PHASE 3",  f"Baseline Q-Learning: 500 updates, Q-sum still rising"),
        ("PHASE 5b", f"Secure Q-Learning: learned policy keeps MED/HIGH vuln tasks local"),
        ("PHASE 4",  f"Threshold controller: max queue {max(lyapunov_q)} vs greedy {greedy_q[-1]} (100 slots)"),
        ("PHASE 5c", f"DQN over continuous state (5 inputs); trained on synthetic reward rules"),
    ]
    colors_map = {"PHASE 1":"#1f4e79","PHASE 2":"#375623","PHASE 5a":"#6B2C91",
                  "PHASE 3":"#C55A11","PHASE 5b":"#C00000",
                  "PHASE 4":"#375623","PHASE 5c":"#6B2C91"}
    ax_summ.text(0.5, 1.0, "Pipeline Results Summary",
                 ha='center', va='top', transform=ax_summ.transAxes,
                 fontsize=12, fontweight='bold', color='#1f4e79')
    for i, (phase, text) in enumerate(summary_lines):
        y = 0.88 - i * 0.125
        ax_summ.text(0.02, y, phase, transform=ax_summ.transAxes,
                     fontsize=9, fontweight='bold', color=colors_map[phase],
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor=colors_map[phase], alpha=0.12))
        ax_summ.text(0.18, y, text, transform=ax_summ.transAxes,
                     fontsize=9, color='#333333', va='center')

    # Pipeline flow label at bottom
    fig.text(0.5, 0.002,
             "FLOW:  Phase1(BW Threshold) → Phase2(Task Classify) → "
             "Phase5a(ODG Vuln) → Phase5b(Secure Q-Agent) → "
             "Phase4(Queue Stability) → Phase5c(DQN Continuous)",
             ha='center', fontsize=8.5, color='#333',
             style='italic',
             bbox=dict(facecolor='#f0f0f0', alpha=0.6, boxstyle='round'))

    plt.savefig('adaptive_offloading_full_pipeline.png', dpi=150, bbox_inches='tight')
    print("[Plots] Saved → adaptive_offloading_full_pipeline.png")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_full_pipeline()
