"""Unit tests for the quantities the paper states. Run from simulation/:  python -m pytest -q"""
import numpy as np

from mecsim import config as C
from mecsim import odg, queueing
from mecsim.agents import always_offload, evaluate, gated
from mecsim.env import MECEnv, reward_terms


def test_breakeven_matches_paper():
    assert abs(C.R_STAR / 1e6 - 13.714) < 1e-3
    assert abs(C.R_STAR_SETUP / 1e6 - 15.0) < 1e-6


def test_breakeven_independent_of_complexity():
    # offloading is faster exactly when rate > R*, for any task size (no setup delay, no queue)
    for c in (0.1, 0.5, 1.0):
        n, m = C.task_size(c)
        lo, hi = C.R_STAR * 0.99, C.R_STAR * 1.01
        t_loc = n / C.F_LOCAL
        assert m / lo + n / C.F_REMOTE > t_loc > m / hi + n / C.F_REMOTE


def test_deterministic_queue_matches_paper():
    d = queueing.deterministic()
    assert d["greedy"][-1] == 297 and max(d["controlled"]) == 21


def test_dpp_queue_bound():
    # Q never exceeds V * b_max + (largest arrival burst); with bmax = 1 and V = 20 this stays small
    rng = np.random.default_rng(0)
    r = queueing.simulate("dpp", rng, slots=3000, V=20)
    assert r["max_q"] <= 20 * 1.0 + 40


def test_odg_example_scores():
    g = odg.example_app()
    assert g.ratio_score("getUserID") == 1.0
    assert abs(g.ratio_score("syncToCloud") - 2 / 7) < 1e-9
    assert g.taint_score("syncToCloud") == 1.0          # transitively reaches getUserID
    assert g.taint_score("calculateBMI") == 0.0


def test_gated_policy_never_offloads_sensitive():
    r = evaluate(gated(always_offload(), "g"), env_seed=1, episodes=30)
    assert r["viol_odg_per_task"] == 0.0 and r["viol_high_rate"] == 0.0


def test_env_reward_prefers_offload_for_heavy_task_on_fast_link():
    r_local = reward_terms(0.8, 1.0, 0.0, 80e6, 0.1, 0)[0]
    r_off = reward_terms(0.8, 1.0, 0.0, 80e6, 0.1, 1)[0]
    assert r_off > r_local


def test_env_episode_terminates():
    env = MECEnv(0)
    env.reset()
    for _ in range(200):
        _, _, done, _ = env.step(0)
        if done:
            break
    assert done
