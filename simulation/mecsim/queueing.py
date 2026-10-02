"""Edge-queue admission control: greedy, fixed threshold, Lyapunov drift-plus-penalty."""
import numpy as np


def deterministic(slots=100, arrival=10, capacity=7, threshold=20, throttled=4):
    """The paper's illustrative case: constant arrivals; once Q >= threshold only `throttled`
    tasks per slot are admitted and the rest run locally."""
    greedy, ctrl = [0], [0]
    redirected = 0
    for _ in range(1, slots):
        greedy.append(max(0, greedy[-1] + arrival - capacity))
        admit = arrival if ctrl[-1] < threshold else throttled
        redirected += arrival - admit
        ctrl.append(max(0, ctrl[-1] + admit - capacity))
    return {"greedy": greedy, "controlled": ctrl, "redirected": redirected,
            "params": dict(slots=slots, arrival=arrival, capacity=capacity,
                           threshold=threshold, throttled=throttled)}


def simulate(policy, rng, slots=2000, lam=10.0, capacity=7, V=20.0, theta=20, bmax=1.0):
    """Stochastic multi-user queue. Each slot Poisson(lam) tasks arrive, each with an offloading
    benefit b ~ U(0.2, bmax). The policy decides which tasks to admit; the rest run locally.

    DPP: maximise V * sum(b_admitted) - Q(t) * admitted in every slot, i.e. admit a task iff
    V * b > Q(t). Queue bound: nothing is admitted once Q >= V * bmax, so Q <= V * bmax + (max
    arrivals in one slot).
    """
    Q, qs, benefit, admitted_n, arrived_n = 0, [], 0.0, 0, 0
    for _ in range(slots):
        a = rng.poisson(lam)
        b = rng.uniform(0.2, bmax, a)
        if policy == "greedy":
            take = np.ones(a, bool)
        elif policy == "threshold":
            take = np.ones(a, bool) if Q < theta else np.zeros(a, bool)
        elif policy == "dpp":
            take = V * b > Q
        else:
            raise ValueError(policy)
        k = int(take.sum())
        benefit += b[take].sum()
        admitted_n += k
        arrived_n += a
        Q = max(0, Q + k - capacity)
        qs.append(Q)
    qs = np.array(qs)
    return {"mean_q": float(qs.mean()), "max_q": int(qs.max()), "final_q": int(qs[-1]),
            "admit_frac": admitted_n / max(arrived_n, 1), "benefit_per_slot": benefit / slots,
            "trace": qs[:300].tolist()}
