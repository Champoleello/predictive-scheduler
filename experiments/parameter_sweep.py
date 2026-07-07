#!/usr/bin/env python3
"""
Experiments on the scheduler parameters (for the report, §8.2)
=================================================================

Two experiments, each repeated over many simulations with different seeds:

  EXPERIMENT A — Predictive vs Reactive
    Compares the predictive scheduler with a "naive" agent that keeps going
    until it slams into the rate limit (429 error).
    Measures: crash rate and wasted tokens (cold start = re-sending the context).

  EXPERIMENT B — Sweep of the safety factor k
    Varies k (how conservative the scheduler is) and measures the trade-off:
    low k  → more steps completed but crash risk;
    high k → zero crashes but premature suspensions (false positives).

Output: experiment_results.csv + PNG charts.
"""

import csv
import random
import statistics
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import (
    AgentState, MockBackend, PredictiveScheduler,
)

N_RUNS = 300          # simulations per configuration
MAX_STEPS = 60


# ---------------------------------------------------------------------------
# Simulation of a PREDICTIVE run (scheduler v4)
# ---------------------------------------------------------------------------

def run_predictive(seed: int, budget: int, k: float):
    random.seed(seed)
    scheduler = PredictiveScheduler(
        backend=MockBackend(avg_tokens=320, name="std"),
        economy_backend=MockBackend(avg_tokens=140, name="eco"),
        initial_remaining_tokens=budget,
        safety_factor_k=k,
    )
    state = AgentState(
        task_description="simulated task",
        messages=[{"role": "user", "content": "Run the long simulated task " * 5}],
    )
    crashed = False
    for _ in range(MAX_STEPS):
        d = scheduler.execute_step(state)
        if d["actual"] > 0 and d["actual"] > d["remaining"] + d["actual"]:
            crashed = True  # should never happen
            break
        if d["action"] == "checkpoint":
            break
    m = scheduler.metrics_summary()
    return {
        "steps": state.step,
        "crashed": crashed or m["would_have_crashed"] > 0,
        "checkpointed": m["checkpoint"] > 0,
        "false_positive": m["false_positive_checkpoints"] > 0,
        "true_positive": m["true_positive_checkpoints"] > 0,
        # tokens "wasted" by the predictive agent = unused remaining budget at suspension
        "wasted_tokens": scheduler.remaining_tokens if m["checkpoint"] else 0,
    }


# ---------------------------------------------------------------------------
# Simulation of a REACTIVE run (baseline: no prediction)
# ---------------------------------------------------------------------------

def run_reactive(seed: int, budget: int):
    random.seed(seed)
    backend = MockBackend(avg_tokens=320, name="std")
    remaining = budget
    messages = [{"role": "user", "content": "Run the long simulated task " * 5}]
    steps, crashed, wasted = 0, False, 0
    for _ in range(MAX_STEPS):
        remaining = max(0, remaining + random.randint(-100, 100))  # simulated headers
        response, actual = backend.generate(messages)
        if actual > remaining:
            # 429 CRASH → cold start: the whole context must be re-sent on restart
            crashed = True
            wasted = sum(backend.estimate_tokens(m["content"]) for m in messages)
            break
        remaining -= actual
        steps += 1
        messages.append({"role": "assistant", "content": response[:350]})
    return {"steps": steps, "crashed": crashed, "wasted_tokens": wasted}


# ---------------------------------------------------------------------------
# EXPERIMENT A — Predictive vs Reactive (fixed k = 2)
# ---------------------------------------------------------------------------

def experiment_a():
    rows = []
    for budget in (4000, 6500, 10000):
        pred = [run_predictive(s, budget, k=2.0) for s in range(N_RUNS)]
        reac = [run_reactive(s, budget) for s in range(N_RUNS)]
        rows.append({
            "budget": budget,
            "pred_crash_rate": sum(r["crashed"] for r in pred) / N_RUNS,
            "reac_crash_rate": sum(r["crashed"] for r in reac) / N_RUNS,
            "pred_avg_steps": statistics.mean(r["steps"] for r in pred),
            "reac_avg_steps": statistics.mean(r["steps"] for r in reac),
            "pred_avg_waste": statistics.mean(r["wasted_tokens"] for r in pred),
            "reac_avg_waste": statistics.mean(r["wasted_tokens"] for r in reac),
        })
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT B — Sweep of k (fixed budget = 6500)
# ---------------------------------------------------------------------------

def experiment_b():
    rows = []
    for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        runs = [run_predictive(s, budget=6500, k=k) for s in range(N_RUNS)]
        n_cp = sum(r["checkpointed"] for r in runs)
        n_tp = sum(r["true_positive"] for r in runs)
        n_fp = sum(r["false_positive"] for r in runs)
        rows.append({
            "k": k,
            "avg_steps": statistics.mean(r["steps"] for r in runs),
            "crash_rate": sum(r["crashed"] for r in runs) / N_RUNS,
            "precision": (n_tp / n_cp) if n_cp else None,
            "false_positive_rate": (n_fp / n_cp) if n_cp else None,
            "avg_wasted_tokens": statistics.mean(r["wasted_tokens"] for r in runs),
        })
    return rows


# ---------------------------------------------------------------------------
# Charts and CSV
# ---------------------------------------------------------------------------

def make_charts(rows_a, rows_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Chart A: predictive vs reactive crash rate ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    budgets = [r["budget"] for r in rows_a]
    x = range(len(budgets))
    w = 0.35
    ax1.bar([i - w/2 for i in x], [r["reac_crash_rate"]*100 for r in rows_a], w,
            label="Reactive (baseline)", color="#d9534f")
    ax1.bar([i + w/2 for i in x], [r["pred_crash_rate"]*100 for r in rows_a], w,
            label="Predictive (v4)", color="#5cb85c")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(budgets)
    ax1.set_xlabel("Initial budget (tokens)"); ax1.set_ylabel("Crash rate (%)")
    ax1.set_title("429 crashes: predictive vs reactive"); ax1.legend()

    ax2.bar([i - w/2 for i in x], [r["reac_avg_waste"] for r in rows_a], w,
            label="Reactive (cold start)", color="#d9534f")
    ax2.bar([i + w/2 for i in x], [r["pred_avg_waste"] for r in rows_a], w,
            label="Predictive (unused margin)", color="#5cb85c")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(budgets)
    ax2.set_xlabel("Initial budget (tokens)"); ax2.set_ylabel("Wasted tokens (mean)")
    ax2.set_title("Token waste per run"); ax2.legend()
    fig.tight_layout()
    fig.savefig("chart_A_predictive_vs_reactive.png", dpi=150)

    # --- Chart B: effect of the safety factor k ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = [r["k"] for r in rows_b]
    ax1.plot(ks, [r["avg_steps"] for r in rows_b], "o-", color="#337ab7", label="Steps completed (mean)")
    ax1.set_xlabel("Safety factor k"); ax1.set_ylabel("Steps completed")
    ax1.set_title("k vs productivity"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(ks, [(r["precision"] or 0)*100 for r in rows_b], "o-",
             color="#5cb85c", label="Suspension precision (%)")
    ax2.plot(ks, [(r["false_positive_rate"] or 0)*100 for r in rows_b], "s--",
             color="#f0ad4e", label="False positive rate (%)")
    ax2.plot(ks, [r["crash_rate"]*100 for r in rows_b], "^:",
             color="#d9534f", label="Crash rate (%)")
    ax2.set_xlabel("Safety factor k"); ax2.set_ylabel("%")
    ax2.set_title("k vs suspension quality"); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig("chart_B_sweep_k.png", dpi=150)


def save_csv(rows_a, rows_b):
    with open("experiment_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXPERIMENT A — Predictive vs Reactive (k=2)"])
        w.writerow(rows_a[0].keys())
        for r in rows_a:
            w.writerow(r.values())
        w.writerow([])
        w.writerow(["EXPERIMENT B — Sweep of k (budget=6500)"])
        w.writerow(rows_b[0].keys())
        for r in rows_b:
            w.writerow(r.values())


if __name__ == "__main__":
    print(f"Running {N_RUNS} simulations per configuration...")
    rows_a = experiment_a()
    rows_b = experiment_b()
    save_csv(rows_a, rows_b)
    make_charts(rows_a, rows_b)

    print("\n=== EXPERIMENT A — Predictive vs Reactive (k=2) ===")
    for r in rows_a:
        print(f"budget {r['budget']:5d} | crash: reactive {r['reac_crash_rate']:5.1%} vs predictive {r['pred_crash_rate']:5.1%}"
              f" | mean waste: {r['reac_avg_waste']:6.0f} vs {r['pred_avg_waste']:6.0f} tokens")

    print("\n=== EXPERIMENT B — Sweep of k (budget=6500) ===")
    for r in rows_b:
        p = f"{r['precision']:.0%}" if r['precision'] is not None else "n/a"
        fp = f"{r['false_positive_rate']:.0%}" if r['false_positive_rate'] is not None else "n/a"
        print(f"k={r['k']:.1f} | mean steps: {r['avg_steps']:5.1f} | crash: {r['crash_rate']:5.1%}"
              f" | precision: {p} | false positive: {fp}")

    print("\nFiles generated: experiment_results.csv, chart_A_predictive_vs_reactive.png, chart_B_sweep_k.png")
