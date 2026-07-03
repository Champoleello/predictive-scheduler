#!/usr/bin/env python3
"""
Esperimenti sui parametri dello scheduler (per il rapporto, §8.2)
=================================================================

Due esperimenti, ognuno ripetuto su molte simulazioni con seed diversi:

  ESPERIMENTO A — Predittivo vs Reattivo
    Confronta lo scheduler predittivo con un agente "ingenuo" che continua
    finché non sbatte contro il rate limit (errore 429).
    Misura: crash rate e token sprecati (cold start = re-invio del contesto).

  ESPERIMENTO B — Sweep del fattore di sicurezza k
    Fa variare k (quanto è prudente lo scheduler) e misura il trade-off:
    k basso  → più passi completati ma rischio di crash;
    k alto   → zero crash ma sospensioni premature (false positive).

Output: esperimenti_risultati.csv + grafici PNG.
"""

import csv
import random
import statistics
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import (
    AgentState, MockBackend, PredictiveScheduler,
)

N_RUNS = 300          # simulazioni per configurazione
MAX_STEPS = 60


# ---------------------------------------------------------------------------
# Simulazione di un run PREDITTIVO (scheduler v4)
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
        task_description="task simulato",
        messages=[{"role": "user", "content": "Esegui il task lungo simulato " * 5}],
    )
    crashed = False
    for _ in range(MAX_STEPS):
        d = scheduler.execute_step(state)
        if d["actual"] > 0 and d["actual"] > d["remaining"] + d["actual"]:
            crashed = True  # non dovrebbe mai accadere
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
        # token "sprecati" dal predittivo = budget residuo non usato alla sospensione
        "wasted_tokens": scheduler.remaining_tokens if m["checkpoint"] else 0,
    }


# ---------------------------------------------------------------------------
# Simulazione di un run REATTIVO (baseline: nessuna previsione)
# ---------------------------------------------------------------------------

def run_reactive(seed: int, budget: int):
    random.seed(seed)
    backend = MockBackend(avg_tokens=320, name="std")
    remaining = budget
    messages = [{"role": "user", "content": "Esegui il task lungo simulato " * 5}]
    steps, crashed, wasted = 0, False, 0
    for _ in range(MAX_STEPS):
        remaining = max(0, remaining + random.randint(-100, 100))  # header simulati
        response, actual = backend.generate(messages)
        if actual > remaining:
            # CRASH 429 → cold start: l'intero contesto va re-inviato al riavvio
            crashed = True
            wasted = sum(backend.estimate_tokens(m["content"]) for m in messages)
            break
        remaining -= actual
        steps += 1
        messages.append({"role": "assistant", "content": response[:350]})
    return {"steps": steps, "crashed": crashed, "wasted_tokens": wasted}


# ---------------------------------------------------------------------------
# ESPERIMENTO A — Predittivo vs Reattivo (k = 2 fisso)
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
# ESPERIMENTO B — Sweep di k (budget = 6500 fisso)
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
# Grafici e CSV
# ---------------------------------------------------------------------------

def make_charts(rows_a, rows_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Grafico A: crash rate predittivo vs reattivo ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    budgets = [r["budget"] for r in rows_a]
    x = range(len(budgets))
    w = 0.35
    ax1.bar([i - w/2 for i in x], [r["reac_crash_rate"]*100 for r in rows_a], w,
            label="Reattivo (baseline)", color="#d9534f")
    ax1.bar([i + w/2 for i in x], [r["pred_crash_rate"]*100 for r in rows_a], w,
            label="Predittivo (v4)", color="#5cb85c")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(budgets)
    ax1.set_xlabel("Budget iniziale (token)"); ax1.set_ylabel("Crash rate (%)")
    ax1.set_title("Crash 429: predittivo vs reattivo"); ax1.legend()

    ax2.bar([i - w/2 for i in x], [r["reac_avg_waste"] for r in rows_a], w,
            label="Reattivo (cold start)", color="#d9534f")
    ax2.bar([i + w/2 for i in x], [r["pred_avg_waste"] for r in rows_a], w,
            label="Predittivo (margine non usato)", color="#5cb85c")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(budgets)
    ax2.set_xlabel("Budget iniziale (token)"); ax2.set_ylabel("Token sprecati (media)")
    ax2.set_title("Spreco di token per run"); ax2.legend()
    fig.tight_layout()
    fig.savefig("grafico_A_predittivo_vs_reattivo.png", dpi=150)

    # --- Grafico B: effetto del fattore di sicurezza k ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = [r["k"] for r in rows_b]
    ax1.plot(ks, [r["avg_steps"] for r in rows_b], "o-", color="#337ab7", label="Passi completati (media)")
    ax1.set_xlabel("Fattore di sicurezza k"); ax1.set_ylabel("Passi completati")
    ax1.set_title("k vs produttività"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(ks, [(r["precision"] or 0)*100 for r in rows_b], "o-",
             color="#5cb85c", label="Suspension precision (%)")
    ax2.plot(ks, [(r["false_positive_rate"] or 0)*100 for r in rows_b], "s--",
             color="#f0ad4e", label="False positive rate (%)")
    ax2.plot(ks, [r["crash_rate"]*100 for r in rows_b], "^:",
             color="#d9534f", label="Crash rate (%)")
    ax2.set_xlabel("Fattore di sicurezza k"); ax2.set_ylabel("%")
    ax2.set_title("k vs qualità delle sospensioni"); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig("grafico_B_sweep_k.png", dpi=150)


def save_csv(rows_a, rows_b):
    with open("esperimenti_risultati.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ESPERIMENTO A — Predittivo vs Reattivo (k=2)"])
        w.writerow(rows_a[0].keys())
        for r in rows_a:
            w.writerow(r.values())
        w.writerow([])
        w.writerow(["ESPERIMENTO B — Sweep di k (budget=6500)"])
        w.writerow(rows_b[0].keys())
        for r in rows_b:
            w.writerow(r.values())


if __name__ == "__main__":
    print(f"Eseguo {N_RUNS} simulazioni per configurazione...")
    rows_a = experiment_a()
    rows_b = experiment_b()
    save_csv(rows_a, rows_b)
    make_charts(rows_a, rows_b)

    print("\n=== ESPERIMENTO A — Predittivo vs Reattivo (k=2) ===")
    for r in rows_a:
        print(f"budget {r['budget']:5d} | crash: reattivo {r['reac_crash_rate']:5.1%} vs predittivo {r['pred_crash_rate']:5.1%}"
              f" | spreco medio: {r['reac_avg_waste']:6.0f} vs {r['pred_avg_waste']:6.0f} token")

    print("\n=== ESPERIMENTO B — Sweep di k (budget=6500) ===")
    for r in rows_b:
        p = f"{r['precision']:.0%}" if r['precision'] is not None else "n/a"
        fp = f"{r['false_positive_rate']:.0%}" if r['false_positive_rate'] is not None else "n/a"
        print(f"k={r['k']:.1f} | passi medi: {r['avg_steps']:5.1f} | crash: {r['crash_rate']:5.1%}"
              f" | precision: {p} | false positive: {fp}")

    print("\nFile generati: esperimenti_risultati.csv, grafico_A_predittivo_vs_reattivo.png, grafico_B_sweep_k.png")
