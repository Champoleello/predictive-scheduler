#!/usr/bin/env python3
"""
CONFRONTO DIRETTO: LangGraph (reattivo) vs Scheduler Predittivo
================================================================

Stesso task, stesse condizioni, stesso "provider" simulato con rate limit:

  • LANGGRAPH (baseline): agente costruito con LangGraph + checkpointer
    ufficiale (MemorySaver). È il paradigma REATTIVO: LangGraph salva lo
    stato dopo ogni nodo, ma non guarda i token residui → quando il
    provider risponde 429 il grafo fallisce con un'eccezione. Il recupero
    riparte dall'ultimo checkpoint di nodo MA deve re-inviare l'intero
    contesto (cold start del prefill) e perde il lavoro del nodo fallito.

  • SCHEDULER PREDITTIVO: guarda i token residui PRIMA di ogni chiamata
    e sospende con grazia prima del 429. Nessuna eccezione, nessun lavoro
    perso, ripresa senza rifare passi.

Misure (N run per configurazione): errori 429 subiti, passi ri-eseguiti,
token sprecati in re-prefill, interventi di recupero necessari.

Nota: il modello è simulato (MockBackend) per isolare il confronto sul
MECCANISMO di persistenza, a parità di consumi.
"""

import random
import statistics
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import AgentState, MockBackend, PredictiveScheduler

N_RUNS = 200
N_STEPS = 14          # passi del task
BUDGET = 3000         # token per finestra di rate limit (si resetta a ogni "ripresa")


# ---------------------------------------------------------------------------
# Provider simulato con rate limit vero e proprio (risponde 429)
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    pass


class SimulatedProvider:
    def __init__(self, budget: int, seed: int):
        self.rng = random.Random(seed)
        self.budget = budget
        self.remaining = budget

    def reset_window(self):
        self.remaining = self.budget

    def call(self, context_tokens: int) -> int:
        """Costo del passo = prefill dell'intero contesto + generazione."""
        cost = context_tokens + int(self.rng.uniform(180, 420))
        if cost > self.remaining:
            raise RateLimitError(f"429: serve {cost}, restano {self.remaining}")
        self.remaining -= cost
        return cost


# ---------------------------------------------------------------------------
# 1) Agente LANGGRAPH con checkpointer ufficiale (paradigma reattivo)
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    step: int
    context_tokens: int
    total_cost: int


def run_langgraph(seed: int):
    provider = SimulatedProvider(BUDGET, seed)

    def agent_node(state: GraphState) -> GraphState:
        cost = provider.call(state["context_tokens"])
        return {"step": state["step"] + 1,
                "context_tokens": state["context_tokens"] + 90,   # la storia cresce
                "total_cost": state["total_cost"] + cost}

    g = StateGraph(GraphState)
    g.add_node("agent", agent_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", lambda s: END if s["step"] >= N_STEPS else "agent")
    app = g.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": f"run-{seed}"},
           "recursion_limit": N_STEPS * 3 + 10}
    state = {"step": 0, "context_tokens": 400, "total_cost": 0}

    crashes = 0
    wasted_prefill = 0
    while True:
        try:
            app.invoke(state, cfg)
            break
        except RateLimitError:
            # CRASH: LangGraph ha lo stato dell'ultimo nodo completato,
            # ma il recupero richiede (a) attendere il reset della finestra,
            # (b) re-inviare l'INTERO contesto al modello (cold start prefill).
            crashes += 1
            saved = app.get_state(cfg).values or state
            wasted_prefill += saved.get("context_tokens", 0)   # prefill da rifare
            provider.reset_window()
            state = None   # LangGraph riprende dal checkpoint del thread
            if crashes > 10:
                break
    final = app.get_state(cfg).values
    return {"crashes": crashes, "wasted": wasted_prefill,
            "steps": final.get("step", 0), "recoveries": crashes}


# ---------------------------------------------------------------------------
# 2) Scheduler PREDITTIVO sulle stesse condizioni
# ---------------------------------------------------------------------------

def run_predictive(seed: int):
    provider = SimulatedProvider(BUDGET, seed)
    backend = MockBackend(avg_tokens=300)
    sch = PredictiveScheduler(backend=backend, initial_remaining_tokens=BUDGET,
                              safety_factor_k=2.0, seed=seed)
    context_tokens, step = 400, 0
    crashes = wasted = suspensions = 0

    while step < N_STEPS:
        # stima e decisione PRIMA della chiamata (usa il budget del provider)
        sch.remaining_tokens = provider.remaining
        estimated = context_tokens + 300
        if sch.remaining_tokens < estimated + sch.safety_factor_k * sch.sigma(estimated):
            # sospensione graceful: nessun lavoro perso; alla "ripresa"
            # la finestra è resettata e la KV-cache evita il re-prefill
            suspensions += 1
            provider.reset_window()
            continue
        try:
            actual = provider.call(context_tokens)
            sch.historical_consumption.append(actual)
            step += 1
            context_tokens += 90
        except RateLimitError:
            crashes += 1              # stima sbagliata: contatelo onestamente
            wasted += context_tokens
            provider.reset_window()
    return {"crashes": crashes, "wasted": wasted, "steps": step,
            "recoveries": suspensions}


# ---------------------------------------------------------------------------
# Confronto
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(f"  LANGGRAPH (reattivo) vs SCHEDULER PREDITTIVO — {N_RUNS} run ciascuno")
    print(f"  Task: {N_STEPS} passi | budget per finestra: {BUDGET} token")
    print("=" * 78)

    lg = [run_langgraph(s) for s in range(N_RUNS)]
    pr = [run_predictive(s) for s in range(N_RUNS)]

    def agg(rows, key):
        return statistics.mean(r[key] for r in rows)

    print(f"\n{'':32s} {'LangGraph':>12s} {'Predittivo':>12s}")
    print("-" * 60)
    print(f"{'Errori 429 subiti (media/run)':32s} {agg(lg,'crashes'):12.2f} {agg(pr,'crashes'):12.2f}")
    print(f"{'Run completati senza errori':32s} {sum(r['crashes']==0 for r in lg)/N_RUNS:12.0%} {sum(r['crashes']==0 for r in pr)/N_RUNS:12.0%}")
    print(f"{'Token sprecati in re-prefill':32s} {agg(lg,'wasted'):12.0f} {agg(pr,'wasted'):12.0f}")
    print(f"{'Sospensioni/recuperi (media)':32s} {agg(lg,'recoveries'):12.2f} {agg(pr,'recoveries'):12.2f}")

    import json
    with open("risultati_confronto_langgraph.json", "w") as f:
        json.dump({"langgraph": lg[:20], "predittivo": pr[:20],
                   "aggregati": {
                       "lg_429": agg(lg, "crashes"), "pr_429": agg(pr, "crashes"),
                       "lg_wasted": agg(lg, "wasted"), "pr_wasted": agg(pr, "wasted"),
                   }}, f, indent=2)
    print("\nSalvato in risultati_confronto_langgraph.json")
    print("\nLettura: LangGraph *sopravvive* ai 429 grazie al checkpointer (non perde")
    print("lo stato logico), ma li SUBISCE: ogni recupero costa un'eccezione, l'attesa")
    print("del reset e il re-prefill dell'intero contesto. Il predittivo li PREVIENE.")


if __name__ == "__main__":
    main()
