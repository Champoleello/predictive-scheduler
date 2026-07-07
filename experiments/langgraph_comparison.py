#!/usr/bin/env python3
"""
HEAD-TO-HEAD: LangGraph (reactive) vs Predictive Scheduler
================================================================

Same task, same conditions, same simulated rate-limited "provider":

  • LANGGRAPH (baseline): agent built with LangGraph + the official
    checkpointer (MemorySaver). This is the REACTIVE paradigm: LangGraph
    saves state after every node, but never looks at the remaining tokens
    → when the provider returns 429 the graph fails with an exception.
    Recovery restarts from the last node checkpoint BUT must re-send the
    whole context (prefill cold start) and loses the failed node's work.

  • PREDICTIVE SCHEDULER: looks at the remaining tokens BEFORE every call
    and suspends gracefully before the 429. No exception, no lost work,
    resume without redoing steps.

Measures (N runs per configuration): 429 errors suffered, steps re-executed,
tokens wasted on re-prefill, recovery interventions needed.

Note: the model is simulated (MockBackend) to isolate the comparison on the
persistence MECHANISM, with identical consumption.
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
N_STEPS = 14          # task steps
BUDGET = 3000         # tokens per rate-limit window (resets on every "resume")


# ---------------------------------------------------------------------------
# Simulated provider with an actual rate limit (returns 429)
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
        """Step cost = prefill of the whole context + generation."""
        cost = context_tokens + int(self.rng.uniform(180, 420))
        if cost > self.remaining:
            raise RateLimitError(f"429: needs {cost}, {self.remaining} left")
        self.remaining -= cost
        return cost


# ---------------------------------------------------------------------------
# 1) LANGGRAPH agent with the official checkpointer (reactive paradigm)
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
                "context_tokens": state["context_tokens"] + 90,   # history grows
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
            # CRASH: LangGraph holds the state of the last completed node,
            # but recovery requires (a) waiting for the window reset,
            # (b) re-sending the WHOLE context to the model (prefill cold start).
            crashes += 1
            saved = app.get_state(cfg).values or state
            wasted_prefill += saved.get("context_tokens", 0)   # prefill to redo
            provider.reset_window()
            state = None   # LangGraph resumes from the thread's checkpoint
            if crashes > 10:
                break
    final = app.get_state(cfg).values
    return {"crashes": crashes, "wasted": wasted_prefill,
            "steps": final.get("step", 0), "recoveries": crashes}


# ---------------------------------------------------------------------------
# 2) PREDICTIVE scheduler under the same conditions
# ---------------------------------------------------------------------------

def run_predictive(seed: int):
    provider = SimulatedProvider(BUDGET, seed)
    backend = MockBackend(avg_tokens=300)
    sch = PredictiveScheduler(backend=backend, initial_remaining_tokens=BUDGET,
                              safety_factor_k=2.0, seed=seed)
    context_tokens, step = 400, 0
    crashes = wasted = suspensions = 0

    while step < N_STEPS:
        # estimate and decide BEFORE the call (uses the provider's budget)
        sch.remaining_tokens = provider.remaining
        estimated = context_tokens + 300
        if sch.remaining_tokens < estimated + sch.safety_factor_k * sch.sigma(estimated):
            # graceful suspension: no lost work; on "resume" the window has
            # reset and the KV-cache avoids the re-prefill
            suspensions += 1
            provider.reset_window()
            continue
        try:
            actual = provider.call(context_tokens)
            sch.historical_consumption.append(actual)
            step += 1
            context_tokens += 90
        except RateLimitError:
            crashes += 1              # wrong estimate: count it honestly
            wasted += context_tokens
            provider.reset_window()
    return {"crashes": crashes, "wasted": wasted, "steps": step,
            "recoveries": suspensions}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(f"  LANGGRAPH (reactive) vs PREDICTIVE SCHEDULER — {N_RUNS} runs each")
    print(f"  Task: {N_STEPS} steps | budget per window: {BUDGET} tokens")
    print("=" * 78)

    lg = [run_langgraph(s) for s in range(N_RUNS)]
    pr = [run_predictive(s) for s in range(N_RUNS)]

    def agg(rows, key):
        return statistics.mean(r[key] for r in rows)

    print(f"\n{'':32s} {'LangGraph':>12s} {'Predictive':>12s}")
    print("-" * 60)
    print(f"{'429 errors suffered (mean/run)':32s} {agg(lg,'crashes'):12.2f} {agg(pr,'crashes'):12.2f}")
    print(f"{'Runs completed without errors':32s} {sum(r['crashes']==0 for r in lg)/N_RUNS:12.0%} {sum(r['crashes']==0 for r in pr)/N_RUNS:12.0%}")
    print(f"{'Tokens wasted on re-prefill':32s} {agg(lg,'wasted'):12.0f} {agg(pr,'wasted'):12.0f}")
    print(f"{'Suspensions/recoveries (mean)':32s} {agg(lg,'recoveries'):12.2f} {agg(pr,'recoveries'):12.2f}")

    import json
    with open("langgraph_comparison_results.json", "w") as f:
        json.dump({"langgraph": lg[:20], "predictive": pr[:20],
                   "aggregates": {
                       "lg_429": agg(lg, "crashes"), "pr_429": agg(pr, "crashes"),
                       "lg_wasted": agg(lg, "wasted"), "pr_wasted": agg(pr, "wasted"),
                   }}, f, indent=2)
    print("\nSaved to langgraph_comparison_results.json")
    print("\nReading: LangGraph *survives* 429s thanks to its checkpointer (it keeps")
    print("the logical state), but it SUFFERS them: every recovery costs an exception,")
    print("waiting for the reset, and re-prefilling the whole context. The predictive")
    print("scheduler PREVENTS them.")


if __name__ == "__main__":
    main()
