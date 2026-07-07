#!/usr/bin/env python3
"""
TRUE WARM START — on-disk KV-cache experiment with llama.cpp
==============================================================

This experiment measures the difference between the three restart levels
(§8.4 of the report) using a REAL model running locally:

  1. WORK SESSION         → the agent builds a long context (the model
                            "digests" it into its KV-cache)
  2. CHECKPOINT           → the KV-cache is serialized TO DISK
                            (POST /slots/0?action=save)
  3. SIMULATED RESTART    → the model's memory is wiped
                            (POST /slots/0?action=erase)
  4. COLD RESUME          → the whole context is re-sent: the model must
                            recompute the full prefill (LOGICAL warm start)
  5. WARM RESUME          → the KV-cache is restored from disk
                            (POST /slots/0?action=restore): the prefill is
                            not recomputed (TRUE warm start)

The comparison between (4) and (5) — re-processed tokens and milliseconds —
is the number that quantifies the "zero-cost paradox" (§8.4).

Prerequisite: a running llama-server.
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    print("'requests' is missing. Run: pip3 install requests")
    sys.exit(1)

SERVER = "http://127.0.0.1:8080"
KV_FILE = "agent_kv.bin"           # KV-cache file inside --slot-save-path
CHECKPOINT_FILE = "checkpoint_kv.json"


# ---------------------------------------------------------------------------
# Calls to the llama.cpp server
# ---------------------------------------------------------------------------

def server_ok() -> bool:
    try:
        return requests.get(f"{SERVER}/health", timeout=5).status_code == 200
    except Exception:
        return False


def completion(prompt: str, n_predict: int = 48):
    """Requests a generation and returns (text, timings)."""
    r = requests.post(f"{SERVER}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.3,
        "cache_prompt": True,   # reuse the prefix KV-cache, if present
        "id_slot": 0,           # pin slot 0 (the one we save)
    }, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d.get("content", ""), d.get("timings", {})


def slot_action(action: str):
    """save / restore / erase of slot 0's KV-cache (with diagnostics)."""
    payload = {"filename": KV_FILE} if action in ("save", "restore") else {}
    t0 = time.time()
    r = requests.post(f"{SERVER}/slots/0?action={action}", json=payload, timeout=600)
    ms = (time.time() - t0) * 1000
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:200]}
    print(f"      [debug {action}] HTTP {r.status_code} → {json.dumps(body)[:180]}")
    r.raise_for_status()
    return ms


def kv_file_size_mb() -> float:
    import os
    path = os.path.join("kv_cache", KV_FILE)
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


# ---------------------------------------------------------------------------
# "Agent-like" context: a long history simulating many work steps
# ---------------------------------------------------------------------------

def build_agent_context() -> str:
    intro = ("You are an autonomous agent analyzing a distributed system. "
             "Below is the full log of the steps already executed.\n\n")
    steps = []
    for i in range(1, 41):
        steps.append(
            f"STEP {i}: analyzed module service_{i:02d}. "
            f"Found {3 + i % 5} dependencies, mean latency {80 + i * 3} ms, "
            f"notes: the module needs error-handling refactoring and is "
            f"tightly coupled to the authentication service. "
        )
    question = ("\n\nQUESTION: based on all the previous steps, "
                "which module is the most critical to refactor first? "
                "Answer in one sentence.")
    return intro + "\n".join(steps) + question


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  TRUE WARM START — on-disk KV-cache (llama.cpp)")
    print("=" * 78)

    if not server_ok():
        print("ERROR: llama-server is not responding at", SERVER)
        print("Start it first.")
        sys.exit(1)

    ctx = build_agent_context()
    results = {}

    # --- 1. WORK SESSION ---------------------------------------------------
    print("\n[1/5] Work session: the model digests the long context...")
    text, t = completion(ctx)
    results["work"] = t
    print(f"      Prefill: {t.get('prompt_n', '?')} tokens in {t.get('prompt_ms', 0):.0f} ms")
    print(f"      Answer: {text.strip()[:90]}...")

    # --- 2. CHECKPOINT: KV-cache TO DISK ------------------------------------
    print("\n[2/5] Checkpoint: serializing the KV-cache to disk...")
    ms = slot_action("save")
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"kv_file": KV_FILE, "context_chars": len(ctx)}, f)
    print(f"      Saved in {ms:.0f} ms | file size: {kv_file_size_mb():.1f} MB")
    if kv_file_size_mb() < 0.5:
        print("      ⚠ The file is too small: the save did NOT capture the KV-cache.")

    # --- 3. SIMULATED RESTART ------------------------------------------------
    print("\n[3/5] Simulated restart: wiping the model's memory (erase)...")
    slot_action("erase")

    # --- 4. COLD RESUME (logical warm start: full re-prefill) ----------------
    print("\n[4/5] COLD RESUME: re-sending the whole context, the model recomputes...")
    _, t_cold = completion(ctx)
    results["cold"] = t_cold
    print(f"      Re-prefill: {t_cold.get('prompt_n', '?')} tokens in {t_cold.get('prompt_ms', 0):.0f} ms")

    # --- 5. WARM RESUME (TRUE warm start: restore from disk) -----------------
    print("\n[5/5] WARM RESUME: wiping again, then restoring the KV from disk...")
    slot_action("erase")
    ms_restore = slot_action("restore")
    _, t_warm = completion(ctx)
    results["warm"] = t_warm
    print(f"      Restore from disk: {ms_restore:.0f} ms | "
          f"residual prefill: {t_warm.get('prompt_n', '?')} tokens in {t_warm.get('prompt_ms', 0):.0f} ms")

    # --- COMPARISON -----------------------------------------------------------
    cold_n, cold_ms = t_cold.get("prompt_n", 0), t_cold.get("prompt_ms", 0)
    warm_n, warm_ms = t_warm.get("prompt_n", 0), t_warm.get("prompt_ms", 0)
    warm_total = warm_ms + ms_restore

    print("\n" + "=" * 78)
    print("  RESULT — the cost of resuming (§8.4 of the report)")
    print("=" * 78)
    print(f"  COLD resume (logical warm start): {cold_n:5d} tokens recomputed | {cold_ms:8.0f} ms")
    print(f"  WARM resume (TRUE warm start):    {warm_n:5d} tokens recomputed | {warm_total:8.0f} ms (incl. restore {ms_restore:.0f} ms)")
    if warm_total > 0 and cold_ms > 0:
        print(f"\n  → Tokens saved: {cold_n - warm_n} ({(1 - warm_n / max(cold_n, 1)) * 100:.0f}%)")
        print(f"  → Resume speedup: {cold_ms / warm_total:.1f}×")
    print("=" * 78)

    with open("warm_start_kv_results.json", "w") as f:
        json.dump({"cold": t_cold, "warm": t_warm, "restore_ms": ms_restore}, f, indent=2)
    print("\nResults saved to warm_start_kv_results.json")


if __name__ == "__main__":
    main()
