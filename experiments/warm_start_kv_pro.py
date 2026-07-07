#!/usr/bin/env python3
"""
TRUE WARM START — PRO experiment: thinking model + large task
==================================================================

Differences from the base experiment (warm_start_kv.py):

  • THINKING model (Qwen3): reasons inside <think>...</think> before
    answering — chosen automatically based on the Mac's RAM.
  • LARGE task: ~7,000-token context (vs 2,700) — the on-disk KV-cache
    weighs ~1 GB and the cold/warm gap becomes far more visible.
  • REPEATED measurements: every resume (cold and warm) is measured N times
    and the mean is reported — a more exact evaluation.
  • Prompt built with the model's official chat template
    (/apply-template endpoint), so thinking activates correctly.

Prerequisite: a running llama-server.
"""

import json
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("'requests' is missing. Run: pip3 install requests")
    sys.exit(1)

SERVER = "http://127.0.0.1:8080"
KV_FILE = "agent_kv_pro.bin"
REPS = 2          # repetitions per measurement (raise to 3 for extra precision)
N_PREDICT = 640   # room for thinking + answer


# ---------------------------------------------------------------------------
# Server API
# ---------------------------------------------------------------------------

def server_ok() -> bool:
    try:
        return requests.get(f"{SERVER}/health", timeout=5).status_code == 200
    except Exception:
        return False


def apply_template(messages) -> str:
    """Uses the model's official chat template (activates thinking)."""
    try:
        r = requests.post(f"{SERVER}/apply-template", json={"messages": messages}, timeout=30)
        if r.status_code == 200:
            return r.json()["prompt"]
    except Exception:
        pass
    # Fallback: ChatML template (Qwen's)
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def completion(prompt: str, n_predict: int = N_PREDICT):
    r = requests.post(f"{SERVER}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.6, "top_p": 0.95,   # recommended for Qwen3 thinking
        "cache_prompt": True,
        "id_slot": 0,
    }, timeout=1800)
    r.raise_for_status()
    d = r.json()
    return d.get("content", ""), d.get("timings", {})


def slot_action(action: str, quiet: bool = False):
    payload = {"filename": KV_FILE} if action in ("save", "restore") else {}
    t0 = time.time()
    r = requests.post(f"{SERVER}/slots/0?action={action}", json=payload, timeout=1800)
    ms = (time.time() - t0) * 1000
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if not quiet:
        keys = {k: v for k, v in body.items() if k in ("n_saved", "n_erased", "n_restored")}
        print(f"      [debug {action}] {keys or body}", flush=True)
    r.raise_for_status()
    return ms, body


def kv_file_size_mb() -> float:
    import os
    path = os.path.join("kv_cache", KV_FILE)
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


# ---------------------------------------------------------------------------
# Large task: a ~120-step agent log (≈7,000 tokens)
# ---------------------------------------------------------------------------

def build_big_context() -> str:
    system = ("You are an autonomous agent expert in software architecture. "
              "You analyze distributed systems and produce justified, "
              "prioritized refactoring recommendations.")
    log = []
    for i in range(1, 121):
        log.append(
            f"STEP {i}: inspected module service_{i:03d}. "
            f"Direct dependencies: {2 + i % 7}; p95 latency: {60 + (i * 7) % 240} ms; "
            f"error rate: {round(0.1 + (i % 9) * 0.4, 1)}%; test coverage: {30 + (i * 13) % 60}%. "
            f"Observations: {'tight coupling to the authentication gateway' if i % 3 == 0 else 'incomplete error handling on async paths' if i % 3 == 1 else 'N+1 queries to the orders database and no cache'}. "
        )
    question = ("Analyze the entire log above. Identify the 3 most critical modules "
                "to refactor, explain why those three by comparing the metrics, "
                "and propose the order of intervention.")
    user = "INSPECTION LOG:\n" + "\n".join(log) + "\n\n" + question
    return apply_template([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


# ---------------------------------------------------------------------------
# Experiment with repeated measurements
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  TRUE WARM START — PRO: thinking model, large task, repeated measures")
    print("=" * 78)

    if not server_ok():
        print("ERROR: llama-server is not responding. Start it first.")
        sys.exit(1)

    prompt = build_big_context()
    print(f"\nContext built: ~{len(prompt) // 4} estimated tokens")

    # --- 1. Work session (with thinking) -----------------------------------
    print("\n[1] Work session: the model reasons over the large task...", flush=True)
    text, t = completion(prompt)
    print(f"    Prefill: {t.get('prompt_n', '?')} tokens in {t.get('prompt_ms', 0) / 1000:.1f} s "
          f"({t.get('prompt_per_second', 0):.0f} tok/s)")
    if "<think>" in text:
        think = text.split("<think>")[1].split("</think>")[0].strip()
        answer = text.split("</think>")[-1].strip()
        print(f"    Thinking (excerpt): {think[:140]}...")
        print(f"    Answer (excerpt): {answer[:140]}...")
    else:
        print(f"    Answer (excerpt): {text.strip()[:140]}...")

    # --- 2. KV checkpoint to disk -------------------------------------------
    print("\n[2] Checkpoint: serializing the KV-cache to disk...", flush=True)
    ms_save, _ = slot_action("save")
    print(f"    Saved in {ms_save:.0f} ms | file: {kv_file_size_mb():.0f} MB")

    # --- 3+4. Repeated measurements: COLD vs WARM ----------------------------
    cold_ms, cold_n = [], []
    warm_ms, warm_n, restore_ms = [], [], []

    for rep in range(1, REPS + 1):
        print(f"\n[3] COLD resume (measure {rep}/{REPS}): erase + full re-prefill...", flush=True)
        slot_action("erase", quiet=True)
        _, tc = completion(prompt, n_predict=8)   # few tokens: we measure the prefill
        cold_ms.append(tc.get("prompt_ms", 0)); cold_n.append(tc.get("prompt_n", 0))
        print(f"    {tc.get('prompt_n', '?')} tokens recomputed in {tc.get('prompt_ms', 0) / 1000:.1f} s")

        print(f"[4] WARM resume (measure {rep}/{REPS}): erase + restore from disk...", flush=True)
        slot_action("erase", quiet=True)
        ms_r, _ = slot_action("restore", quiet=True)
        _, tw = completion(prompt, n_predict=8)
        warm_ms.append(tw.get("prompt_ms", 0)); warm_n.append(tw.get("prompt_n", 0)); restore_ms.append(ms_r)
        print(f"    restore {ms_r:.0f} ms + {tw.get('prompt_n', '?')} tokens in {tw.get('prompt_ms', 0):.0f} ms")

    # --- Summary --------------------------------------------------------------
    c_ms, c_n = statistics.mean(cold_ms), statistics.mean(cold_n)
    w_ms, w_n = statistics.mean(warm_ms), statistics.mean(warm_n)
    r_ms = statistics.mean(restore_ms)
    warm_total = w_ms + r_ms

    print("\n" + "=" * 78)
    print(f"  RESULT (mean over {REPS} measurements)")
    print("=" * 78)
    print(f"  COLD resume: {c_n:7.0f} tokens recomputed | {c_ms / 1000:8.1f} s")
    print(f"  WARM resume: {w_n:7.0f} tokens recomputed | {warm_total / 1000:8.2f} s (restore {r_ms / 1000:.2f} s)")
    print(f"\n  → Tokens saved: {c_n - w_n:.0f} ({(1 - w_n / max(c_n, 1)) * 100:.0f}%)")
    print(f"  → Resume speedup: {c_ms / max(warm_total, 1):.1f}×")
    print(f"  → On-disk KV-cache: {kv_file_size_mb():.0f} MB")
    print("=" * 78)

    with open("warm_start_kv_pro_results.json", "w") as f:
        json.dump({"cold_ms": cold_ms, "cold_n": cold_n, "warm_ms": warm_ms,
                   "warm_n": warm_n, "restore_ms": restore_ms,
                   "kv_mb": kv_file_size_mb(), "reps": REPS}, f, indent=2)
    print("\nSaved to warm_start_kv_pro_results.json")


if __name__ == "__main__":
    main()
