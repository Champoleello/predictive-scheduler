#!/usr/bin/env python3
"""
FINAL END-TO-END A/B — same task, with and without the system
==================================================================

Real thinking model (Qwen3), complex multi-step task, simulated rate limit
with budget windows. Two conditions identical in every respect:

  A) WITHOUT the system ("killed"): the agent never looks at the budget.
     When a call overshoots → 429, the session dies (the server's KV-cache
     is lost, as in a real process restart). On resume, after the window
     reset, it must RE-SEND the whole conversation (full re-prefill) and
     REDO the failed step.

  B) WITH the system: the scheduler predicts the overshoot BEFORE the
     call, saves state+KV to disk in one transaction, and on resume
     restores the KV: only the new question is recomputed.

Measures per condition: total time, seconds spent recovering, prefill
tokens recomputed during recoveries, 429 errors suffered, steps redone.

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
KV_FILE = "end_to_end_ab_kv.bin"
N_PREDICT = 512          # room for thinking + answer
BUDGET = 6500            # tokens per (simulated) rate-limit window
SAFETY_K = 2.0


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def apply_template(messages) -> str:
    try:
        r = requests.post(f"{SERVER}/apply-template", json={"messages": messages}, timeout=30)
        if r.status_code == 200:
            return r.json()["prompt"]
    except Exception:
        pass
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in messages) + "<|im_start|>assistant\n"


def completion(messages, n_predict=N_PREDICT):
    prompt = apply_template(messages)
    r = requests.post(f"{SERVER}/completion", json={
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.6, "top_p": 0.95,
        "cache_prompt": True, "id_slot": 0,
    }, timeout=3600)
    r.raise_for_status()
    d = r.json()
    t = d.get("timings", {})
    content = d.get("content", "")
    if "</think>" in content:
        content = content.split("</think>")[-1]
    # CLOUD-PROVIDER accounting: the input is charged IN FULL on every call
    # (the local cache reduces compute, not the rate-limit count).
    full_input = sum(estimate_tokens(m["content"]) for m in messages)
    used = int(full_input + t.get("predicted_n", 0))
    return content.strip(), used, t


def slot(action):
    payload = {"filename": KV_FILE} if action in ("save", "restore") else {}
    r = requests.post(f"{SERVER}/slots/0?action={action}", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# The complex task (identical in both conditions)
# ---------------------------------------------------------------------------

def registry(n=60):
    rows = []
    for i in range(1, n + 1):
        rows.append(f"MODULE service_{i:03d}: dependencies {2 + i % 7}, "
                    f"p95 latency {60 + (i * 7) % 240} ms, error rate {round(0.1 + (i % 9) * 0.4, 1)}%, "
                    f"test coverage {30 + (i * 13) % 60}%, "
                    f"{'coupled to the auth gateway' if i % 3 == 0 else 'incomplete error handling' if i % 3 == 1 else 'N+1 queries and no cache'}.")
    return "\n".join(rows)


SYSTEM = ("You are a software architect. Reason carefully and answer "
          "rigorously but concisely (max 150 words per answer).")

QUESTIONS = [
    "Analyze the registry and identify the 3 most critical modules, justifying with the metrics.",
    "For each of the 3, estimate the impact of a failure on the dependency chain.",
    "Propose the optimal refactoring order and justify it by weighing risk against cost.",
    "Write the operational plan: 3 concrete interventions for the most urgent module.",
]


def initial_messages():
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "INSPECTION REGISTRY:\n" + registry() +
             "\n\nAnswer the upcoming questions one at a time."},
            ]


# ---------------------------------------------------------------------------
# Estimation and budget (identical in both conditions; only B uses them to decide)
# ---------------------------------------------------------------------------

def estimate_tokens(text): return max(8, len(text) // 4)


def estimate_step(messages, sigma_hist):
    inp = sum(estimate_tokens(m["content"]) for m in messages)
    est = inp + N_PREDICT
    if len(sigma_hist) >= 2:
        mean = sum(sigma_hist) / len(sigma_hist)
        sigma = (sum((x - mean) ** 2 for x in sigma_hist) / len(sigma_hist)) ** 0.5
    else:
        sigma = est * 0.25
    return est, sigma


# ---------------------------------------------------------------------------
# CONDITION A — without the system: killed & cold resume
# ---------------------------------------------------------------------------

def run_condition_a():
    print("\n" + "=" * 78)
    print("  CONDITION A — WITHOUT the system (killed → cold resume)")
    print("=" * 78)
    slot("erase")
    messages = initial_messages()
    remaining = BUDGET
    stats = {"t0": time.time(), "recovery_s": 0.0, "reprefill_tokens": 0,
             "errors_429": 0, "steps_redone": 0}

    step = 0
    while step < len(QUESTIONS):
        messages.append({"role": "user", "content": QUESTIONS[step]})
        est_input = sum(estimate_tokens(m["content"]) for m in messages)

        # the agent does NOT look at the budget: it just calls
        if est_input + N_PREDICT > remaining:
            # ---- 429: the session dies ----
            stats["errors_429"] += 1
            stats["steps_redone"] += 1
            print(f"  step {step + 1}: ❌ 429! session lost, waiting for the reset...")
            remaining = BUDGET                    # window has reset
            slot("erase")                         # process dead → KV lost
            t0 = time.time()
            # cold resume: re-prefill of the WHOLE conversation (same call)
            text, used, t = completion(messages)
            dt = time.time() - t0
            stats["recovery_s"] += dt
            stats["reprefill_tokens"] += int(t.get("prompt_n", 0))
            print(f"  step {step + 1}: cold recovery — {t.get('prompt_n', 0):.0f} tok "
                  f"of re-prefill in {dt:.1f} s")
        else:
            text, used, t = completion(messages)
            print(f"  step {step + 1}: ok ({t.get('prompt_n', 0):.0f} tok prefill, "
                  f"{used} used, budget {remaining})")
        remaining -= used
        messages.append({"role": "assistant", "content": text[:600]})
        step += 1

    stats["total_s"] = time.time() - stats["t0"]
    return stats


# ---------------------------------------------------------------------------
# CONDITION B — with the system: prediction + KV-checkpoint
# ---------------------------------------------------------------------------

def run_condition_b():
    print("\n" + "=" * 78)
    print("  CONDITION B — WITH the system (prediction + KV warm start)")
    print("=" * 78)
    slot("erase")
    messages = initial_messages()
    remaining = BUDGET
    hist = []
    stats = {"t0": time.time(), "recovery_s": 0.0, "reprefill_tokens": 0,
             "errors_429": 0, "steps_redone": 0}

    step = 0
    while step < len(QUESTIONS):
        messages.append({"role": "user", "content": QUESTIONS[step]})
        est, sigma = estimate_step(messages, hist)

        if remaining < est + SAFETY_K * sigma:
            # ---- checkpoint BEFORE the error ----
            messages.pop()                        # the question is re-asked on resume
            t0 = time.time()
            info = slot("save")                   # KV to disk (+ state: here in RAM)
            print(f"  step {step + 1}: 🛑 predictive checkpoint "
                  f"({info.get('n_saved', 0)} KV cells) — waiting for the reset...")
            remaining = BUDGET
            slot("erase")                         # "process restart"
            slot("restore")                       # TRUE warm start from disk
            stats["recovery_s"] += time.time() - t0
            continue                              # no work lost

        text, used, t = completion(messages)
        hist.append(used)
        print(f"  step {step + 1}: ok ({t.get('prompt_n', 0):.0f} tok prefill, "
              f"{used} used, budget {remaining})")
        remaining -= used
        messages.append({"role": "assistant", "content": text[:600]})
        step += 1

    stats["total_s"] = time.time() - stats["t0"]
    return stats


# ---------------------------------------------------------------------------

def main():
    try:
        requests.get(f"{SERVER}/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: llama-server is not running (start it first)")
        sys.exit(1)

    print("=" * 78)
    print("  FINAL A/B — same complex task, thinking model")
    print(f"  4 analysis questions | budget per window: {BUDGET} tokens")
    print("=" * 78)

    a = run_condition_a()
    b = run_condition_b()

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"{'':38s} {'A (without)':>12s} {'B (with)':>12s}")
    print("-" * 66)
    print(f"{'429 errors suffered':38s} {a['errors_429']:12d} {b['errors_429']:12d}")
    print(f"{'Steps redone':38s} {a['steps_redone']:12d} {b['steps_redone']:12d}")
    print(f"{'Tokens re-processed in recoveries':38s} {a['reprefill_tokens']:12d} {b['reprefill_tokens']:12d}")
    print(f"{'Time spent recovering':38s} {a['recovery_s']:11.1f}s {b['recovery_s']:11.1f}s")
    print(f"{'Total task time':38s} {a['total_s']:11.1f}s {b['total_s']:11.1f}s")
    print("=" * 78)

    with open("end_to_end_ab_results.json", "w") as f:
        json.dump({"A_without": a, "B_with": b}, f, indent=2)
    print("\nSaved to end_to_end_ab_results.json")


if __name__ == "__main__":
    main()
