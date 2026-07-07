#!/usr/bin/env python3
"""
SCHEDULER + INTEGRATED KV-CHECKPOINT — the complete system
==========================================================

Up to now we had two pieces demonstrated separately:
  • the predictive scheduler that DECIDES when to stop (v4, Groq)
  • true warm start via on-disk KV-cache (warm_start_kv)

This file joins them: when the scheduler decides "checkpoint", it saves in
a SINGLE TRANSACTION both the logical state (messages, steps, idempotency)
and the model's KV-cache. On resume it restores both: the agent restarts
from the exact step AND the model does not recompute the prefill.

    decision (§4) ──► checkpoint ──► [ KV-cache on disk + JSON state ]
                                            │
    relaunch ──► restore KV + state ──► resumes at ~zero cost

Design note that emerged: with the KV-checkpoint, actions that MODIFY the
history (compress/summarize, §5.1–5.2) invalidate the prefix cache.
In this version the escalation therefore jumps straight to checkpoint:
suspension is now so cheap (≈tens of ms) that suspending costs less
than compressing.

Prerequisite: a running llama-server (started by start_warm_start_kv_pro.command).
The rate limit is simulated (none exists locally): the budget starts low and
shrinks each step, so the checkpoint triggers mid-task.
"""

import json
import os
import sys
import time
import uuid

try:
    import requests
except ImportError:
    print("'requests' is missing. Run: pip3 install requests")
    sys.exit(1)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import AgentState, BaseLLMBackend, PredictiveScheduler

SERVER = "http://127.0.0.1:8080"
KV_FILE = "agent_kv_integrated.bin"
CHECKPOINT_FILE = "checkpoint_integrated.json"


# =============================================================================
# 1. llama-server BACKEND (official chat template + slot 0)
# =============================================================================

class LlamaServerBackend(BaseLLMBackend):
    name = "llama-server"

    def apply_template(self, messages) -> str:
        try:
            r = requests.post(f"{SERVER}/apply-template", json={"messages": messages}, timeout=30)
            if r.status_code == 200:
                return r.json()["prompt"]
        except Exception:
            pass
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + "<|im_start|>assistant\n"

    def generate(self, messages, max_tokens=400, temperature=0.6):
        prompt = self.apply_template(messages)
        r = requests.post(f"{SERVER}/completion", json={
            "prompt": prompt, "n_predict": max_tokens,
            "temperature": temperature, "top_p": 0.95,
            "cache_prompt": True, "id_slot": 0,
        }, timeout=1800)
        r.raise_for_status()
        d = r.json()
        t = d.get("timings", {})
        self.last_prompt_n = t.get("prompt_n", 0)      # recomputed prefill tokens
        self.last_prompt_ms = t.get("prompt_ms", 0)
        content = d.get("content", "")
        # strip the thinking from the history (only the final answer remains)
        if "</think>" in content:
            content = content.split("</think>")[-1]
        tokens = t.get("prompt_n", 0) + t.get("predicted_n", 0)
        return content.strip(), max(1, int(tokens))

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


# =============================================================================
# 2. SCHEDULER WITH TRANSACTIONAL KV-CHECKPOINT
# =============================================================================

class KVCheckpointScheduler(PredictiveScheduler):

    def _base_estimate(self, messages):
        # Estimate tuned for this workload (short answers, /no_think):
        # full input + ~260 tokens of expected generation.
        input_tokens = sum(self.backend.estimate_tokens(m["content"]) for m in messages)
        return input_tokens + 260

    def decide_and_apply(self, state, estimated, context_tokens):
        # No compress/summarize: they would invalidate the prefix KV-cache.
        # With true warm start, suspending costs less than compressing.
        return "checkpoint" if self.should_checkpoint(estimated) else "continue"

    # ---- KV lifecycle: SEMI-TEMPORARY memory -------------------------------
    # The KV blob is an accelerator, not the truth (that is the JSON manifest):
    # it can therefore be deleted aggressively. Rules:
    #   save     → new blob, delete the previous one (max 1 on disk)
    #   restore  → blob marked "consumed"
    #   1st successful step after resume → consumed blob deleted
    #   task completed / orphan blobs → full cleanup
    _consumed_kv: str = ""
    _current_kv: str = ""

    @staticmethod
    def _kv_path(name):
        return os.path.join("kv_cache", name)

    def _delete_kv(self, name: str, reason: str):
        try:
            if name and os.path.exists(self._kv_path(name)):
                mb = os.path.getsize(self._kv_path(name)) / 1e6
                os.remove(self._kv_path(name))
                print(f"  ♻ KV '{name}' deleted ({mb:.0f} MB freed — {reason})")
        except OSError:
            pass

    def gc_consumed_kv(self):
        """Call after the first successful step following a resume."""
        if self._consumed_kv:
            self._delete_kv(self._consumed_kv, "consumed after resume")
            self._consumed_kv = ""

    def gc_orphans(self):
        """On startup: delete blobs not referenced by the current manifest."""
        referenced = self._current_kv
        if os.path.isdir("kv_cache"):
            for f_ in os.listdir("kv_cache"):
                if f_.startswith("agent_kv_") and f_ != referenced:
                    self._delete_kv(f_, "orphan blob")

    # ---- transaction: KV first, then the logical state (commit) ----------
    def save_checkpoint_kv(self, state: AgentState) -> dict:
        t0 = time.time()
        cp_id = uuid.uuid4().hex
        new_kv = f"agent_kv_{cp_id[:8]}.bin"
        r = requests.post(f"{SERVER}/slots/0?action=save",
                          json={"filename": new_kv}, timeout=1800)
        r.raise_for_status()
        kv_info = r.json()                       # contains n_saved
        data = {
            "checkpoint_id": cp_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kv_file": new_kv,
            "kv_n_saved": kv_info.get("n_saved", 0),
            "state": {
                "step": state.step,
                "messages": state.messages,
                "total_tokens_used": state.total_tokens_used,
                "task_description": state.task_description,
                "idempotency_keys": state.idempotency_keys,
            },
        }
        tmp = CHECKPOINT_FILE + ".tmp"           # atomic write:
        with open(tmp, "w") as f:                # if the KV save fails,
            json.dump(data, f, indent=2)         # the old checkpoint stays valid
        os.replace(tmp, CHECKPOINT_FILE)
        # only AFTER the manifest commit does the old blob become obsolete
        old = self._current_kv
        self._current_kv = new_kv
        if old and old != new_kv:
            self._delete_kv(old, "replaced by the new checkpoint")
        ms = (time.time() - t0) * 1000
        return {"ms": ms, "n_saved": kv_info.get("n_saved", 0)}

    def load_checkpoint_kv(self):
        if not os.path.exists(CHECKPOINT_FILE):
            return None, {}
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        info = {"kv_restored": False, "n_restored": 0}
        self._current_kv = data.get("kv_file", "")
        self.gc_orphans()
        try:
            r = requests.post(f"{SERVER}/slots/0?action=restore",
                              json={"filename": data["kv_file"]}, timeout=1800)
            r.raise_for_status()
            n = r.json().get("n_restored", 0)
            info = {"kv_restored": n == data.get("kv_n_saved", -1), "n_restored": n}
            self._consumed_kv = data["kv_file"]   # semi-temporary: marked consumed
            self._current_kv = ""
        except Exception as e:
            print(f"  (KV restore failed: {e} → continuing with logical warm start)")
        s = data["state"]
        state = AgentState(step=s["step"], messages=s["messages"],
                           total_tokens_used=s["total_tokens_used"],
                           task_description=s["task_description"],
                           idempotency_keys=s.get("idempotency_keys", []))
        return state, info


# =============================================================================
# 3. DEMO — multi-step task with a mid-task checkpoint and ~zero-cost resume
# =============================================================================

QUESTIONS = [
    "List the 3 main risks of an LLM agent without resource management.",
    "For each risk, give one metric to measure it.",
    "Which of the three risks is most urgent? Justify by comparing the metrics.",
    "Propose a mitigation policy for the most urgent risk.",
    "What role does the KV-cache play in resuming a suspended agent?",
    "Summarize the whole conversation in 3 bullet points.",
]


def main():
    print("=" * 78)
    print("  PREDICTIVE SCHEDULER + KV-CHECKPOINT — integrated system")
    print("=" * 78)

    try:
        requests.get(f"{SERVER}/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: llama-server is not running. Start it first (see docs).")
        sys.exit(1)

    backend = LlamaServerBackend()
    scheduler = KVCheckpointScheduler(
        backend=backend,
        initial_remaining_tokens=1200,   # SIMULATED low budget: forces a mid-task checkpoint
        safety_factor_k=2.0,
    )

    state, info = scheduler.load_checkpoint_kv()
    if state:
        scheduler.remaining_tokens = 1200           # window "has reset"
        print(f"↻ RESUMED from step {state.step + 1} | KV restored: "
              f"{'yes, ' + str(info['n_restored']) + ' cells' if info['kv_restored'] else 'NO (logical fallback)'}")
    else:
        state = AgentState(
            task_description="Risk analysis of an autonomous LLM agent",
            messages=[{"role": "system",
                       "content": "Answer concisely (max 100 words). /no_think"}],
        )

    while state.step < len(QUESTIONS):
        state.add_message("user", QUESTIONS[state.step])
        d = scheduler.execute_step(state)
        prefill = getattr(backend, "last_prompt_n", "?")
        print(f"Step {d['step']:2d} | budget(sim): {d['remaining']:5d} | "
              f"recomputed prefill: {prefill:>5} tok | {d['action'].upper()}")

        if d["action"] != "checkpoint":
            # first successful step after resume → the consumed KV can be deleted
            scheduler.gc_consumed_kv()

        if d["action"] == "checkpoint":
            state.messages.pop()   # the unanswered question will be re-added on resume
            info = scheduler.save_checkpoint_kv(state)
            print(f"\n🛑 TRANSACTIONAL CHECKPOINT: logical state + KV-cache "
                  f"({info['n_saved']} cells) saved in {info['ms']:.0f} ms")
            print("   Run this script again: it will resume from the exact step WITHOUT re-prefill.")
            return

    print("-" * 78)
    print(f"✅ Task completed in {state.step} steps | tokens used: {state.total_tokens_used}")
    # full cleanup: manifest + any leftover KV blobs
    scheduler._delete_kv(scheduler._current_kv, "task completed")
    scheduler._delete_kv(scheduler._consumed_kv, "task completed")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
