#!/usr/bin/env python3
"""
Predictive Scheduler with REAL TELEMETRY — Groq (free tier)
==========================================================

Up to this point the provider headers were simulated. This version reads the
REAL headers Groq returns with every response:

    x-ratelimit-remaining-tokens   → tokens left in the current minute (TPM)
    x-ratelimit-limit-tokens       → your TPM limit (6,000 on the free plan)
    x-ratelimit-reset-tokens       → time until reset (e.g. "7.66s")

Groq's free plan (6,000 tokens/minute) is perfect here: the limit is so low
that the scheduler actually kicks in, at zero cost.

HOW TO TRY IT (5 minutes):
  1. Go to https://console.groq.com and sign up (free, email only)
  2. Menu "API Keys" → "Create API Key" → copy the key (starts with gsk_)
  3. In the terminal:
         export GROQ_API_KEY="gsk_..."
         pip3 install requests
         python3 groq_real_telemetry.py
"""

import json
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    print("The 'requests' library is missing. Run: pip3 install requests")
    sys.exit(1)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import (
    AgentState, BaseLLMBackend, PredictiveScheduler,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CHECKPOINT_FILE = "checkpoint_groq.json"


# =============================================================================
# 1. GROQ BACKEND — real calls, real headers
# =============================================================================

def parse_reset(value: str) -> float:
    """Converts '7.66s' or '2m59.56s' to seconds."""
    if not value:
        return 60.0
    m = re.match(r"(?:(\d+)m)?([\d.]+)s?", value)
    if not m:
        return 60.0
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return minutes * 60 + seconds


class GroqBackend(BaseLLMBackend):
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.name = f"groq/{model}"
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print(__doc__)
            print("ERROR: GROQ_API_KEY environment variable not set (see instructions above).")
            sys.exit(1)
        # latest telemetry headers received from the provider
        self.last_headers: dict = {}
        self.got_429 = False

    def generate(self, messages, max_tokens=600, temperature=0.6):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=60,
        )
        # --- REAL TELEMETRY: the headers the report talks about (§3.1) ---
        self.last_headers = {
            "remaining_tokens": resp.headers.get("x-ratelimit-remaining-tokens"),
            "limit_tokens": resp.headers.get("x-ratelimit-limit-tokens"),
            "reset_tokens": resp.headers.get("x-ratelimit-reset-tokens"),
            "remaining_requests": resp.headers.get("x-ratelimit-remaining-requests"),
        }

        if resp.status_code == 429:
            # Should not happen: the scheduler exists to prevent this.
            self.got_429 = True
            retry = resp.headers.get("retry-after", "?")
            return f"[ERROR 429 — rate limit! retry-after: {retry}s]", 0

        if resp.status_code in (400, 404) and "model" in resp.text.lower():
            print(f"\nERROR: model '{self.model}' is not available.")
            try:
                models = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}, timeout=15,
                ).json()
                print("Models available on your account:")
                for m in sorted(x["id"] for x in models.get("data", [])):
                    print(f"  - {m}")
                print(f"\nRe-run with: python3 groq_real_telemetry.py MODEL_NAME")
            except Exception:
                pass
            sys.exit(1)

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        tokens = data.get("usage", {}).get("total_tokens", self.estimate_tokens(content))
        return content.strip(), tokens

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


# =============================================================================
# 2. SCHEDULER WITH REAL TELEMETRY
#    (identical to v4, but reads the real headers instead of simulating them)
# =============================================================================

class RealTelemetryScheduler(PredictiveScheduler):

    def simulate_provider_headers(self):
        """Override: no simulation — uses Groq's latest REAL headers."""
        h = self.backend.last_headers
        if h.get("remaining_tokens") is not None:
            return {"x-ratelimit-remaining-tokens": int(float(h["remaining_tokens"]))}
        # First call: no headers received yet → use the current value
        return {"x-ratelimit-remaining-tokens": self.remaining_tokens}

    def seconds_to_reset(self) -> float:
        return parse_reset(self.backend.last_headers.get("reset_tokens", ""))


# =============================================================================
# 3. DEMO — multi-step task against a REAL rate limit
# =============================================================================

TASK_STEPS = [
    "Explain in 3 sentences what proactive checkpointing for LLM agents is.",
    "List 3 advantages of warm start over cold start.",
    "Briefly describe what a transformer's KV-cache is.",
    "Explain what an LLM provider's rate-limiting headers are.",
    "Summarize in one sentence why a safety factor k is needed in the estimate.",
    "Describe the difference between context compression and summarization.",
    "Explain what an idempotency key is and why checkpoints need it.",
    "List 3 metrics for evaluating a predictive suspension system.",
    "Explain the concept of context rot in one sentence.",
    "Conclude with a 2-sentence recap of everything discussed.",
]


def main(model: str = "llama-3.1-8b-instant"):
    print("=" * 78)
    print("  PREDICTIVE SCHEDULER + GROQ — REAL rate-limit telemetry")
    print("=" * 78)

    backend = GroqBackend(model=model)
    scheduler = RealTelemetryScheduler(
        backend=backend,
        initial_remaining_tokens=6000,   # free-plan TPM (will be overwritten by real headers)
        safety_factor_k=2.0,
    )

    state = scheduler.load_checkpoint(CHECKPOINT_FILE)
    if state:
        print(f"↻ WARM START: resuming from step {state.step + 1}")
        # Telemetry ping: a tiny request whose only purpose is reading the
        # provider's FRESH headers (the value saved in the checkpoint is stale).
        backend.generate([{"role": "user", "content": "ping"}], max_tokens=1)
        h = backend.last_headers
        if h.get("remaining_tokens"):
            scheduler.remaining_tokens = int(float(h["remaining_tokens"]))
        print(f"   Telemetry refreshed: {scheduler.remaining_tokens} tokens available right now")
    else:
        state = AgentState(
            task_description="10-step mini-course on checkpointing for LLM agents",
            messages=[{"role": "system",
                       "content": "Answer concisely (max 120 words)."}],
        )

    icons = {"continue": "✓", "compress": "⚡", "summarize": "📝",
             "model_switch": "🔀", "checkpoint": "🛑"}

    while state.step < len(TASK_STEPS):
        state.add_message("user", TASK_STEPS[state.step])
        d = scheduler.execute_step(state)

        reset = scheduler.seconds_to_reset()
        print(f"Step {d['step']:2d} | Rem(REAL): {d['remaining']:5d} | Est: {d['estimated']:4d} | "
              f"Real: {d['actual']:4d} | Risk: {d['risk']:.2f} | reset in {reset:5.1f}s | "
              f"{icons[d['action']]} {d['action'].upper()}")

        if backend.got_429:
            print("\n❌ The provider returned 429: the estimate was not conservative enough.")
            print("   Try raising safety_factor_k (e.g. 3.0) and re-run.")
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            return

        if d["action"] == "checkpoint":
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            print(f"\n🛑 GRACEFUL CHECKPOINT → {CHECKPOINT_FILE}")
            print(f"   The TPM window resets in ~{reset:.0f} seconds.")
            print(f"   Run the script again after the reset to resume (warm start).")
            return

        # small courtesy pause to avoid hitting the requests/minute limit
        time.sleep(2.5)

    print("-" * 78)
    print(f"✅ Task complete! Steps: {state.step} | Tokens used: {state.total_tokens_used}")
    print(f"   429 errors suffered: 0 (the scheduler did its job)")
    print("\nMetrics (§8.2):")
    for k, v in scheduler.metrics_summary().items():
        print(f"  {k}: {v}")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b-instant")
