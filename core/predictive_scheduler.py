#!/usr/bin/env python3
"""
Predictive Resource-Aware Scheduler v4 — aligned with the research report v1.0
==============================================================================

This version faithfully implements what the report promises:

  §4  Mathematical formalization:
      - T_estimated = T_input + max_tokens + tool_overhead + ε
        where ε comes from MOVING AVERAGES of past estimation errors (not random noise)
      - σ = standard deviation of the observed real consumption
      - Checkpoint rule: T_remaining < T_estimated + k·σ
      - Risk(state) = w1·(T_est/T_rem) + w2·(C_ctx/C_max) + w3·(estimated_cost/remaining_budget)

  §5  Extended decision policy (escalation ladder, actually EXECUTED):
      1. compress     → compresses the oldest tool outputs
      2. summarize    → replaces old history with a summary
      3. model_switch → switches to a cheaper model
      4. checkpoint   → graceful save and suspension

  §3.1 Side-effect-aware serialization:
      - JSON checkpoint with checkpoint_id, idempotency keys and timestamp

  §8.2 Prototype metrics:
      - suspension precision, false positive rate, token waste avoided, recovery info

NOTE: the demo at the bottom of this file uses MockBackend — SIMULATED numbers
for illustration only. Real, reproducible results live in experiments/.
"""

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


# =============================================================================
# 1. AGENT STATE
# =============================================================================

@dataclass
class AgentState:
    step: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
    total_tokens_used: int = 0
    task_description: str = ""
    # §3.1 — idempotency keys: one per step with side effects
    idempotency_keys: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


# =============================================================================
# 2. LLM BACKENDS (unchanged from v3)
# =============================================================================

class BaseLLMBackend:
    name = "base"

    def generate(self, messages, max_tokens=600, temperature=0.6) -> Tuple[str, int]:
        raise NotImplementedError

    def estimate_tokens(self, text: str) -> int:
        raise NotImplementedError


class MockBackend(BaseLLMBackend):
    """Simulated backend (fast, for exercising the scheduler).

    Returns canned replies and RANDOM token counts — for demos and tests only.
    For real measurements see experiments/.
    """

    def __init__(self, avg_tokens: int = 320, name: str = "mock-standard"):
        self.avg_tokens = avg_tokens
        self.name = name

    def generate(self, messages, max_tokens=600, temperature=0.6):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        response = f"[{self.name}] Analyzed: {last[:60]}... Proceeding to the next step."
        tokens = int(self.avg_tokens * random.uniform(0.75, 1.35))
        return response, min(tokens, max_tokens)

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


class LiteLLMBackend(BaseLLMBackend):
    """Real backend via LiteLLM (Ollama, OpenAI, Anthropic, Groq, ...)."""

    def __init__(self, model: str = "ollama/llama3.2", api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is not installed. Run: pip install litellm")
        self.model = model
        self.name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    def generate(self, messages, max_tokens=600, temperature=0.6):
        try:
            response = litellm.completion(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else self.estimate_tokens(content)
            return content.strip(), tokens
        except Exception as e:
            print(f"[LiteLLMBackend] Error: {e}")
            return f"[LiteLLM error] {str(e)[:100]}", 50

    def estimate_tokens(self, text: str) -> int:
        try:
            return litellm.token_counter(model=self.model, text=text)
        except Exception:
            return max(10, len(text) // 3)


# =============================================================================
# 3. PREDICTIVE SCHEDULER v4 — faithful to §4 and §5 of the report
# =============================================================================

MAX_TOKENS_OUT = 650      # max_tokens of the next call
TOOL_OVERHEAD = 280       # estimated tool overhead


class PredictiveScheduler:
    def __init__(
        self,
        backend: BaseLLMBackend,
        economy_backend: Optional[BaseLLMBackend] = None,   # for the model switch (§5.3)
        initial_remaining_tokens: int = 7000,
        safety_factor_k: float = 2.0,
        context_limit: int = 128_000,
        budget_total_tokens: int = 50_000,                   # economic budget (§4, w3 term)
        weights: Tuple[float, float, float] = (0.55, 0.20, 0.25),  # w1, w2, w3
        seed: Optional[int] = None,
    ):
        self.backend = backend
        self.economy_backend = economy_backend
        self.switched = False

        self.remaining_tokens = initial_remaining_tokens
        self.safety_factor_k = safety_factor_k
        self.context_limit = context_limit
        self.budget_total = budget_total_tokens
        self.budget_used = 0
        self.w1, self.w2, self.w3 = weights

        # §4 — history for ε and σ
        self.historical_consumption: List[int] = []   # real consumption
        self.estimation_errors: List[int] = []        # error = real - base_estimate

        # §8.2 — metrics
        self.metrics = {
            "steps": 0, "continue": 0, "compress": 0, "summarize": 0,
            "model_switch": 0, "checkpoint": 0,
            "true_positive_checkpoints": 0,   # checkpoint that actually avoided a crash
            "false_positive_checkpoints": 0,  # unnecessary suspension
            "would_have_crashed": 0,          # steps where, without the scheduler → 429
            "tokens_saved_compress": 0,       # tokens saved by compression
            "tokens_saved_summarize": 0,      # tokens saved by summarization
        }

        if seed is not None:
            random.seed(seed)

    # ---------------- Telemetry (simulated: provider headers) ----------------

    def simulate_provider_headers(self):
        noise = random.randint(-100, 100)
        current = max(0, self.remaining_tokens + noise)
        return {"x-ratelimit-remaining-tokens": current}

    # ---------------- §4: estimation with moving-average ε ----------------

    def _base_estimate(self, messages: List[Dict]) -> int:
        input_tokens = sum(self.backend.estimate_tokens(m["content"]) for m in messages)
        return input_tokens + MAX_TOKENS_OUT + TOOL_OVERHEAD

    def epsilon(self) -> int:
        """ε = moving average of the estimation errors over the last 6 steps (§4)."""
        if not self.estimation_errors:
            return 0
        window = self.estimation_errors[-6:]
        return int(sum(window) / len(window))

    def estimate_next_step_cost(self, messages: List[Dict]) -> int:
        return max(200, self._base_estimate(messages) + self.epsilon())

    def sigma(self, fallback_estimate: int) -> float:
        """σ = standard deviation of the observed real consumption (§4)."""
        h = self.historical_consumption
        if len(h) >= 4:
            mean = sum(h) / len(h)
            return (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
        return fallback_estimate * 0.28  # conservative until there is history

    # ---------------- §4: checkpoint rule and risk function ----------------

    def should_checkpoint(self, estimated_cost: int) -> bool:
        margin = self.safety_factor_k * self.sigma(estimated_cost)
        return self.remaining_tokens < estimated_cost + margin

    def compute_risk(self, estimated_cost: int, context_tokens: int) -> float:
        budget_remaining = max(1, self.budget_total - self.budget_used)
        return (
            self.w1 * (estimated_cost / max(self.remaining_tokens, 1))
            + self.w2 * (context_tokens / self.context_limit)
            + self.w3 * (estimated_cost / budget_remaining)
        )

    # ---------------- §5: actions of the extended policy (actually EXECUTED) ----------------

    def compress_tool_outputs(self, state: AgentState) -> int:
        """§5.1 — compresses the oldest assistant messages (except the last 3)."""
        saved = 0
        for m in state.messages[:-3]:
            if m["role"] == "assistant" and len(m["content"]) > 120:
                saved += self.backend.estimate_tokens(m["content"][120:])
                m["content"] = m["content"][:117] + "..."
        return saved

    def summarize_history(self, state: AgentState) -> int:
        """§5.2 — semantic garbage collection: replaces old history with a summary."""
        if len(state.messages) <= 4:
            return 0
        old = state.messages[:-3]
        before = sum(self.backend.estimate_tokens(m["content"]) for m in old)
        summary = (f"[SUMMARY of the first {len(old)} messages] Task: {state.task_description}. "
                   f"Completed {state.step} steps; intermediate results kept in the checkpoint.")
        state.messages = [{"role": "system", "content": summary}] + state.messages[-3:]
        return max(0, before - self.backend.estimate_tokens(summary))

    def switch_model(self) -> bool:
        """§5.3 — switches to the economy model, if available and not already done."""
        if self.economy_backend and not self.switched:
            self.backend = self.economy_backend
            self.switched = True
            return True
        return False

    def decide_and_apply(self, state: AgentState, estimated: int, context_tokens: int) -> str:
        """
        §5 — Escalation ladder: tries the actions in order and stops
        as soon as the safety rule (§4) is satisfied again.
        """
        if not self.should_checkpoint(estimated):
            return "continue"

        # 1) Tool-output compression
        self.metrics["tokens_saved_compress"] += self.compress_tool_outputs(state)
        estimated = self.estimate_next_step_cost(state.messages)
        if not self.should_checkpoint(estimated):
            return "compress"

        # 2) Selective summarization
        self.metrics["tokens_saved_summarize"] += self.summarize_history(state)
        estimated = self.estimate_next_step_cost(state.messages)
        if not self.should_checkpoint(estimated):
            return "summarize"

        # 3) Model switch
        if self.switch_model():
            estimated = self.estimate_next_step_cost(state.messages)
            if not self.should_checkpoint(estimated):
                return "model_switch"

        # 4) Graceful checkpoint
        return "checkpoint"

    # ---------------- Execution loop ----------------

    def execute_step(self, state: AgentState):
        headers = self.simulate_provider_headers()
        self.remaining_tokens = headers["x-ratelimit-remaining-tokens"]

        estimated = self.estimate_next_step_cost(state.messages)
        context_tokens = sum(self.backend.estimate_tokens(m["content"]) for m in state.messages)
        risk = self.compute_risk(estimated, context_tokens)

        action = self.decide_and_apply(state, estimated, context_tokens)
        self.metrics["steps"] += 1
        self.metrics[action] += 1

        if action == "checkpoint":
            # §8.2 — was the checkpoint necessary? (checked against the cost that would have occurred)
            base = self._base_estimate(state.messages)
            hypothetical = int(base * random.uniform(0.85, 1.15))
            if hypothetical > self.remaining_tokens:
                self.metrics["true_positive_checkpoints"] += 1
            else:
                self.metrics["false_positive_checkpoints"] += 1
            return {"step": state.step, "remaining": self.remaining_tokens,
                    "estimated": estimated, "actual": 0, "action": action,
                    "risk": round(risk, 3), "model": self.backend.name}

        # Model call (real or mock)
        response, actual = self.backend.generate(state.messages, max_tokens=MAX_TOKENS_OUT)

        # §8.2 — without the scheduler, would this step have caused a 429?
        if actual > self.remaining_tokens:
            self.metrics["would_have_crashed"] += 1

        # update history for ε and σ (§4)
        self.estimation_errors.append(actual - self._base_estimate(state.messages))
        self.historical_consumption.append(actual)

        self.remaining_tokens = max(0, self.remaining_tokens - actual)
        self.budget_used += actual
        state.total_tokens_used += actual
        state.step += 1
        state.add_message("assistant", response[:350])
        # every step records its idempotency key (§3.1)
        state.idempotency_keys.append(f"step-{state.step}-{uuid.uuid4().hex[:8]}")

        return {"step": state.step, "remaining": self.remaining_tokens,
                "estimated": estimated, "actual": actual, "action": action,
                "risk": round(risk, 3), "model": self.backend.name}

    # ---------------- §3.1: checkpoint with idempotency ----------------

    def save_checkpoint(self, state: AgentState, filename="checkpoint_v4.json"):
        data = {
            "checkpoint_id": uuid.uuid4().hex,          # unique identity of the checkpoint
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": asdict(state),                      # includes idempotency_keys
            "scheduler": {
                "remaining_tokens": self.remaining_tokens,
                "budget_used": self.budget_used,
                "historical_consumption": self.historical_consumption[-10:],
                "estimation_errors": self.estimation_errors[-10:],
                "switched": self.switched,
            },
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename

    def load_checkpoint(self, filename="checkpoint_v4.json") -> Optional[AgentState]:
        try:
            with open(filename) as f:
                data = json.load(f)
            state = AgentState(**data["state"])
            s = data["scheduler"]
            self.remaining_tokens = s["remaining_tokens"]
            self.budget_used = s.get("budget_used", 0)
            self.historical_consumption = s.get("historical_consumption", [])
            self.estimation_errors = s.get("estimation_errors", [])
            self.switched = s.get("switched", False)
            return state
        except Exception:
            return None

    # ---------------- §8.2: metrics summary ----------------

    def metrics_summary(self) -> Dict:
        m = dict(self.metrics)
        cp = m["true_positive_checkpoints"] + m["false_positive_checkpoints"]
        m["suspension_precision"] = round(m["true_positive_checkpoints"] / cp, 2) if cp else None
        m["false_positive_rate"] = round(m["false_positive_checkpoints"] / cp, 2) if cp else None
        return m


# =============================================================================
# 4. DEMO — simulated backend, illustrative only (real results in experiments/)
# =============================================================================

def run_v4(backend_type: str = "mock", model: str = "ollama/llama3.2",
           resume: bool = True, seed: Optional[int] = 42):
    print("=" * 78)
    print("  PREDICTIVE SCHEDULER v4 — aligned with the research report v1.0")
    print("  NOTE: demo uses a SIMULATED backend (random numbers), for illustration")
    print("        only. Real, reproducible results live in experiments/.")
    print("=" * 78)

    if backend_type == "litellm" and LITELLM_AVAILABLE:
        backend = LiteLLMBackend(model=model)
        economy = None  # with LiteLLM you could pass e.g. LiteLLMBackend("ollama/qwen2.5:0.5b")
    else:
        backend = MockBackend(avg_tokens=320, name="mock-standard")
        economy = MockBackend(avg_tokens=140, name="mock-economy")  # "cheap" model

    scheduler = PredictiveScheduler(
        backend=backend,
        economy_backend=economy,
        initial_remaining_tokens=6500,
        safety_factor_k=2.0,
        budget_total_tokens=50_000,
        seed=seed,
    )

    state = (scheduler.load_checkpoint() if resume else None) or AgentState(
        task_description="Analysis and refactoring of a complex distributed system",
        messages=[{"role": "user", "content": "Analyze and refactor the distributed system XYZ."}],
    )
    if state.step > 0:
        # Warm start: in reality you resume once the rate-limit window
        # has reset → the provider reports a full budget again.
        scheduler.remaining_tokens = 6500
        print(f"↻ WARM START from checkpoint (resuming from step {state.step + 1}, window reset)")

    print(f"Task: {state.task_description}")
    print("-" * 78)

    icons = {"continue": "✓", "compress": "⚡", "summarize": "📝",
             "model_switch": "🔀", "checkpoint": "🛑"}

    for _ in range(25):
        d = scheduler.execute_step(state)
        print(f"Step {d['step']:2d} | Rem: {d['remaining']:5d} | Est: {d['estimated']:4d} | "
              f"Real: {d['actual']:4d} | Risk: {d['risk']:.2f} | "
              f"{icons[d['action']]} {d['action'].upper():12s} | {d['model']}")
        if d["action"] == "checkpoint":
            fname = scheduler.save_checkpoint(state)
            print(f"          → Checkpoint saved: {fname} (with idempotency keys)")
            break
        time.sleep(0.05)

    print("-" * 78)
    print(f"Steps: {state.step} | Tokens used: {state.total_tokens_used}")
    print("\nMetrics (§8.2):")
    for k, v in scheduler.metrics_summary().items():
        print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    run_v4(
        backend_type="mock",          # switch to "litellm" when you want real calls
        model="ollama/llama3.2",
        resume=True,                  # resumes from the checkpoint if present (warm start)
    )
