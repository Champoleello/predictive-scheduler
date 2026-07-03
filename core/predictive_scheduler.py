#!/usr/bin/env python3
"""
Predictive Resource-Aware Scheduler v4 — Allineato al Rapporto di Ricerca v1.0
==============================================================================

Questa versione implementa FEDELMENTE ciò che il rapporto promette:

  §4  Formalizzazione matematica:
      - T_estimated = T_input + max_tokens + tool_overhead + ε
        dove ε deriva da MEDIE MOBILI degli errori di stima passati (non rumore casuale)
      - σ = deviazione standard dei consumi reali osservati
      - Regola checkpoint: T_remaining < T_estimated + k·σ
      - Risk(state) = w1·(T_est/T_rem) + w2·(C_ctx/C_max) + w3·(costo_stimato/budget_residuo)

  §5  Politica decisionale estesa (scala di escalation, ESEGUITA davvero):
      1. compress     → comprime gli output dei tool più vecchi
      2. summarize    → sostituisce la storia vecchia con un riassunto
      3. model_switch → passa a un modello più economico
      4. checkpoint   → salvataggio graceful e sospensione

  §3.1 Serializzazione sensibile agli effetti collaterali:
      - checkpoint JSON con checkpoint_id, chiavi di idempotenza e timestamp

  §8.2 Metriche del prototipo:
      - suspension precision, false positive rate, token waste avoided, recovery info
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
# 1. STATO DELL'AGENTE
# =============================================================================

@dataclass
class AgentState:
    step: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
    total_tokens_used: int = 0
    task_description: str = ""
    # §3.1 — chiavi di idempotenza: una per ogni passo con effetti collaterali
    idempotency_keys: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


# =============================================================================
# 2. LLM BACKENDS (invariati dalla v3)
# =============================================================================

class BaseLLMBackend:
    name = "base"

    def generate(self, messages, max_tokens=600, temperature=0.6) -> Tuple[str, int]:
        raise NotImplementedError

    def estimate_tokens(self, text: str) -> int:
        raise NotImplementedError


class MockBackend(BaseLLMBackend):
    """Backend simulato (veloce, per testare lo scheduler)."""

    def __init__(self, avg_tokens: int = 320, name: str = "mock-standard"):
        self.avg_tokens = avg_tokens
        self.name = name

    def generate(self, messages, max_tokens=600, temperature=0.6):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        response = f"[{self.name}] Analizzato: {last[:60]}... Procedo col passo successivo."
        tokens = int(self.avg_tokens * random.uniform(0.75, 1.35))
        return response, min(tokens, max_tokens)

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


class LiteLLMBackend(BaseLLMBackend):
    """Backend reale via LiteLLM (Ollama, OpenAI, Anthropic, Groq, ...)."""

    def __init__(self, model: str = "ollama/llama3.2", api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM non è installato. Esegui: pip install litellm")
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
            print(f"[LiteLLMBackend] Errore: {e}")
            return f"[Errore LiteLLM] {str(e)[:100]}", 50

    def estimate_tokens(self, text: str) -> int:
        try:
            return litellm.token_counter(model=self.model, text=text)
        except Exception:
            return max(10, len(text) // 3)


# =============================================================================
# 3. PREDICTIVE SCHEDULER v4 — fedele a §4 e §5 del rapporto
# =============================================================================

MAX_TOKENS_OUT = 650      # max_tokens della prossima chiamata
TOOL_OVERHEAD = 280       # overhead stimato dei tool


class PredictiveScheduler:
    def __init__(
        self,
        backend: BaseLLMBackend,
        economy_backend: Optional[BaseLLMBackend] = None,   # per il model-switch (§5.3)
        initial_remaining_tokens: int = 7000,
        safety_factor_k: float = 2.0,
        context_limit: int = 128_000,
        budget_total_tokens: int = 50_000,                   # budget economico (§4, termine w3)
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

        # §4 — storici per ε e σ
        self.historical_consumption: List[int] = []   # consumi reali
        self.estimation_errors: List[int] = []        # errore = reale - stima_base

        # §8.2 — metriche
        self.metrics = {
            "steps": 0, "continue": 0, "compress": 0, "summarize": 0,
            "model_switch": 0, "checkpoint": 0,
            "true_positive_checkpoints": 0,   # checkpoint che ha davvero evitato un crash
            "false_positive_checkpoints": 0,  # sospensione inutile
            "would_have_crashed": 0,          # passi in cui senza scheduler → 429
            "tokens_saved_compress": 0,       # token risparmiati dalla compressione
            "tokens_saved_summarize": 0,      # token risparmiati dalla summarization
        }

        if seed is not None:
            random.seed(seed)

    # ---------------- Telemetria (simulata: header del provider) ----------------

    def simulate_provider_headers(self):
        noise = random.randint(-100, 100)
        current = max(0, self.remaining_tokens + noise)
        return {"x-ratelimit-remaining-tokens": current}

    # ---------------- §4: stima con ε da medie mobili ----------------

    def _base_estimate(self, messages: List[Dict]) -> int:
        input_tokens = sum(self.backend.estimate_tokens(m["content"]) for m in messages)
        return input_tokens + MAX_TOKENS_OUT + TOOL_OVERHEAD

    def epsilon(self) -> int:
        """ε = media mobile degli errori di stima degli ultimi 6 passi (§4)."""
        if not self.estimation_errors:
            return 0
        window = self.estimation_errors[-6:]
        return int(sum(window) / len(window))

    def estimate_next_step_cost(self, messages: List[Dict]) -> int:
        return max(200, self._base_estimate(messages) + self.epsilon())

    def sigma(self, fallback_estimate: int) -> float:
        """σ = deviazione standard dei consumi reali osservati (§4)."""
        h = self.historical_consumption
        if len(h) >= 4:
            mean = sum(h) / len(h)
            return (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
        return fallback_estimate * 0.28  # prudente finché non c'è storia

    # ---------------- §4: regola di checkpoint e funzione di rischio ----------------

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

    # ---------------- §5: azioni della politica estesa (ESEGUITE davvero) ----------------

    def compress_tool_outputs(self, state: AgentState) -> int:
        """§5.1 — comprime i messaggi assistant più vecchi (tranne gli ultimi 3)."""
        saved = 0
        for m in state.messages[:-3]:
            if m["role"] == "assistant" and len(m["content"]) > 120:
                saved += self.backend.estimate_tokens(m["content"][120:])
                m["content"] = m["content"][:117] + "..."
        return saved

    def summarize_history(self, state: AgentState) -> int:
        """§5.2 — garbage collection semantica: sostituisce la storia vecchia con un riassunto."""
        if len(state.messages) <= 4:
            return 0
        old = state.messages[:-3]
        before = sum(self.backend.estimate_tokens(m["content"]) for m in old)
        summary = (f"[RIASSUNTO dei primi {len(old)} messaggi] Task: {state.task_description}. "
                   f"Completati {state.step} passi; risultati intermedi conservati nel checkpoint.")
        state.messages = [{"role": "system", "content": summary}] + state.messages[-3:]
        return max(0, before - self.backend.estimate_tokens(summary))

    def switch_model(self) -> bool:
        """§5.3 — passa al modello economico, se disponibile e non già fatto."""
        if self.economy_backend and not self.switched:
            self.backend = self.economy_backend
            self.switched = True
            return True
        return False

    def decide_and_apply(self, state: AgentState, estimated: int, context_tokens: int) -> str:
        """
        §5 — Scala di escalation: prova le azioni in ordine e si ferma
        appena la regola di sicurezza (§4) è di nuovo soddisfatta.
        """
        if not self.should_checkpoint(estimated):
            return "continue"

        # 1) Compressione output tool
        self.metrics["tokens_saved_compress"] += self.compress_tool_outputs(state)
        estimated = self.estimate_next_step_cost(state.messages)
        if not self.should_checkpoint(estimated):
            return "compress"

        # 2) Summarization selettiva
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

    # ---------------- Ciclo di esecuzione ----------------

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
            # §8.2 — il checkpoint era necessario? (verifica col costo che si sarebbe realizzato)
            base = self._base_estimate(state.messages)
            hypothetical = int(base * random.uniform(0.85, 1.15))
            if hypothetical > self.remaining_tokens:
                self.metrics["true_positive_checkpoints"] += 1
            else:
                self.metrics["false_positive_checkpoints"] += 1
            return {"step": state.step, "remaining": self.remaining_tokens,
                    "estimated": estimated, "actual": 0, "action": action,
                    "risk": round(risk, 3), "model": self.backend.name}

        # Chiamata al modello (reale o mock)
        response, actual = self.backend.generate(state.messages, max_tokens=MAX_TOKENS_OUT)

        # §8.2 — senza scheduler, questo passo avrebbe causato un 429?
        if actual > self.remaining_tokens:
            self.metrics["would_have_crashed"] += 1

        # aggiorna storici per ε e σ (§4)
        self.estimation_errors.append(actual - self._base_estimate(state.messages))
        self.historical_consumption.append(actual)

        self.remaining_tokens = max(0, self.remaining_tokens - actual)
        self.budget_used += actual
        state.total_tokens_used += actual
        state.step += 1
        state.add_message("assistant", response[:350])
        # ogni passo registra la sua chiave di idempotenza (§3.1)
        state.idempotency_keys.append(f"step-{state.step}-{uuid.uuid4().hex[:8]}")

        return {"step": state.step, "remaining": self.remaining_tokens,
                "estimated": estimated, "actual": actual, "action": action,
                "risk": round(risk, 3), "model": self.backend.name}

    # ---------------- §3.1: checkpoint con idempotenza ----------------

    def save_checkpoint(self, state: AgentState, filename="checkpoint_v4.json"):
        data = {
            "checkpoint_id": uuid.uuid4().hex,          # identità univoca del checkpoint
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": asdict(state),                      # include idempotency_keys
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

    # ---------------- §8.2: riepilogo metriche ----------------

    def metrics_summary(self) -> Dict:
        m = dict(self.metrics)
        cp = m["true_positive_checkpoints"] + m["false_positive_checkpoints"]
        m["suspension_precision"] = round(m["true_positive_checkpoints"] / cp, 2) if cp else None
        m["false_positive_rate"] = round(m["false_positive_checkpoints"] / cp, 2) if cp else None
        return m


# =============================================================================
# 4. DEMO
# =============================================================================

def run_v4(backend_type: str = "mock", model: str = "ollama/llama3.2",
           resume: bool = True, seed: Optional[int] = 42):
    print("=" * 78)
    print("  PREDICTIVE SCHEDULER v4 — allineato al Rapporto di Ricerca v1.0")
    print("=" * 78)

    if backend_type == "litellm" and LITELLM_AVAILABLE:
        backend = LiteLLMBackend(model=model)
        economy = None  # con LiteLLM potresti passare es. LiteLLMBackend("ollama/qwen2.5:0.5b")
    else:
        backend = MockBackend(avg_tokens=320, name="mock-standard")
        economy = MockBackend(avg_tokens=140, name="mock-economy")  # modello "economico"

    scheduler = PredictiveScheduler(
        backend=backend,
        economy_backend=economy,
        initial_remaining_tokens=6500,
        safety_factor_k=2.0,
        budget_total_tokens=50_000,
        seed=seed,
    )

    state = (scheduler.load_checkpoint() if resume else None) or AgentState(
        task_description="Analisi e refactoring di un sistema distribuito complesso",
        messages=[{"role": "user", "content": "Analizza e rifattorizza il sistema distribuito XYZ."}],
    )
    if state.step > 0:
        # Warm Start: nella realtà si riprende quando la finestra di rate limit
        # si è resettata → il provider restituisce di nuovo il budget pieno.
        scheduler.remaining_tokens = 6500
        print(f"↻ WARM START dal checkpoint (riparto dal passo {state.step + 1}, finestra resettata)")

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
            print(f"          → Checkpoint salvato: {fname} (con idempotency keys)")
            break
        time.sleep(0.05)

    print("-" * 78)
    print(f"Passi: {state.step} | Token usati: {state.total_tokens_used}")
    print("\nMetriche (§8.2):")
    for k, v in scheduler.metrics_summary().items():
        print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    run_v4(
        backend_type="mock",          # cambia in "litellm" quando vuoi
        model="ollama/llama3.2",
        resume=True,                  # riparte dal checkpoint se esiste (Warm Start)
    )
