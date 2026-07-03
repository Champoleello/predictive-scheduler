#!/usr/bin/env python3
"""
Scheduler Predittivo con TELEMETRIA REALE — Groq (gratis)
==========================================================

Fin qui gli header del provider erano simulati. Questa versione legge gli
header VERI restituiti da Groq a ogni risposta:

    x-ratelimit-remaining-tokens   → token rimasti in questo minuto (TPM)
    x-ratelimit-limit-tokens       → il tuo limite TPM (6.000 sul piano free)
    x-ratelimit-reset-tokens       → tra quanto si resetta (es. "7.66s")

Il piano gratuito di Groq (6.000 token/minuto) è perfetto: il limite è così
basso che lo scheduler entra in azione davvero, senza spendere nulla.

COME PROVARLO (5 minuti):
  1. Vai su https://console.groq.com e registrati (gratis, basta email)
  2. Menu "API Keys" → "Create API Key" → copia la chiave (inizia con gsk_)
  3. Nel Terminale:
         export GROQ_API_KEY="gsk_..."
         cd ~/Desktop/test
         pip3 install requests
         python3 scheduler_groq_reale.py
"""

import json
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    print("Manca la libreria 'requests'. Esegui: pip3 install requests")
    sys.exit(1)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import (
    AgentState, BaseLLMBackend, PredictiveScheduler,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CHECKPOINT_FILE = "checkpoint_groq.json"


# =============================================================================
# 1. BACKEND GROQ — chiamate vere, header veri
# =============================================================================

def parse_reset(value: str) -> float:
    """Converte '7.66s' o '2m59.56s' in secondi."""
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
            print("ERRORE: variabile GROQ_API_KEY non impostata (vedi istruzioni sopra).")
            sys.exit(1)
        # ultimi header di telemetria ricevuti dal provider
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
        # --- TELEMETRIA REALE: gli header di cui parla il rapporto (§3.1) ---
        self.last_headers = {
            "remaining_tokens": resp.headers.get("x-ratelimit-remaining-tokens"),
            "limit_tokens": resp.headers.get("x-ratelimit-limit-tokens"),
            "reset_tokens": resp.headers.get("x-ratelimit-reset-tokens"),
            "remaining_requests": resp.headers.get("x-ratelimit-remaining-requests"),
        }

        if resp.status_code == 429:
            # Non dovrebbe succedere: lo scheduler esiste per evitarlo.
            self.got_429 = True
            retry = resp.headers.get("retry-after", "?")
            return f"[ERRORE 429 — rate limit! retry-after: {retry}s]", 0

        if resp.status_code in (400, 404) and "model" in resp.text.lower():
            print(f"\nERRORE: il modello '{self.model}' non è disponibile.")
            try:
                models = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}, timeout=15,
                ).json()
                print("Modelli disponibili sul tuo account:")
                for m in sorted(x["id"] for x in models.get("data", [])):
                    print(f"  - {m}")
                print(f"\nRilancia con: python3 scheduler_groq_reale.py NOME_MODELLO")
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
# 2. SCHEDULER CON TELEMETRIA REALE
#    (identico al v4, ma legge gli header veri invece di simularli)
# =============================================================================

class RealTelemetryScheduler(PredictiveScheduler):

    def simulate_provider_headers(self):
        """Override: niente simulazione — usa gli ultimi header VERI di Groq."""
        h = self.backend.last_headers
        if h.get("remaining_tokens") is not None:
            return {"x-ratelimit-remaining-tokens": int(float(h["remaining_tokens"]))}
        # Prima chiamata: nessun header ancora ricevuto → usa il valore corrente
        return {"x-ratelimit-remaining-tokens": self.remaining_tokens}

    def seconds_to_reset(self) -> float:
        return parse_reset(self.backend.last_headers.get("reset_tokens", ""))


# =============================================================================
# 3. DEMO — task multi-passo con rate limit VERO
# =============================================================================

TASK_STEPS = [
    "Spiega in 3 frasi cos'è il checkpointing proattivo per agenti LLM.",
    "Elenca 3 vantaggi del warm start rispetto al cold start.",
    "Descrivi brevemente cos'è la KV-cache di un transformer.",
    "Spiega cosa sono gli header di rate limiting di un provider LLM.",
    "Riassumi in una frase perché serve un fattore di sicurezza k nella stima.",
    "Descrivi la differenza tra compressione e summarization del contesto.",
    "Spiega cos'è una chiave di idempotenza e perché serve nei checkpoint.",
    "Elenca 3 metriche per valutare un sistema di sospensione predittiva.",
    "Spiega il concetto di context rot in una frase.",
    "Concludi con un riepilogo di 2 frasi su tutto quanto discusso.",
]


def main(model: str = "llama-3.1-8b-instant"):
    print("=" * 78)
    print("  PREDICTIVE SCHEDULER + GROQ — telemetria di rate limit REALE")
    print("=" * 78)

    backend = GroqBackend(model=model)
    scheduler = RealTelemetryScheduler(
        backend=backend,
        initial_remaining_tokens=6000,   # TPM del piano free (verrà sovrascritto dagli header veri)
        safety_factor_k=2.0,
    )

    state = scheduler.load_checkpoint(CHECKPOINT_FILE)
    if state:
        print(f"↻ WARM START: riparto dal passo {state.step + 1}")
        # Ping di telemetria: richiesta minuscola solo per leggere gli header
        # FRESCHI del provider (il valore salvato nel checkpoint è vecchio).
        backend.generate([{"role": "user", "content": "ping"}], max_tokens=1)
        h = backend.last_headers
        if h.get("remaining_tokens"):
            scheduler.remaining_tokens = int(float(h["remaining_tokens"]))
        print(f"   Telemetria aggiornata: {scheduler.remaining_tokens} token disponibili adesso")
    else:
        state = AgentState(
            task_description="Mini-corso a 10 passi sul checkpointing per agenti LLM",
            messages=[{"role": "system",
                       "content": "Rispondi in italiano, in modo conciso (max 120 parole)."}],
        )

    icons = {"continue": "✓", "compress": "⚡", "summarize": "📝",
             "model_switch": "🔀", "checkpoint": "🛑"}

    while state.step < len(TASK_STEPS):
        state.add_message("user", TASK_STEPS[state.step])
        d = scheduler.execute_step(state)

        reset = scheduler.seconds_to_reset()
        print(f"Step {d['step']:2d} | Rem(REALE): {d['remaining']:5d} | Est: {d['estimated']:4d} | "
              f"Real: {d['actual']:4d} | Risk: {d['risk']:.2f} | reset in {reset:5.1f}s | "
              f"{icons[d['action']]} {d['action'].upper()}")

        if backend.got_429:
            print("\n❌ Il provider ha risposto 429: la stima non è stata abbastanza prudente.")
            print("   Prova ad alzare safety_factor_k (es. 3.0) e rilancia.")
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            return

        if d["action"] == "checkpoint":
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            print(f"\n🛑 GRACEFUL CHECKPOINT → {CHECKPOINT_FILE}")
            print(f"   La finestra TPM si resetta tra ~{reset:.0f} secondi.")
            print(f"   Rilancia lo script dopo il reset per riprendere (Warm Start).")
            return

        # piccola pausa di cortesia per non colpire il limite di richieste/minuto
        time.sleep(2.5)

    print("-" * 78)
    print(f"✅ Task completato! Passi: {state.step} | Token usati: {state.total_tokens_used}")
    print(f"   Errori 429 subiti: 0 (lo scheduler ha fatto il suo lavoro)")
    print("\nMetriche (§8.2):")
    for k, v in scheduler.metrics_summary().items():
        print(f"  {k}: {v}")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else "llama-3.1-8b-instant")
