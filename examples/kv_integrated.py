#!/usr/bin/env python3
"""
SCHEDULER + KV-CHECKPOINT INTEGRATO — il sistema completo
==========================================================

Fin qui avevamo due pezzi dimostrati separatamente:
  • lo scheduler predittivo che DECIDE quando fermarsi (v4, Groq)
  • il warm start vero via KV-cache su disco (warm_start_kv)

Questo file li unisce: quando lo scheduler decide "checkpoint", salva in
un'UNICA TRANSAZIONE sia lo stato logico (messaggi, passi, idempotenza)
sia la KV-cache del modello. Alla ripresa ripristina entrambi: l'agente
riparte dal passo esatto E il modello non ricalcola il prefill.

    decisione (§4) ──► checkpoint ──► [ KV-cache su disco + stato JSON ]
                                            │
    rilancio ──► restore KV + stato ──► riprende a costo ~zero

Nota di design emersa: con il KV-checkpoint, le azioni che MODIFICANO la
storia (compress/summarize, §5.1–5.2) invalidano la cache del prefisso.
In questa versione l'escalation salta quindi direttamente al checkpoint:
il costo della sospensione è ormai così basso (≈decine di ms) che
sospendere è più economico che comprimere.

Prerequisito: llama-server attivo (avviato da avvia_warm_start_kv_pro.command).
Il rate limit è simulato (in locale non esiste): il budget parte basso e
cala a ogni passo, per far scattare il checkpoint a metà task.
"""

import json
import os
import sys
import time
import uuid

try:
    import requests
except ImportError:
    print("Manca 'requests'. Esegui: pip3 install requests")
    sys.exit(1)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import AgentState, BaseLLMBackend, PredictiveScheduler

SERVER = "http://127.0.0.1:8080"
KV_FILE = "agente_kv_integrato.bin"
CHECKPOINT_FILE = "checkpoint_integrato.json"


# =============================================================================
# 1. BACKEND llama-server (chat template ufficiale + slot 0)
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
        self.last_prompt_n = t.get("prompt_n", 0)      # token di prefill ricalcolati
        self.last_prompt_ms = t.get("prompt_ms", 0)
        content = d.get("content", "")
        # togli il thinking dalla storia (resta solo la risposta finale)
        if "</think>" in content:
            content = content.split("</think>")[-1]
        tokens = t.get("prompt_n", 0) + t.get("predicted_n", 0)
        return content.strip(), max(1, int(tokens))

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


# =============================================================================
# 2. SCHEDULER CON KV-CHECKPOINT TRANSAZIONALE
# =============================================================================

class KVCheckpointScheduler(PredictiveScheduler):

    def _base_estimate(self, messages):
        # Stima tarata su questo workload (risposte brevi, /no_think):
        # input completo + ~260 token di generazione attesa.
        input_tokens = sum(self.backend.estimate_tokens(m["content"]) for m in messages)
        return input_tokens + 260

    def decide_and_apply(self, state, estimated, context_tokens):
        # Niente compress/summarize: invaliderebbero la KV-cache del prefisso.
        # Con il warm start vero, sospendere costa meno che comprimere.
        return "checkpoint" if self.should_checkpoint(estimated) else "continue"

    # ---- ciclo di vita della KV: memoria SEMI-TEMPORANEA ------------------
    # Il blob KV è un acceleratore, non la verità (quella è il manifest JSON):
    # può quindi essere cancellato aggressivamente. Regole:
    #   save     → nuovo blob, cancella il precedente (max 1 su disco)
    #   restore  → blob marcato "consumato"
    #   1° passo riuscito post-ripresa → blob consumato cancellato
    #   task completato / blob orfani → pulizia totale
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
                print(f"  ♻ KV '{name}' cancellata ({mb:.0f} MB liberati — {reason})")
        except OSError:
            pass

    def gc_consumed_kv(self):
        """Da chiamare dopo il primo passo riuscito post-ripresa."""
        if self._consumed_kv:
            self._delete_kv(self._consumed_kv, "consumata dopo la ripresa")
            self._consumed_kv = ""

    def gc_orphans(self):
        """All'avvio: elimina blob non referenziati dal manifest corrente."""
        referenced = self._current_kv
        if os.path.isdir("kv_cache"):
            for f_ in os.listdir("kv_cache"):
                if f_.startswith("agente_kv_") and f_ != referenced:
                    self._delete_kv(f_, "blob orfano")

    # ---- transazione: prima la KV, poi lo stato logico (commit) ----------
    def save_checkpoint_kv(self, state: AgentState) -> dict:
        t0 = time.time()
        cp_id = uuid.uuid4().hex
        new_kv = f"agente_kv_{cp_id[:8]}.bin"
        r = requests.post(f"{SERVER}/slots/0?action=save",
                          json={"filename": new_kv}, timeout=1800)
        r.raise_for_status()
        kv_info = r.json()                       # contiene n_saved
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
        tmp = CHECKPOINT_FILE + ".tmp"           # scrittura atomica:
        with open(tmp, "w") as f:                # se il salvataggio KV fallisce,
            json.dump(data, f, indent=2)         # il vecchio checkpoint resta valido
        os.replace(tmp, CHECKPOINT_FILE)
        # solo DOPO il commit del manifest, il vecchio blob diventa obsoleto
        old = self._current_kv
        self._current_kv = new_kv
        if old and old != new_kv:
            self._delete_kv(old, "sostituita dal nuovo checkpoint")
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
            self._consumed_kv = data["kv_file"]   # semi-temporanea: marcata consumata
            self._current_kv = ""
        except Exception as e:
            print(f"  (restore KV fallito: {e} → proseguo con warm start logico)")
        s = data["state"]
        state = AgentState(step=s["step"], messages=s["messages"],
                           total_tokens_used=s["total_tokens_used"],
                           task_description=s["task_description"],
                           idempotency_keys=s.get("idempotency_keys", []))
        return state, info


# =============================================================================
# 3. DEMO — task multi-passo con checkpoint a metà e ripresa a costo ~zero
# =============================================================================

QUESTIONS = [
    "Elenca i 3 rischi principali di un agente LLM senza gestione delle risorse.",
    "Per ciascun rischio, indica una metrica per misurarlo.",
    "Qual è il rischio più urgente dei tre? Motiva confrontando le metriche.",
    "Proponi una politica di mitigazione per il rischio più urgente.",
    "Che ruolo ha la KV-cache nella ripresa di un agente sospeso?",
    "Riassumi tutta la conversazione in 3 punti.",
]


def main():
    print("=" * 78)
    print("  SCHEDULER PREDITTIVO + KV-CHECKPOINT — sistema integrato")
    print("=" * 78)

    try:
        requests.get(f"{SERVER}/health", timeout=5).raise_for_status()
    except Exception:
        print("ERRORE: llama-server non attivo. Avvialo con avvia_warm_start_kv_pro.command")
        sys.exit(1)

    backend = LlamaServerBackend()
    scheduler = KVCheckpointScheduler(
        backend=backend,
        initial_remaining_tokens=1200,   # budget SIMULATO basso: forza il checkpoint a metà task
        safety_factor_k=2.0,
    )

    state, info = scheduler.load_checkpoint_kv()
    if state:
        scheduler.remaining_tokens = 1200           # finestra "resettata"
        print(f"↻ RIPRESA dal passo {state.step + 1} | KV ripristinata: "
              f"{'sì, ' + str(info['n_restored']) + ' celle' if info['kv_restored'] else 'NO (fallback logico)'}")
    else:
        state = AgentState(
            task_description="Analisi dei rischi di un agente LLM autonomo",
            messages=[{"role": "system",
                       "content": "Rispondi in italiano, conciso (max 100 parole). /no_think"}],
        )

    while state.step < len(QUESTIONS):
        state.add_message("user", QUESTIONS[state.step])
        d = scheduler.execute_step(state)
        prefill = getattr(backend, "last_prompt_n", "?")
        print(f"Passo {d['step']:2d} | budget(sim): {d['remaining']:5d} | "
              f"prefill ricalcolato: {prefill:>5} tok | {d['action'].upper()}")

        if d["action"] != "checkpoint":
            # primo passo riuscito dopo la ripresa → la KV consumata si può cancellare
            scheduler.gc_consumed_kv()

        if d["action"] == "checkpoint":
            state.messages.pop()   # la domanda non risposta verrà ri-aggiunta alla ripresa
            info = scheduler.save_checkpoint_kv(state)
            print(f"\n🛑 CHECKPOINT TRANSAZIONALE: stato logico + KV-cache "
                  f"({info['n_saved']} celle) salvati in {info['ms']:.0f} ms")
            print("   Rilancia questo script: riprenderà dal passo esatto SENZA re-prefill.")
            return

    print("-" * 78)
    print(f"✅ Task completato in {state.step} passi | token usati: {state.total_tokens_used}")
    # pulizia totale: manifest + eventuali blob KV residui
    scheduler._delete_kv(scheduler._current_kv, "task completato")
    scheduler._delete_kv(scheduler._consumed_kv, "task completato")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
