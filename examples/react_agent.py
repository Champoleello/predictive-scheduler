#!/usr/bin/env python3
"""
ReAct Agent v5 — Agente con tool calling + Predictive Scheduler
================================================================

Il prototipo diventa un AGENTE VERO (§3.1 del rapporto):

    Agent (ReAct / tool-calling loop)
        ↓
    Resource-Aware Predictive Scheduler   ← decide continue/compress/.../checkpoint
        ↓
    LLM Provider (Mock, Ollama via LiteLLM, ...)

Il ciclo ReAct classico:
    Thought  → l'agente ragiona su cosa fare
    Action   → sceglie un tool (read_file, list_files, search, calculator, write_note)
    Observation → riceve il risultato del tool
    ... ripete finché non produce "Final Answer".

Lo scheduler si inserisce PRIMA di ogni chiamata LLM: se le risorse scarseggiano
prova compressione → summarization → model switch → e come ultima risorsa fa un
graceful checkpoint. Al riavvio l'agente riparte da dove si era fermato (Warm Start),
senza ri-eseguire i tool già eseguiti (chiavi di idempotenza, §3.1).

Uso:
    python3 react_agent_v5.py                  # demo con MockBackend
    (in fondo al file puoi passare a "litellm" + Ollama)
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
from predictive_scheduler import (
    AgentState, BaseLLMBackend, MockBackend, LiteLLMBackend,
    PredictiveScheduler, LITELLM_AVAILABLE,
)

CHECKPOINT_FILE = "checkpoint_react_v5.json"


# =============================================================================
# 1. TOOLS — semplici ma reali
# =============================================================================

def tool_list_files(arg: str) -> str:
    import os
    try:
        files = sorted(os.listdir(arg or "."))[:20]
        return "File nella cartella: " + ", ".join(files)
    except Exception as e:
        return f"Errore: {e}"


def tool_read_file(arg: str) -> str:
    try:
        with open(arg) as f:
            content = f.read(1500)
        return f"Contenuto di {arg} (primi 1500 caratteri):\n{content}"
    except Exception as e:
        return f"Errore: {e}"


def tool_search(arg: str) -> str:
    """Ricerca simulata (nessuna rete). Sostituibile con una vera API."""
    fake_db = {
        "checkpoint": "Il checkpointing proattivo salva lo stato prima dell'esaurimento risorse.",
        "rate limit": "I provider LLM espongono header come x-ratelimit-remaining-tokens.",
        "react": "ReAct alterna ragionamento (Thought) e azioni (Action) con tool.",
    }
    for key, val in fake_db.items():
        if key in arg.lower():
            return f"[search] {val}"
    return f"[search] Nessun risultato preciso per '{arg}'. Prova altri termini."


def tool_calculator(arg: str) -> str:
    try:
        if not re.fullmatch(r"[0-9+\-*/(). %]+", arg):
            return "Errore: espressione non valida (solo numeri e + - * / % parentesi)."
        return f"Risultato: {eval(arg)}"  # input già validato dalla regex
    except Exception as e:
        return f"Errore: {e}"


def tool_write_note(arg: str) -> str:
    with open("note_agente.txt", "a") as f:
        f.write(arg + "\n")
    return "Nota salvata in note_agente.txt"


TOOLS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "calculator": tool_calculator,
    "write_note": tool_write_note,
}

SYSTEM_PROMPT = """Sei un agente autonomo. Risolvi il task usando i tool disponibili.
Rispondi SEMPRE in questo formato (una tripletta per volta):

Thought: <il tuo ragionamento>
Action: <nome_tool>
Action Input: <argomento del tool>

Tool disponibili: list_files, read_file, search, calculator, write_note.

Quando hai finito, rispondi con:
Thought: <ragionamento finale>
Final Answer: <risposta completa al task>
"""


# =============================================================================
# 2. MOCK "SCRIPTED" — un finto LLM che parla in formato ReAct
#    (serve per testare l'agente senza modello vero)
# =============================================================================

class ScriptedReActBackend(BaseLLMBackend):
    name = "mock-react"

    SCRIPT = [
        ("Devo capire cosa contiene la cartella di lavoro.", "list_files", "."),
        ("Cerco informazioni sul checkpointing.", "search", "checkpoint proattivo"),
        ("Verifico come funziona il rate limit.", "search", "rate limit header"),
        ("Faccio un calcolo di esempio sul budget.", "calculator", "6500 - 320*12"),
        ("Salvo un appunto sui risultati.", "write_note", "Budget residuo stimato dopo 12 passi: 2660 token"),
        ("Approfondisco il pattern ReAct.", "search", "react pattern"),
        ("Ricontrollo i file generati.", "list_files", "."),
        ("Calcolo il margine di sicurezza con k=2 e sigma=90.", "calculator", "2*90"),
    ]

    def __init__(self, avg_tokens: int = 300, name: str = "mock-react"):
        self.avg_tokens = avg_tokens
        self.name = name
        self.call_count = 0

    def generate(self, messages, max_tokens=600, temperature=0.6):
        import random
        i = self.call_count
        self.call_count += 1
        tokens = int(self.avg_tokens * random.uniform(0.8, 1.3))
        if i < len(self.SCRIPT):
            thought, action, arg = self.SCRIPT[i]
            text = f"Thought: {thought}\nAction: {action}\nAction Input: {arg}"
        else:
            text = ("Thought: Ho raccolto abbastanza informazioni.\n"
                    "Final Answer: Analisi completata: cartella ispezionata, concetti chiave "
                    "verificati (checkpoint proattivo, rate limit header, pattern ReAct) e "
                    "calcoli di budget salvati in note_agente.txt.")
        return text, min(tokens, max_tokens)

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


# =============================================================================
# 3. PARSER ReAct
# =============================================================================

def parse_react(text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Ritorna (thought, action, action_input, final_answer)."""
    # Thought multilinea: si ferma ad Action / Final Answer (miglioria da review esterna)
    thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    final = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    action = re.search(r"Action:\s*(\w+)", text)
    arg = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text)
    return (
        thought.group(1).strip() if thought else "",
        action.group(1).strip() if action else None,
        arg.group(1).strip() if arg else None,
        final.group(1).strip() if final else None,
    )


# =============================================================================
# 4. CICLO DELL'AGENTE con scheduler integrato
# =============================================================================

def run_agent(task: str, backend_type: str = "mock", model: str = "ollama/llama3.2",
              max_react_steps: int = 30, initial_remaining_tokens: int = 6500):
    print("=" * 78)
    print("  ReAct AGENT v5 + Predictive Scheduler")
    print("=" * 78)

    if backend_type == "litellm" and LITELLM_AVAILABLE:
        backend = LiteLLMBackend(model=model)
        economy = None
    else:
        backend = ScriptedReActBackend(avg_tokens=300, name="mock-react")
        economy = ScriptedReActBackend(avg_tokens=130, name="mock-react-eco")
        economy.call_count = 99  # il modello economico va dritto alla risposta finale

    scheduler = PredictiveScheduler(
        backend=backend,
        economy_backend=economy,
        initial_remaining_tokens=initial_remaining_tokens,
        safety_factor_k=2.0,
    )

    # ---- Warm Start (§3.1): riprende dal checkpoint se esiste ----
    state = scheduler.load_checkpoint(CHECKPOINT_FILE)
    if state:
        scheduler.remaining_tokens = initial_remaining_tokens  # finestra resettata
        # tool già eseguiti: NON rifarli (le chiavi tool hanno formato "azione|input")
        executed_tools = {k for k in state.idempotency_keys if "|" in k}
        if isinstance(backend, ScriptedReActBackend):
            backend.call_count = state.step                    # riallinea lo script
        print(f"↻ WARM START: riparto dal passo {state.step + 1}, "
              f"{len(executed_tools)} azioni già eseguite (non verranno ripetute)")
    else:
        state = AgentState(
            task_description=task,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
        )
        executed_tools = set()

    print(f"Task: {task}")
    print("-" * 78)

    icons = {"continue": "✓", "compress": "⚡", "summarize": "📝",
             "model_switch": "🔀", "checkpoint": "🛑"}

    for _ in range(max_react_steps):
        # --- Lo scheduler decide e (se serve) agisce PRIMA della chiamata LLM ---
        d = scheduler.execute_step(state)
        print(f"[scheduler] step {d['step']:2d} | rem {d['remaining']:5d} | "
              f"risk {d['risk']:.2f} | {icons[d['action']]} {d['action']}")

        if d["action"] == "checkpoint":
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            print(f"\n🛑 GRACEFUL CHECKPOINT → {CHECKPOINT_FILE}")
            print("   Rilancia lo script per riprendere da qui (Warm Start).")
            return None

        # --- L'ultima risposta dell'assistente è la mossa ReAct ---
        llm_text = state.messages[-1]["content"]
        thought, action, arg, final = parse_react(llm_text)

        if final:
            print("-" * 78)
            print(f"✅ FINAL ANSWER: {final}")
            print(f"Passi: {state.step} | Token usati: {state.total_tokens_used}")
            print("\nMetriche scheduler (§8.2):")
            for k, v in scheduler.metrics_summary().items():
                print(f"  {k}: {v}")
            # task finito → il checkpoint non serve più
            import os
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
            return final

        if action and action in TOOLS:
            # chiave di idempotenza: stessa azione+input non viene rieseguita (§3.1)
            key = f"{action}|{arg}"
            if key in executed_tools:
                observation = "[skip] Azione già eseguita prima del checkpoint (idempotenza)."
            else:
                observation = TOOLS[action](arg or "")
                executed_tools.add(key)
                state.idempotency_keys.append(key)
            print(f"  Thought: {thought[:70]}")
            print(f"  Action:  {action}({arg}) → {observation[:80]}")
            state.add_message("user", f"Observation: {observation[:400]}")
        else:
            state.add_message("user", "Observation: formato non valido. Usa Thought/Action/Action Input o Final Answer.")

    print("Limite di passi raggiunto senza Final Answer.")
    return None


if __name__ == "__main__":
    run_agent(
        task="Ispeziona la cartella di lavoro, raccogli informazioni sul checkpointing "
             "proattivo e sul rate limiting, fai i calcoli di budget necessari e "
             "produci un riepilogo finale.",
        backend_type="mock",              # cambia in "litellm" per Ollama/OpenAI/...
        model="ollama/llama3.2",
        initial_remaining_tokens=6500,
    )
