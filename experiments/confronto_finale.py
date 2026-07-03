#!/usr/bin/env python3
"""
CONFRONTO FINALE END-TO-END — stesso task, con e senza il sistema
==================================================================

Modello thinking reale (Qwen3), task complesso multi-passo, rate limit
simulato con finestre di budget. Due condizioni identiche in tutto:

  A) SENZA il sistema ("killed"): l'agente non guarda il budget. Quando
     una chiamata sfora → 429, la sessione muore (la KV-cache del server
     va persa, come in un vero riavvio del processo). Alla ripresa, dopo
     il reset della finestra, deve RE-INVIARE tutta la conversazione
     (re-prefill completo) e RIFARE il passo fallito.

  B) CON il sistema: lo scheduler prevede lo sforamento PRIMA della
     chiamata, salva stato+KV su disco in una transazione, e alla ripresa
     ripristina la KV: ricalcola solo la domanda nuova.

Misure per condizione: tempo totale, secondi spesi in recupero, token di
prefill ricalcolati nei recuperi, errori 429 subiti, passi rifatti.

Prerequisito: llama-server attivo (avvia_warm_start_kv_pro.command).
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    print("Manca 'requests'. Esegui: pip3 install requests")
    sys.exit(1)

SERVER = "http://127.0.0.1:8080"
KV_FILE = "confronto_finale_kv.bin"
N_PREDICT = 512          # spazio per thinking + risposta
BUDGET = 6500            # token per finestra di rate limit (simulata)
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
    # Contabilità da PROVIDER CLOUD: l'input si paga TUTTO a ogni chiamata
    # (la cache locale riduce il calcolo, non il conteggio del rate limit).
    full_input = sum(estimate_tokens(m["content"]) for m in messages)
    used = int(full_input + t.get("predicted_n", 0))
    return content.strip(), used, t


def slot(action):
    payload = {"filename": KV_FILE} if action in ("save", "restore") else {}
    r = requests.post(f"{SERVER}/slots/0?action={action}", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Il task complesso (identico nelle due condizioni)
# ---------------------------------------------------------------------------

def registry(n=60):
    rows = []
    for i in range(1, n + 1):
        rows.append(f"MODULO servizio_{i:03d}: dipendenze {2 + i % 7}, "
                    f"latenza p95 {60 + (i * 7) % 240} ms, error rate {round(0.1 + (i % 9) * 0.4, 1)}%, "
                    f"copertura test {30 + (i * 13) % 60}%, "
                    f"{'accoppiamento col gateway auth' if i % 3 == 0 else 'gestione errori incompleta' if i % 3 == 1 else 'query N+1 e cache assente'}.")
    return "\n".join(rows)


SYSTEM = ("Sei un architetto software. Ragiona con attenzione e rispondi in italiano, "
          "in modo rigoroso ma conciso (max 150 parole per risposta).")

QUESTIONS = [
    "Analizza il registro e identifica i 3 moduli più critici, motivando con le metriche.",
    "Per ciascuno dei 3, stima l'impatto di un guasto sulla catena delle dipendenze.",
    "Proponi l'ordine di refactoring ottimale e giustificalo confrontando rischio e costo.",
    "Scrivi il piano operativo: 3 interventi concreti per il modulo più urgente.",
]


def initial_messages():
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "REGISTRO ISPEZIONI:\n" + registry() +
             "\n\nRispondi alle domande che seguiranno una alla volta."},
            ]


# ---------------------------------------------------------------------------
# Stima e budget (identici nelle due condizioni; li usa solo la B per decidere)
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
# CONDIZIONE A — senza sistema: killed & cold resume
# ---------------------------------------------------------------------------

def run_condition_a():
    print("\n" + "=" * 78)
    print("  CONDIZIONE A — SENZA sistema (killed → cold resume)")
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

        # l'agente NON guarda il budget: chiama e basta
        if est_input + N_PREDICT > remaining:
            # ---- 429: la sessione muore ----
            stats["errors_429"] += 1
            stats["steps_redone"] += 1
            print(f"  passo {step + 1}: ❌ 429! sessione persa, attendo il reset...")
            remaining = BUDGET                    # finestra resettata
            slot("erase")                         # processo morto → KV persa
            t0 = time.time()
            # cold resume: re-prefill dell'INTERA conversazione (stessa chiamata)
            text, used, t = completion(messages)
            dt = time.time() - t0
            stats["recovery_s"] += dt
            stats["reprefill_tokens"] += int(t.get("prompt_n", 0))
            print(f"  passo {step + 1}: recupero freddo — {t.get('prompt_n', 0):.0f} tok "
                  f"di re-prefill in {dt:.1f} s")
        else:
            text, used, t = completion(messages)
            print(f"  passo {step + 1}: ok ({t.get('prompt_n', 0):.0f} tok prefill, "
                  f"{used} usati, budget {remaining})")
        remaining -= used
        messages.append({"role": "assistant", "content": text[:600]})
        step += 1

    stats["total_s"] = time.time() - stats["t0"]
    return stats


# ---------------------------------------------------------------------------
# CONDIZIONE B — con il sistema: predizione + KV-checkpoint
# ---------------------------------------------------------------------------

def run_condition_b():
    print("\n" + "=" * 78)
    print("  CONDIZIONE B — CON il sistema (predizione + KV warm start)")
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
            # ---- checkpoint PRIMA dell'errore ----
            messages.pop()                        # la domanda si rifà alla ripresa
            t0 = time.time()
            info = slot("save")                   # KV su disco (+ stato: qui in RAM)
            print(f"  passo {step + 1}: 🛑 checkpoint predittivo "
                  f"({info.get('n_saved', 0)} celle KV) — attendo il reset...")
            remaining = BUDGET
            slot("erase")                         # "riavvio del processo"
            slot("restore")                       # warm start VERO dal disco
            stats["recovery_s"] += time.time() - t0
            continue                              # nessun lavoro perso

        text, used, t = completion(messages)
        hist.append(used)
        print(f"  passo {step + 1}: ok ({t.get('prompt_n', 0):.0f} tok prefill, "
              f"{used} usati, budget {remaining})")
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
        print("ERRORE: llama-server non attivo (usa avvia_warm_start_kv_pro.command)")
        sys.exit(1)

    print("=" * 78)
    print("  CONFRONTO FINALE — stesso task complesso, modello thinking")
    print(f"  4 domande di analisi | budget per finestra: {BUDGET} token")
    print("=" * 78)

    a = run_condition_a()
    b = run_condition_b()

    print("\n" + "=" * 78)
    print("  VERDETTO")
    print("=" * 78)
    print(f"{'':38s} {'A (senza)':>12s} {'B (con)':>12s}")
    print("-" * 66)
    print(f"{'Errori 429 subiti':38s} {a['errors_429']:12d} {b['errors_429']:12d}")
    print(f"{'Passi rifatti':38s} {a['steps_redone']:12d} {b['steps_redone']:12d}")
    print(f"{'Token ri-processati nei recuperi':38s} {a['reprefill_tokens']:12d} {b['reprefill_tokens']:12d}")
    print(f"{'Tempo speso in recupero':38s} {a['recovery_s']:11.1f}s {b['recovery_s']:11.1f}s")
    print(f"{'Tempo totale del task':38s} {a['total_s']:11.1f}s {b['total_s']:11.1f}s")
    print("=" * 78)

    with open("risultati_confronto_finale.json", "w") as f:
        json.dump({"A_senza": a, "B_con": b}, f, indent=2)
    print("\nSalvato in risultati_confronto_finale.json — mandamelo!")


if __name__ == "__main__":
    main()
