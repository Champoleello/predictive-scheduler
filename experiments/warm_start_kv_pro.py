#!/usr/bin/env python3
"""
WARM START VERO — Esperimento PRO: modello thinking + task grande
==================================================================

Differenze rispetto all'esperimento base (warm_start_kv.py):

  • Modello THINKING (Qwen3): ragiona dentro <think>...</think> prima di
    rispondere — scelto automaticamente in base alla RAM del Mac.
  • Task GRANDE: contesto da ~7.000 token (vs 2.700) — la KV-cache su disco
    peserà ~1 GB e il divario freddo/caldo diventa molto più evidente.
  • Misure RIPETUTE: ogni ripresa (fredda e calda) è misurata N volte e
    viene riportata la media — valutazione più esatta.
  • Prompt costruito con il chat template ufficiale del modello
    (endpoint /apply-template), così il thinking si attiva correttamente.

Prerequisito: llama-server avviato da avvia_warm_start_kv_pro.command.
"""

import json
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("Manca 'requests'. Esegui: pip3 install requests")
    sys.exit(1)

SERVER = "http://127.0.0.1:8080"
KV_FILE = "agente_kv_pro.bin"
REPS = 2          # ripetizioni per misura (alza a 3 se vuoi ancora più precisione)
N_PREDICT = 640   # spazio per il thinking + risposta


# ---------------------------------------------------------------------------
# Server API
# ---------------------------------------------------------------------------

def server_ok() -> bool:
    try:
        return requests.get(f"{SERVER}/health", timeout=5).status_code == 200
    except Exception:
        return False


def apply_template(messages) -> str:
    """Usa il chat template ufficiale del modello (attiva il thinking)."""
    try:
        r = requests.post(f"{SERVER}/apply-template", json={"messages": messages}, timeout=30)
        if r.status_code == 200:
            return r.json()["prompt"]
    except Exception:
        pass
    # Fallback: template ChatML (quello di Qwen)
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def completion(prompt: str, n_predict: int = N_PREDICT):
    r = requests.post(f"{SERVER}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.6, "top_p": 0.95,   # raccomandati per Qwen3 thinking
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
# Task grande: registro di ~120 passi di un agente (≈7.000 token)
# ---------------------------------------------------------------------------

def build_big_context() -> str:
    system = ("Sei un agente autonomo esperto di architetture software. "
              "Analizzi sistemi distribuiti e produci raccomandazioni di refactoring "
              "motivate e prioritizzate.")
    log = []
    for i in range(1, 121):
        log.append(
            f"PASSO {i}: ispezionato il modulo servizio_{i:03d}. "
            f"Dipendenze dirette: {2 + i % 7}; latenza p95: {60 + (i * 7) % 240} ms; "
            f"error rate: {round(0.1 + (i % 9) * 0.4, 1)}%; copertura test: {30 + (i * 13) % 60}%. "
            f"Osservazioni: {'accoppiamento stretto con il gateway di autenticazione' if i % 3 == 0 else 'gestione errori incompleta nei percorsi asincroni' if i % 3 == 1 else 'query N+1 verso il database ordini e cache assente'}. "
        )
    question = ("Analizza l'intero registro qui sopra. Identifica i 3 moduli più critici "
                "da rifattorizzare, spiega perché proprio quei tre confrontando le metriche, "
                "e proponi l'ordine di intervento.")
    user = "REGISTRO DELLE ISPEZIONI:\n" + "\n".join(log) + "\n\n" + question
    return apply_template([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


# ---------------------------------------------------------------------------
# Esperimento con misure ripetute
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  WARM START VERO — PRO: modello thinking, task grande, misure ripetute")
    print("=" * 78)

    if not server_ok():
        print("ERRORE: llama-server non risponde. Avvialo con avvia_warm_start_kv_pro.command")
        sys.exit(1)

    prompt = build_big_context()
    print(f"\nContesto costruito: ~{len(prompt) // 4} token stimati")

    # --- 1. Sessione di lavoro (con thinking) -----------------------------
    print("\n[1] Sessione di lavoro: il modello ragiona sul task grande...", flush=True)
    text, t = completion(prompt)
    print(f"    Prefill: {t.get('prompt_n', '?')} token in {t.get('prompt_ms', 0) / 1000:.1f} s "
          f"({t.get('prompt_per_second', 0):.0f} tok/s)")
    if "<think>" in text:
        think = text.split("<think>")[1].split("</think>")[0].strip()
        answer = text.split("</think>")[-1].strip()
        print(f"    Thinking (estratto): {think[:140]}...")
        print(f"    Risposta (estratto): {answer[:140]}...")
    else:
        print(f"    Risposta (estratto): {text.strip()[:140]}...")

    # --- 2. Checkpoint della KV su disco -----------------------------------
    print("\n[2] Checkpoint: serializzo la KV-cache su disco...", flush=True)
    ms_save, _ = slot_action("save")
    print(f"    Salvata in {ms_save:.0f} ms | file: {kv_file_size_mb():.0f} MB")

    # --- 3+4. Misure ripetute: FREDDA vs CALDA ------------------------------
    cold_ms, cold_n = [], []
    warm_ms, warm_n, restore_ms = [], [], []

    for rep in range(1, REPS + 1):
        print(f"\n[3] Ripresa FREDDA (misura {rep}/{REPS}): erase + re-prefill completo...", flush=True)
        slot_action("erase", quiet=True)
        _, tc = completion(prompt, n_predict=8)   # pochi token: misuriamo il prefill
        cold_ms.append(tc.get("prompt_ms", 0)); cold_n.append(tc.get("prompt_n", 0))
        print(f"    {tc.get('prompt_n', '?')} token ricalcolati in {tc.get('prompt_ms', 0) / 1000:.1f} s")

        print(f"[4] Ripresa CALDA (misura {rep}/{REPS}): erase + restore dal disco...", flush=True)
        slot_action("erase", quiet=True)
        ms_r, _ = slot_action("restore", quiet=True)
        _, tw = completion(prompt, n_predict=8)
        warm_ms.append(tw.get("prompt_ms", 0)); warm_n.append(tw.get("prompt_n", 0)); restore_ms.append(ms_r)
        print(f"    restore {ms_r:.0f} ms + {tw.get('prompt_n', '?')} token in {tw.get('prompt_ms', 0):.0f} ms")

    # --- Riepilogo -----------------------------------------------------------
    c_ms, c_n = statistics.mean(cold_ms), statistics.mean(cold_n)
    w_ms, w_n = statistics.mean(warm_ms), statistics.mean(warm_n)
    r_ms = statistics.mean(restore_ms)
    warm_total = w_ms + r_ms

    print("\n" + "=" * 78)
    print(f"  RISULTATO (media su {REPS} misure)")
    print("=" * 78)
    print(f"  Ripresa FREDDA: {c_n:7.0f} token ricalcolati | {c_ms / 1000:8.1f} s")
    print(f"  Ripresa CALDA:  {w_n:7.0f} token ricalcolati | {warm_total / 1000:8.2f} s (restore {r_ms / 1000:.2f} s)")
    print(f"\n  → Token risparmiati: {c_n - w_n:.0f} ({(1 - w_n / max(c_n, 1)) * 100:.0f}%)")
    print(f"  → Speedup della ripresa: {c_ms / max(warm_total, 1):.1f}×")
    print(f"  → KV-cache su disco: {kv_file_size_mb():.0f} MB")
    print("=" * 78)

    with open("risultati_warm_start_kv_pro.json", "w") as f:
        json.dump({"cold_ms": cold_ms, "cold_n": cold_n, "warm_ms": warm_ms,
                   "warm_n": warm_n, "restore_ms": restore_ms,
                   "kv_mb": kv_file_size_mb(), "reps": REPS}, f, indent=2)
    print("\nSalvato in risultati_warm_start_kv_pro.json — mandamelo!")


if __name__ == "__main__":
    main()
