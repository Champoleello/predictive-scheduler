#!/usr/bin/env python3
"""
WARM START VERO — Esperimento KV-cache su disco con llama.cpp
==============================================================

Questo esperimento misura la differenza tra i tre livelli di ripartenza (§8.4
del rapporto) usando un modello REALE che gira sul tuo Mac:

  1. SESSIONE DI LAVORO   → l'agente costruisce un contesto lungo (il modello
                            lo "digerisce" nella sua KV-cache)
  2. CHECKPOINT           → la KV-cache viene serializzata SU DISCO
                            (POST /slots/0?action=save)
  3. RIAVVIO SIMULATO     → la memoria del modello viene azzerata
                            (POST /slots/0?action=erase)
  4. RIPRESA FREDDA       → re-invio dell'intero contesto: il modello deve
                            ricalcolare tutto il prefill (warm start LOGICO)
  5. RIPRESA CALDA        → ripristino della KV-cache dal disco
                            (POST /slots/0?action=restore): il prefill non
                            viene ricalcolato (warm start VERO)

Il confronto tra (4) e (5) — token ri-processati e millisecondi — è il dato
che quantifica il "paradosso del costo zero" (§8.4).

Prerequisito: llama-server in esecuzione (usa avvia_warm_start_kv.command).
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
KV_FILE = "agente_kv.bin"          # file della KV-cache dentro --slot-save-path
CHECKPOINT_FILE = "checkpoint_kv.json"


# ---------------------------------------------------------------------------
# Chiamate al server llama.cpp
# ---------------------------------------------------------------------------

def server_ok() -> bool:
    try:
        return requests.get(f"{SERVER}/health", timeout=5).status_code == 200
    except Exception:
        return False


def completion(prompt: str, n_predict: int = 48):
    """Chiede una generazione e ritorna (testo, timings)."""
    r = requests.post(f"{SERVER}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.3,
        "cache_prompt": True,   # riusa la KV-cache del prefisso, se presente
        "id_slot": 0,           # forza l'uso dello slot 0 (quello che salviamo)
    }, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d.get("content", ""), d.get("timings", {})


def slot_action(action: str):
    """save / restore / erase della KV-cache dello slot 0 (con diagnostica)."""
    payload = {"filename": KV_FILE} if action in ("save", "restore") else {}
    t0 = time.time()
    r = requests.post(f"{SERVER}/slots/0?action={action}", json=payload, timeout=600)
    ms = (time.time() - t0) * 1000
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:200]}
    print(f"      [debug {action}] HTTP {r.status_code} → {json.dumps(body)[:180]}")
    r.raise_for_status()
    return ms


def kv_file_size_mb() -> float:
    import os
    path = os.path.join("kv_cache", KV_FILE)
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


# ---------------------------------------------------------------------------
# Contesto "da agente": storia lunga che simula molti passi di lavoro
# ---------------------------------------------------------------------------

def build_agent_context() -> str:
    intro = ("Sei un agente autonomo che sta analizzando un sistema distribuito. "
             "Di seguito il registro completo dei passi già eseguiti.\n\n")
    steps = []
    for i in range(1, 41):
        steps.append(
            f"PASSO {i}: analizzato il modulo servizio_{i:02d}. "
            f"Trovate {3 + i % 5} dipendenze, latenza media {80 + i * 3} ms, "
            f"note: il modulo richiede refactoring della gestione errori e "
            f"presenta accoppiamento stretto con il servizio di autenticazione. "
        )
    question = ("\n\nDOMANDA: sulla base di tutti i passi precedenti, "
                "qual è il modulo più critico da rifattorizzare per primo? "
                "Rispondi in una frase.")
    return intro + "\n".join(steps) + question


# ---------------------------------------------------------------------------
# Esperimento
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  WARM START VERO — KV-cache su disco (llama.cpp)")
    print("=" * 78)

    if not server_ok():
        print("ERRORE: llama-server non risponde su", SERVER)
        print("Avvialo prima con: avvia_warm_start_kv.command")
        sys.exit(1)

    ctx = build_agent_context()
    results = {}

    # --- 1. SESSIONE DI LAVORO -------------------------------------------
    print("\n[1/5] Sessione di lavoro: il modello digerisce il contesto lungo...")
    text, t = completion(ctx)
    results["lavoro"] = t
    print(f"      Prefill: {t.get('prompt_n', '?')} token in {t.get('prompt_ms', 0):.0f} ms")
    print(f"      Risposta: {text.strip()[:90]}...")

    # --- 2. CHECKPOINT: KV-cache SU DISCO ---------------------------------
    print("\n[2/5] Checkpoint: serializzo la KV-cache su disco...")
    ms = slot_action("save")
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"kv_file": KV_FILE, "context_chars": len(ctx)}, f)
    print(f"      Salvata in {ms:.0f} ms | dimensione file: {kv_file_size_mb():.1f} MB")
    if kv_file_size_mb() < 0.5:
        print("      ⚠ Il file è troppo piccolo: il salvataggio NON ha catturato la KV-cache.")

    # --- 3. RIAVVIO SIMULATO ----------------------------------------------
    print("\n[3/5] Riavvio simulato: azzero la memoria del modello (erase)...")
    slot_action("erase")

    # --- 4. RIPRESA FREDDA (warm start logico: re-prefill completo) --------
    print("\n[4/5] RIPRESA FREDDA: re-invio tutto il contesto, il modello ricalcola...")
    _, t_cold = completion(ctx)
    results["fredda"] = t_cold
    print(f"      Re-prefill: {t_cold.get('prompt_n', '?')} token in {t_cold.get('prompt_ms', 0):.0f} ms")

    # --- 5. RIPRESA CALDA (warm start VERO: restore dal disco) -------------
    print("\n[5/5] RIPRESA CALDA: azzero di nuovo, poi ripristino la KV dal disco...")
    slot_action("erase")
    ms_restore = slot_action("restore")
    _, t_warm = completion(ctx)
    results["calda"] = t_warm
    print(f"      Restore dal disco: {ms_restore:.0f} ms | "
          f"prefill residuo: {t_warm.get('prompt_n', '?')} token in {t_warm.get('prompt_ms', 0):.0f} ms")

    # --- CONFRONTO ----------------------------------------------------------
    cold_n, cold_ms = t_cold.get("prompt_n", 0), t_cold.get("prompt_ms", 0)
    warm_n, warm_ms = t_warm.get("prompt_n", 0), t_warm.get("prompt_ms", 0)
    warm_total = warm_ms + ms_restore

    print("\n" + "=" * 78)
    print("  RISULTATO — il costo della ripresa (§8.4 del rapporto)")
    print("=" * 78)
    print(f"  Ripresa FREDDA (warm start logico): {cold_n:5d} token ricalcolati | {cold_ms:8.0f} ms")
    print(f"  Ripresa CALDA  (warm start VERO):   {warm_n:5d} token ricalcolati | {warm_total:8.0f} ms (di cui restore {ms_restore:.0f} ms)")
    if warm_total > 0 and cold_ms > 0:
        print(f"\n  → Token risparmiati: {cold_n - warm_n} ({(1 - warm_n / max(cold_n, 1)) * 100:.0f}%)")
        print(f"  → Speedup della ripresa: {cold_ms / warm_total:.1f}×")
    print("=" * 78)

    with open("risultati_warm_start_kv.json", "w") as f:
        json.dump({"fredda": t_cold, "calda": t_warm, "restore_ms": ms_restore}, f, indent=2)
    print("\nRisultati salvati in risultati_warm_start_kv.json — mandameli!")


if __name__ == "__main__":
    main()
