# Resource-Aware Predictive Scheduler for LLM Agents

**Proactive checkpointing for autonomous LLM agents: predict resource exhaustion
*before* the 429 error, suspend gracefully, resume at near-zero cost.**

Current agent frameworks handle persistence *reactively*: they save state after
nodes complete, but crash when the provider rate-limits them. This project adds a
predictive layer between the agent and the LLM provider that:

1. reads real-time rate-limit telemetry (`x-ratelimit-remaining-tokens`, ...)
2. estimates the cost of the *next* step (with moving-average error correction)
3. evaluates a multi-dimensional risk function
4. decides: **continue / compress / summarize / model-switch / graceful checkpoint**
5. serializes state with idempotency keys — and, on KV-capable runtimes,
   the model's KV-cache too (true warm start)

## Validated results

| Experiment | Result |
|---|---|
| Simulation (300 runs/config) | reactive baseline crashes 100% → predictive 0% (k=4), token waste −80% |
| Real provider (Groq free tier, 6K TPM) | 6,043-token task completed across 2 rate windows, **zero 429 errors** |
| True warm start (llama.cpp, KV on disk) | three scales (0.5B/4B/8B): 100% prefill saved, speedup **50× / 63× / 93×**; warm resume ~constant at ~0.5 s |
| Integrated system (decision + KV checkpoint) | transactional suspend in 46 ms, resume re-prefills only the new message |
| vs LangGraph (official MemorySaver) | LangGraph: 7.8× 429/run, 9,239 tokens wasted; predictive: **0 and 0** |
| End-to-end A/B (Qwen3-4B / 8B thinking) | recovery 21.6→0.4 s (**54×**) and 37.3→0.4 s (**93×**, matches the isolated benchmark) |

Key architectural finding: **KV-checkpointing inverts the escalation hierarchy** —
compression/summarization invalidate the KV prefix cache, so on KV-capable runtimes
graceful suspension becomes the *first* choice, not the last resort.

## Layout

```
core/         predictive_scheduler.py — the predictive scheduler (§4–§5 of the paper)
examples/     react_agent.py         — ReAct agent with tools + scheduler
              groq_real_telemetry.py — real rate-limit headers (Groq)
              kv_integrated.py       — transactional state+KV checkpoint (llama.cpp)
experiments/  parameter sweeps, KV warm-start benchmarks, LangGraph comparison
docs/         charts
```

## Quick start

```bash
pip install requests
python core/predictive_scheduler.py      # mock demo, no keys needed

# real rate limits (free Groq key):
export GROQ_API_KEY=gsk_...
python examples/groq_real_telemetry.py

# true warm start (needs llama.cpp):
llama-server -hf unsloth/Qwen3-4B-GGUF:Q4_K_M --slot-save-path ./kv_cache -c 12288
python examples/kv_integrated.py
```

## Paper

See *"Scheduler Predittivo Resource-Aware per Agenti LLM Autonomi"* (v1.5) for the
formalization (risk function, checkpoint rule `T_remaining < T_estimated + k·σ`),
the OS analogy, and full experimental protocol.

## License

MIT
