#!/usr/bin/env python3
"""
ReAct Agent v5 — tool-calling agent + Predictive Scheduler
================================================================

The prototype becomes a REAL AGENT (§3.1 of the report):

    Agent (ReAct / tool-calling loop)
        ↓
    Resource-Aware Predictive Scheduler   ← decides continue/compress/.../checkpoint
        ↓
    LLM Provider (Mock, Ollama via LiteLLM, ...)

The classic ReAct loop:
    Thought  → the agent reasons about what to do
    Action   → it picks a tool (read_file, list_files, search, calculator, write_note)
    Observation → it receives the tool result
    ... repeats until it produces a "Final Answer".

The scheduler steps in BEFORE every LLM call: when resources run low it tries
compression → summarization → model switch → and, as a last resort, performs a
graceful checkpoint. On restart the agent resumes where it stopped (warm start),
without re-running tools that already ran (idempotency keys, §3.1).

Usage:
    python3 react_agent.py                  # demo with MockBackend (SIMULATED numbers)
    (at the bottom of the file you can switch to "litellm" + Ollama)
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
# 1. TOOLS — simple but real
# =============================================================================

def tool_list_files(arg: str) -> str:
    import os
    try:
        files = sorted(os.listdir(arg or "."))[:20]
        return "Files in the folder: " + ", ".join(files)
    except Exception as e:
        return f"Error: {e}"


def tool_read_file(arg: str) -> str:
    try:
        with open(arg) as f:
            content = f.read(1500)
        return f"Content of {arg} (first 1500 characters):\n{content}"
    except Exception as e:
        return f"Error: {e}"


def tool_search(arg: str) -> str:
    """Simulated search (no network). Replaceable with a real API."""
    fake_db = {
        "checkpoint": "Proactive checkpointing saves state before resources run out.",
        "rate limit": "LLM providers expose headers like x-ratelimit-remaining-tokens.",
        "react": "ReAct alternates reasoning (Thought) and actions (Action) with tools.",
    }
    for key, val in fake_db.items():
        if key in arg.lower():
            return f"[search] {val}"
    return f"[search] No exact result for '{arg}'. Try different terms."


def tool_calculator(arg: str) -> str:
    try:
        if not re.fullmatch(r"[0-9+\-*/(). %]+", arg):
            return "Error: invalid expression (only digits and + - * / % parentheses)."
        return f"Result: {eval(arg)}"  # input already validated by the regex
    except Exception as e:
        return f"Error: {e}"


def tool_write_note(arg: str) -> str:
    with open("agent_notes.txt", "a") as f:
        f.write(arg + "\n")
    return "Note saved to agent_notes.txt"


TOOLS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "calculator": tool_calculator,
    "write_note": tool_write_note,
}

SYSTEM_PROMPT = """You are an autonomous agent. Solve the task using the available tools.
ALWAYS reply in this format (one triplet at a time):

Thought: <your reasoning>
Action: <tool_name>
Action Input: <tool argument>

Available tools: list_files, read_file, search, calculator, write_note.

When you are done, reply with:
Thought: <final reasoning>
Final Answer: <complete answer to the task>
"""


# =============================================================================
# 2. SCRIPTED MOCK — a fake LLM that speaks the ReAct format
#    (lets you test the agent without a real model; numbers are SIMULATED)
# =============================================================================

class ScriptedReActBackend(BaseLLMBackend):
    name = "mock-react"

    SCRIPT = [
        ("I need to see what the working folder contains.", "list_files", "."),
        ("Let me look up information on checkpointing.", "search", "proactive checkpoint"),
        ("Let me check how rate limits work.", "search", "rate limit header"),
        ("Let me run a sample budget calculation.", "calculator", "6500 - 320*12"),
        ("Let me save a note about the results.", "write_note", "Estimated remaining budget after 12 steps: 2660 tokens"),
        ("Let me dig into the ReAct pattern.", "search", "react pattern"),
        ("Let me re-check the generated files.", "list_files", "."),
        ("Compute the safety margin with k=2 and sigma=90.", "calculator", "2*90"),
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
            text = ("Thought: I have gathered enough information.\n"
                    "Final Answer: Analysis complete: folder inspected, key concepts "
                    "verified (proactive checkpointing, rate-limit headers, ReAct pattern) and "
                    "budget calculations saved to agent_notes.txt.")
        return text, min(tokens, max_tokens)

    def estimate_tokens(self, text: str) -> int:
        return max(8, len(text) // 4)


# =============================================================================
# 3. ReAct PARSER
# =============================================================================

def parse_react(text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Returns (thought, action, action_input, final_answer)."""
    # Multiline Thought: stops at Action / Final Answer (improvement from external review)
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
# 4. AGENT LOOP with integrated scheduler
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
        economy.call_count = 99  # the economy model goes straight to the final answer

    scheduler = PredictiveScheduler(
        backend=backend,
        economy_backend=economy,
        initial_remaining_tokens=initial_remaining_tokens,
        safety_factor_k=2.0,
    )

    # ---- Warm start (§3.1): resumes from the checkpoint if present ----
    state = scheduler.load_checkpoint(CHECKPOINT_FILE)
    if state:
        scheduler.remaining_tokens = initial_remaining_tokens  # window has reset
        # tools already run: do NOT re-run them (tool keys have the "action|input" format)
        executed_tools = {k for k in state.idempotency_keys if "|" in k}
        if isinstance(backend, ScriptedReActBackend):
            backend.call_count = state.step                    # realign the script
        print(f"↻ WARM START: resuming from step {state.step + 1}, "
              f"{len(executed_tools)} actions already executed (they will not be repeated)")
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
        # --- The scheduler decides and (if needed) acts BEFORE the LLM call ---
        d = scheduler.execute_step(state)
        print(f"[scheduler] step {d['step']:2d} | rem {d['remaining']:5d} | "
              f"risk {d['risk']:.2f} | {icons[d['action']]} {d['action']}")

        if d["action"] == "checkpoint":
            scheduler.save_checkpoint(state, CHECKPOINT_FILE)
            print(f"\n🛑 GRACEFUL CHECKPOINT → {CHECKPOINT_FILE}")
            print("   Run the script again to resume from here (warm start).")
            return None

        # --- The assistant's last reply is the ReAct move ---
        llm_text = state.messages[-1]["content"]
        thought, action, arg, final = parse_react(llm_text)

        if final:
            print("-" * 78)
            print(f"✅ FINAL ANSWER: {final}")
            print(f"Steps: {state.step} | Tokens used: {state.total_tokens_used}")
            print("\nScheduler metrics (§8.2):")
            for k, v in scheduler.metrics_summary().items():
                print(f"  {k}: {v}")
            # task finished → the checkpoint is no longer needed
            import os
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
            return final

        if action and action in TOOLS:
            # idempotency key: the same action+input is never re-executed (§3.1)
            key = f"{action}|{arg}"
            if key in executed_tools:
                observation = "[skip] Action already executed before the checkpoint (idempotency)."
            else:
                observation = TOOLS[action](arg or "")
                executed_tools.add(key)
                state.idempotency_keys.append(key)
            print(f"  Thought: {thought[:70]}")
            print(f"  Action:  {action}({arg}) → {observation[:80]}")
            state.add_message("user", f"Observation: {observation[:400]}")
        else:
            state.add_message("user", "Observation: invalid format. Use Thought/Action/Action Input or Final Answer.")

    print("Step limit reached without a Final Answer.")
    return None


if __name__ == "__main__":
    run_agent(
        task="Inspect the working folder, gather information on proactive "
             "checkpointing and rate limiting, run the necessary budget "
             "calculations and produce a final summary.",
        backend_type="mock",              # switch to "litellm" for Ollama/OpenAI/...
        model="ollama/llama3.2",
        initial_remaining_tokens=6500,
    )
