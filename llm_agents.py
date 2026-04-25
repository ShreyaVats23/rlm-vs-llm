"""
llm_agents.py — Linear LLM Multi-Agent Pipeline (Baseline)
===========================================================
Architecture: Three-agent linear pipeline.
  LLM-A  Orchestrator — plans subtasks, delegates, synthesises.
  LLM-B  Researcher   — academic theory and explanation.
  LLM-C  Implementer  — examples, comparisons, code.

Provider is resolved at import time from .env:
    API_PROVIDER=groq | gemini | openai
    API_KEY=...
No other file needs to change when you switch APIs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from a2a_bus import A2ABus, A2ATask
from config import (
    BACKOFF_BASE,
    BACKOFF_FACTOR,
    BACKOFF_MAX,
    LLM_MAX_TOKENS,
    MAX_RETRIES,
    MODEL,
)
from llm_provider import llm_complete


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalise(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _extract_subtasks(plan_raw: str, query: str) -> list[str]:
    """Parse the orchestrator's planning output into exactly two subtask strings."""
    try:
        parsed = json.loads(plan_raw)
        if isinstance(parsed, list) and len(parsed) >= 2:
            return [str(parsed[0]).strip(), str(parsed[1]).strip()]
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*?\]", plan_raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) >= 2:
                return [str(parsed[0]).strip(), str(parsed[1]).strip()]
        except Exception:
            pass

    return [
        f"Explain and analyse the core theory and concepts for: {query}",
        f"Provide structured comparison, examples, and code if relevant for: {query}",
    ]


def _heuristic_confidence(answer: str) -> float:
    score  = 0.55
    score += min(len(answer) / 2_000, 0.15)
    if any(k in answer.lower() for k in ("conclusion", "comparison", "example", "analysis", "however")):
        score += 0.05
    return round(min(score, 0.78), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Base agent
# ──────────────────────────────────────────────────────────────────────────────
class LLMAgent:
    """
    Stateless LLM agent.  Uses llm_provider.llm_complete — provider-agnostic.
    """

    def __init__(self, name: str, role: str) -> None:
        self.name   = name
        self.role   = role
        self._calls = 0

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        attempt: int = 0,
    ) -> str:
        try:
            msg = await asyncio.to_thread(
                llm_complete,
                MODEL,
                messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.2,
                system=self.role,
            )
            self._calls += 1
            return _normalise(msg.content)

        except Exception as exc:
            error = str(exc)
            if ("quota" in error.lower() or "rate" in error.lower()) and attempt < MAX_RETRIES:
                wait = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
                print(f"  [{self.name}] Rate-limit — backing off {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
                return await self.call(messages, attempt=attempt + 1)

            print(f"  [{self.name}] API error: {error[:160]}")
            return "Could not generate answer due to an API failure."


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — LLM-A
# ──────────────────────────────────────────────────────────────────────────────
class LLMOrchestrator(LLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "LLM-A",
            role = (
                "You are a structured reasoning orchestrator. "
                "Break tasks into two focused subtasks, then synthesise specialist "
                "outputs into a clear, complete, well-structured final answer. "
                "For planning, return ONLY a valid JSON array of 2 subtask strings."
            ),
        )
        self.bus = bus

    async def _delegate(self, receiver: str, subtask: str) -> dict:
        task = A2ATask(sender="LLM-A", receiver=receiver, payload={"instruction": subtask})
        print(f"  [A2A] LLM-A -> {receiver}: {subtask[:90]}...")
        return await self.bus.send(task, timeout=90.0)

    async def run(self, query: str) -> dict[str, Any]:
        t0          = time.monotonic()
        self._calls = 0

        print(f"\n{'=' * 70}")
        print(f"  LLM-A | {query[:68]}...")
        print(f"{'=' * 70}")

        # Step 1: Plan
        print("  [LLM-A] Step 1/4 — Planning")
        plan_raw = await self.call([{
            "role":    "user",
            "content": (
                "Break this task into exactly 2 subtasks as a JSON array.\n"
                "Subtask 1: theory, concepts, and academic explanation.\n"
                "Subtask 2: structured comparison, examples, and code if relevant.\n\n"
                f"Task: {query}\n\n"
                'Return format: ["subtask one", "subtask two"]'
            ),
        }])
        print(f"  [LLM-A] Plan: {plan_raw[:140]}...")
        subtasks = _extract_subtasks(plan_raw, query)
        for i, s in enumerate(subtasks, 1):
            print(f"    {i}. {s[:110]}")

        # Step 2: Parallel delegation
        print("  [LLM-A] Step 2/4 — Delegating to LLM-B and LLM-C (parallel)")
        r_b, r_c = await asyncio.gather(
            self._delegate("LLM-B", subtasks[0]),
            self._delegate("LLM-C", subtasks[1]),
        )
        b_out = _normalise(r_b.get("result", ""))
        c_out = _normalise(r_c.get("result", ""))

        # Step 3: Draft synthesis
        print("  [LLM-A] Step 3/4 — Drafting synthesis")
        draft = await self.call([{
            "role":    "user",
            "content": (
                "Combine the following specialist outputs into one structured answer.\n"
                "Use ## headings and cover all important points. Do not repeat yourself.\n\n"
                f"## Research (Agent B):\n{b_out}\n\n"
                f"## Implementation (Agent C):\n{c_out}"
            ),
        }])

        # Step 4: Refine + Verify
        print("  [LLM-A] Step 4/4 — Refine & Verify")
        final = await self.call([{
            "role":    "user",
            "content": (
                "You have a draft answer and the original task. Perform both jobs in one pass:\n\n"
                "1. IMPROVE the draft — clearer, more complete, ## headings, ## Conclusion.\n"
                "2. VERIFY — ensure all parts of the original task are fully addressed.\n\n"
                "Return ONLY the improved final answer — no commentary.\n\n"
                f"Original task:\n{query}\n\n"
                f"Draft:\n{draft}"
            ),
        }])

        elapsed     = round(time.monotonic() - t0, 2)
        total_calls = self._calls + r_b.get("calls", 0) + r_c.get("calls", 0)
        print(f"  [LLM-A] Done — time={elapsed}s  total_calls={total_calls}")

        return {
            "answer":     final,
            "time_s":     elapsed,
            "tool_calls": 0,
            "ctx_depth":  1,
            "calls":      total_calls,
            "confidence": _heuristic_confidence(final),
            "subtasks":   subtasks,
            "steps":      4,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Specialist B — Research & Theory
# ──────────────────────────────────────────────────────────────────────────────
class LLMSpecialistB(LLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "LLM-B",
            role = (
                "You are a research and theory specialist. "
                "Provide structured, accurate, academically grounded explanations. "
                "Use ## headings, clear definitions, and logical flow."
            ),
        )
        self.bus = bus

    async def run(self) -> None:
        print("  [LLM-B] Listening...")
        while True:
            task        = await self.bus.listen("LLM-B")
            self._calls = 0
            instruction = _normalise(task.payload.get("instruction", ""))
            print(f"  [LLM-B] Received: {instruction[:100]}...")

            result = await self.call([{
                "role":    "user",
                "content": (
                    "Answer with structured ## headings and clear explanations.\n"
                    "Cover: background, core concepts, key distinctions, and limitations.\n"
                    "Make the explanation deep, logical, and well-structured in a single pass.\n\n"
                    f"{instruction}"
                ),
            }])

            print("  [LLM-B] Responding to LLM-A")
            self.bus.respond(task, {"result": result, "tool_calls": 0, "ctx_depth": 0, "calls": self._calls})


# ──────────────────────────────────────────────────────────────────────────────
# Specialist C — Implementation & Comparison
# ──────────────────────────────────────────────────────────────────────────────
class LLMSpecialistC(LLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "LLM-C",
            role = (
                "You are an implementation and comparison specialist. "
                "Provide practical examples, comparison tables, and clean code when relevant. "
                "Be structured, concrete, and actionable."
            ),
        )
        self.bus = bus

    async def run(self) -> None:
        print("  [LLM-C] Listening...")
        while True:
            task        = await self.bus.listen("LLM-C")
            self._calls = 0
            instruction = _normalise(task.payload.get("instruction", ""))
            print(f"  [LLM-C] Received: {instruction[:100]}...")

            result = await self.call([{
                "role":    "user",
                "content": (
                    "Solve this with clear ## section headings, concrete examples, and code if relevant.\n"
                    "Include a comparison table where applicable.\n"
                    "Ensure examples are specific and any code is correct and well-commented.\n\n"
                    f"{instruction}"
                ),
            }])

            print("  [LLM-C] Responding to LLM-A")
            self.bus.respond(task, {"result": result, "tool_calls": 0, "ctx_depth": 0, "calls": self._calls})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
async def run_llm_system(query: str) -> dict[str, Any]:
    bus  = A2ABus()
    orch = LLMOrchestrator(bus)
    sp_b = LLMSpecialistB(bus)
    sp_c = LLMSpecialistC(bus)

    for name in ("LLM-A", "LLM-B", "LLM-C"):
        bus.register(name)

    t_b = asyncio.create_task(sp_b.run())
    t_c = asyncio.create_task(sp_c.run())

    try:
        return await orch.run(query)
    finally:
        t_b.cancel()
        t_c.cancel()


if __name__ == "__main__":
    _query  = "What is the difference between a list and a tuple in Python?"
    _result = asyncio.run(run_llm_system(_query))

    print("\n" + "=" * 70)
    print("LLM FINAL ANSWER")
    print("=" * 70)
    print(_result["answer"])
    print(f"\nTime:       {_result['time_s']}s")
    print(f"Ctx depth:  {_result['ctx_depth']}")
    print(f"Tool calls: {_result['tool_calls']}")
    print(f"Confidence: {_result['confidence']}")