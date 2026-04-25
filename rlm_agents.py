"""
rlm_agents.py — Recursive Language Model (RLM) Agent System
============================================================
Architecture: Three-agent recursive orchestration with:
  - Persistent recursive context (rlm_context.py)
  - MCP tool integration (search, shared memory)
  - A2A task bus for structured agent-to-agent delegation
  - Adaptive complexity routing (simple → multi-hop)
  - Safer fallback behaviour under API/rate-limit pressure

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
    FINAL_MIN_LEN,
    HEAVY_SIGNALS,
    MAX_RETRIES,
    MODEL_A,
    MULTIHOP_SIGNALS,
    RLM_MAX_TOKENS,
    TOOL_BUDGET,
)
from llm_provider import llm_complete
from mcp_tools import MCPRegistry
from rlm_context import RecursiveContext


# ──────────────────────────────────────────────────────────────────────────────
# Global throttling (prevents agent burst collisions)
# ──────────────────────────────────────────────────────────────────────────────
_GLOBAL_LLM_LOCK     = asyncio.Lock()
_GLOBAL_LAST_CALL_TS = 0.0
_GLOBAL_MIN_GAP_S    = 0.35


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalise(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _safe_json(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def _extract_json_array(text: str) -> list[str] | None:
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return None
    parsed = _safe_json(match.group())
    if isinstance(parsed, list):
        return [_normalise(i) for i in parsed if _normalise(i)]
    return None


def detect_complexity(query: str) -> str:
    """Route a query to one of four complexity tiers."""
    q  = query.lower().strip()
    wc = len(query.split())

    if any(sig in q for sig in MULTIHOP_SIGNALS):
        return "multi-hop"
    if any(sig in q for sig in HEAVY_SIGNALS) or wc > 22:
        return "heavy"
    if wc < 14:
        return "simple"
    return "medium"


def _budget(complexity: str, key: str) -> Any:
    return TOOL_BUDGET.get(complexity, TOOL_BUDGET["medium"])[key]


def _needs_code(query: str) -> bool:
    return any(
        kw in query.lower()
        for kw in ("python", "code", "implement", "script",
                   "benchmark", "algorithm", "function", "program")
    )


def _extract_subtasks(plan_raw: str, query: str) -> list[str]:
    """Parse the orchestrator's JSON plan into exactly 2 subtask strings."""
    parsed = _extract_json_array(plan_raw)
    if parsed and len(parsed) >= 2:
        good = [s for s in parsed[:2] if len(s) > 15]
        if len(good) == 2:
            return good

    lines: list[str] = []
    for line in plan_raw.splitlines():
        clean = re.sub(r"^[\-\*\d\.\)\s]+", "", line.strip())
        if len(clean) > 20:
            lines.append(clean)
    if len(lines) >= 2:
        return lines[:2]

    if _needs_code(query):
        return [
            f"Research the theory, concepts, trade-offs, and academic background for: {query}",
            f"Implement code, provide benchmarks, practical examples, and system analysis for: {query}",
        ]

    return [
        f"Analyse the core concepts, limitations, assumptions, and context depth for: {query}",
        f"Produce a structured comparison, evidence-backed reasoning, and mitigation synthesis for: {query}",
    ]


def _heuristic_confidence(answer: str, tool_calls: int, steps: int) -> float:
    score  = 0.40
    score += min(len(answer) / 1600, 0.22)
    score += min(tool_calls * 0.06, 0.18)
    score += min(steps * 0.04, 0.14)
    if any(k in answer.lower() for k in
           ("conclusion", "analysis", "comparison", "verified", "evidence")):
        score += 0.06
    return round(min(score, 0.97), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Base RLM agent
# ──────────────────────────────────────────────────────────────────────────────
class RLMAgent:
    """
    Stateful RLM agent with recursive context, tool access, and adaptive
    complexity routing.  All LLM calls go through llm_provider.llm_complete,
    so no provider-specific code lives here.
    """

    def __init__(
        self,
        name: str,
        role: str,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.name          = name
        self.role          = role
        self.allowed_tools = allowed_tools
        self.tools         = self._build_tool_list()

        self._tool_calls:     int   = 0
        self._steps_taken:    int   = 0
        self._complexity:     str   = "medium"
        self._max_tool_calls: int   = 2
        self._max_tool_rounds: int  = 2
        self._call_delay:     float = 0.08

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_tool_list(self) -> list[dict]:
        all_tools = MCPRegistry.schemas()
        if not self.allowed_tools:
            return all_tools
        return [t for t in all_tools if t["function"]["name"] in self.allowed_tools]

    def _apply_complexity_policy(self, complexity: str) -> None:
        self._complexity       = complexity
        self._max_tool_calls   = _budget(complexity, "calls")
        self._max_tool_rounds  = _budget(complexity, "rounds")
        self._call_delay       = _budget(complexity, "delay")

    def _reset_counters(self) -> None:
        self._tool_calls  = 0
        self._steps_taken = 0

    # ── LLM calls ─────────────────────────────────────────────────────────────

    async def _throttle_global(self) -> None:
        global _GLOBAL_LAST_CALL_TS

        async with _GLOBAL_LLM_LOCK:
            now  = time.monotonic()
            wait = _GLOBAL_MIN_GAP_S - (now - _GLOBAL_LAST_CALL_TS)
            if wait > 0:
                await asyncio.sleep(wait)
            _GLOBAL_LAST_CALL_TS = time.monotonic()

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool = True,
        attempt:   int  = 0,
    ) -> Any | None:
        """
        Call the provider via llm_complete (runs in a thread so it doesn't
        block the event loop).  Returns an LLMMessage or None on failure.
        """
        if self._call_delay > 0:
            await asyncio.sleep(self._call_delay)

        await self._throttle_global()

        tools_payload: list[dict] | None = None
        if use_tools and self.tools and self._tool_calls < self._max_tool_calls:
            # Pass OpenAI-style tool schemas; each provider translates them
            tools_payload = self.tools

        try:
            msg = await asyncio.to_thread(
                llm_complete,
                MODEL_A,
                messages,
                max_tokens=RLM_MAX_TOKENS,
                temperature=0.25,
                tools=tools_payload,
                system=self.role,
            )
            return msg

        except Exception as exc:
            error = str(exc)

            if ("quota" in error.lower() or "rate" in error.lower()) and attempt < MAX_RETRIES:
                wait = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
                print(
                    f"    [{self.name}] Rate-limit — backing off {wait:.1f}s "
                    f"(attempt {attempt + 1})"
                )
                await asyncio.sleep(wait)
                return await self._call_llm(
                    messages,
                    use_tools=use_tools,
                    attempt=attempt + 1,
                )

            print(f"    [{self.name}] LLM error: {error[:180]}")
            return None

    # ── Tool handling ─────────────────────────────────────────────────────────

    async def _handle_tools(self, msg: Any, ctx: RecursiveContext) -> Any | None:
        """
        Execute tool calls, push results into context, and re-query the model.
        msg is an LLMMessage; .tool_calls is a list[_ToolCall] or None.
        """
        rounds = 0

        while getattr(msg, "tool_calls", None) and rounds < self._max_tool_rounds:
            tool_results_text: list[str] = []

            for tc in msg.tool_calls:
                if self._tool_calls >= self._max_tool_calls:
                    break

                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                print(f"    [{self.name}] MCP -> {tc.function.name}({list(args.keys())})")
                output = _normalise(MCPRegistry.call(tc.function.name, args, agent=self.name))
                self._tool_calls += 1

                if tc.function.name == "search_web" and output and "No evidence" not in output:
                    MCPRegistry.call(
                        "memory_store",
                        {"key": f"{self.name}_ev_{self._tool_calls}", "value": output[:1000]},
                        agent=self.name,
                    )

                ctx.push("tool_result", output[:1000], {"tool": tc.function.name})
                tool_results_text.append(f"[Tool:{tc.function.name}] {output[:600]}")

            updated_messages = ctx.to_messages() + [
                {"role": "user", "content": "\n\n".join(tool_results_text)},
            ]

            msg = await self._call_llm(updated_messages, use_tools=True)
            if msg is None:
                break

            rounds += 1

        return msg

    # ── Critique / verification ───────────────────────────────────────────────

    async def _critique(self, ctx: RecursiveContext) -> str:
        msgs = ctx.to_messages() + [{
            "role":    "user",
            "content": (
                "Act as a critical reviewer. Identify:\n"
                "- Logical gaps or unsupported jumps.\n"
                "- Missing definitions or examples.\n"
                "- Contradictions in the reasoning.\n"
                "- Whether depth matches task complexity.\n"
                "Return a short structured critique."
            ),
        }]
        msg = await self._call_llm(msgs, use_tools=False)
        return _normalise(getattr(msg, "content", "")) or "Critique unavailable."

    async def _verify(self, ctx: RecursiveContext, instruction: str) -> str:
        msgs = ctx.to_messages() + [{
            "role":    "user",
            "content": (
                "Critically verify all reasoning so far.\n"
                "1. Which parts of the original task are fully addressed?\n"
                "2. Which claims are weakly supported or speculative?\n"
                "3. What tool evidence exists and how strong is it?\n"
                "4. What is still missing or incomplete?\n"
                "5. What must the final synthesis include to be complete?\n"
                "Be precise and concise."
            ),
        }]
        msg = await self._call_llm(msgs, use_tools=False)
        return _normalise(getattr(msg, "content", "")) or "Verification unavailable."

    # ── Core reasoning loop ───────────────────────────────────────────────────

    async def recursive_think(
        self,
        instruction: str,
        *,
        inherited_ctx:  RecursiveContext | None = None,
        research_style: bool                    = False,
        complexity:     str | None              = None,
    ) -> dict[str, Any]:
        self._reset_counters()
        complexity = complexity or detect_complexity(instruction)
        self._apply_complexity_policy(complexity)

        max_depth   = _budget(complexity, "depth")
        total_steps = _budget(complexity, "steps")

        ctx = RecursiveContext(agent=self.name, max_depth=max_depth)

        if inherited_ctx:
            for layer in inherited_ctx.stack[-10:]:
                content = _normalise(layer.content)
                if content:
                    ctx.push("inherited", content[:600], {"from": inherited_ctx.agent})

        ctx.push("input", instruction, {"complexity": complexity})

        require_grounding = (complexity in ("heavy", "multi-hop")) or research_style
        run_verification  = complexity != "simple"
        run_critique      = complexity in ("heavy", "multi-hop")

        for step in range(total_steps):
            if ctx.is_done():
                break

            directive = self._build_directive(step, total_steps, complexity, research_style)
            messages  = ctx.to_messages() + [{"role": "user", "content": directive}]

            can_use_tools = (
                self._max_tool_calls > 0 and self._tool_calls < self._max_tool_calls
            )

            msg = await self._call_llm(messages, use_tools=can_use_tools)

            if msg is None:
                ctx.push(
                    "reasoning",
                    "Recovered from API failure — continuing with partial context.",
                    {"step": step, "error": "api_failure"},
                )
                continue

            if getattr(msg, "tool_calls", None) and can_use_tools:
                msg = await self._handle_tools(msg, ctx)
                if msg is None:
                    ctx.push(
                        "reasoning",
                        "Tool-assisted reasoning was interrupted by API limits.",
                        {"step": step, "error": "tool_phase_interrupted"},
                    )
                    continue

            reasoning = _normalise(getattr(msg, "content", ""))
            if reasoning:
                ctx.push("reasoning", reasoning, {"step": step})
                self._steps_taken += 1

            if complexity == "simple" and self._steps_taken >= 1 and len(reasoning) > 80:
                break

            if step > 0 and reasoning.strip().startswith("FINAL:"):
                print(f"    [{self.name}] FINAL signal at step {step + 1}")
                break

        if require_grounding and not self._has_grounding(ctx):
            ctx.push(
                "critique",
                "WARNING: No tool grounding found. Distinguish verified claims from inference.",
                {"auto": True},
            )

        if run_critique and run_verification:
            critique, verification = await asyncio.gather(
                self._critique(ctx),
                self._verify(ctx, instruction),
            )
            ctx.push("critique",      critique,      {"phase": "pre-verification"})
            ctx.push("verification",  verification,  {"phase": "pre-synthesis"})
        elif run_critique:
            critique = await self._critique(ctx)
            ctx.push("critique", critique, {"phase": "pre-verification"})
        elif run_verification:
            verification = await self._verify(ctx, instruction)
            ctx.push("verification", verification, {"phase": "pre-synthesis"})

        final_msgs = ctx.to_messages() + [{
            "role":    "user",
            "content": (
                "Write the complete, final answer to the original task.\n\n"
                "Requirements:\n"
                "1. Address every part of the task directly and explicitly.\n"
                "2. Use clear ## section headings.\n"
                "3. Distinguish verified facts from inference.\n"
                "4. Include examples, code, or comparisons where relevant.\n"
                "5. End with a concise ## Conclusion.\n"
                "Produce a rich, structured, academically rigorous response."
            ),
        }]

        final_msg = await self._call_llm(final_msgs, use_tools=False)

        if final_msg and getattr(final_msg, "content", None):
            answer = _normalise(final_msg.content)
        else:
            recent_items = [
                l.content[:200]
                for l in ctx.stack[-6:]
                if getattr(l, "content", None)
            ]
            answer = (
                "## Partial Result (API Constraint)\n\n"
                "Due to API limits, full synthesis could not be completed.\n\n"
                "## Extracted Reasoning\n\n"
                + ("\n".join(f"- {item}" for item in recent_items)
                   if recent_items else "- No intermediate reasoning captured.")
                + "\n\n## Conclusion\n"
                  "The reasoning above represents partial progress toward the final answer."
            )

        ctx.push("synthesis", answer, {"final": True})
        print(f"    [{self.name}] ctx stats: {ctx.get_stats()}")

        return {
            "result":     answer,
            "ctx":        ctx,
            "tool_calls": self._tool_calls,
            "steps":      self._steps_taken,
            "ctx_depth":  ctx.depth,
            "complexity": complexity,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _has_grounding(self, ctx: RecursiveContext) -> bool:
        return any(l.role in ("tool_result", "summary") for l in ctx.stack)

    @staticmethod
    def _build_directive(
        step:           int,
        total_steps:    int,
        complexity:     str,
        research_style: bool,
    ) -> str:
        if step == 0:
            if complexity == "simple":
                directive = (
                    "Answer this directly and efficiently. "
                    "Focus on the key distinctions. Be clear and structured."
                )
            else:
                directive = (
                    "Deeply decompose this task. "
                    "Identify what must be researched, compared, or verified. "
                    "Use tools where evidence would strengthen the answer. "
                    "Do NOT produce the final answer yet."
                )
        elif step == total_steps - 1:
            directive = (
                f"[Final reasoning step {step + 1}/{total_steps}] "
                "Consolidate all prior context. Remove redundancy. "
                "Ensure all sub-questions are addressed. "
                "Start with 'FINAL:' to signal completion."
            )
        else:
            directive = (
                f"[Step {step + 1}/{total_steps}] "
                "Build on prior context. Add NEW value only — do not repeat. "
                "Deepen analysis, add missing comparisons, or verify uncertain claims. "
                "Use tools if additional grounding is needed. "
                "Start with 'FINAL:' only when the task is fully addressed."
            )

        if research_style and complexity != "simple":
            directive += (
                " Structure output with clear headings: "
                "## Background | ## Analysis | ## Findings | ## Comparison | ## Conclusion"
            )

        return directive


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — RLM-A
# ──────────────────────────────────────────────────────────────────────────────
class RLMOrchestrator(RLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "RLM-A",
            role = (
                "You are an expert recursive orchestrator. "
                "Decompose hard tasks into exactly 2 specialist subtasks, "
                "delegate via A2A, then synthesise specialist outputs into a "
                "complete, evidence-backed, well-structured final answer. "
                "For planning, return ONLY a valid JSON array of 2 subtask strings."
            ),
            allowed_tools=["search_web", "memory_store", "memory_list"],
        )
        self.bus = bus

    async def _plan_subtasks(self, query: str, complexity: str) -> list[str]:
        hints = {
            "multi-hop": (
                "Subtask 1 = Map the complete workflow, stages, and context flow.\n"
                "Subtask 2 = Compare LLM vs RLM failure modes, mitigations, and performance.\n"
            ),
            "heavy": (
                "Subtask 1 = Concept extraction, limitations, context flow, and deep analysis.\n"
                "Subtask 2 = Structured comparison, evidence synthesis, and mitigation design.\n"
            ),
        }

        if _needs_code(query):
            hint = (
                "Subtask 1 = Theory, research, academic background, and trade-off analysis.\n"
                "Subtask 2 = Implementation, code examples, benchmarks, and system design.\n"
            )
        else:
            hint = hints.get(complexity, (
                "Subtask 1 = Core theory and detailed explanation.\n"
                "Subtask 2 = Practical comparison, examples, and conclusion.\n"
            ))

        msg = await self._call_llm([{
            "role":    "user",
            "content": (
                f"Query: {query}\n\n"
                "Return ONLY a valid JSON array of exactly 2 subtask strings. "
                "No explanation. Just the JSON array.\n"
                f"{hint}"
                'Format: ["subtask one here", "subtask two here"]'
            ),
        }], use_tools=False)

        plan_raw = _normalise(getattr(msg, "content", "")) if msg else ""
        print(f"  [RLM-A] Raw plan: {plan_raw[:160]}...")
        subtasks = _extract_subtasks(plan_raw, query)

        for i, s in enumerate(subtasks, 1):
            print(f"    {i}. {s[:100]}...")

        return subtasks

    async def _delegate(
        self,
        receiver: str,
        subtask:  str,
        snap:     dict,
        timeout:  float,
    ) -> dict:
        task = A2ATask(
            sender   = self.name,
            receiver = receiver,
            payload  = {"instruction": subtask, "ctx_snapshot": snap},
        )
        print(f"\n  [A2A+CTX] {self.name} -> {receiver}")
        print(f"            {subtask[:90]}...")
        print(f"            ctx_depth={snap.get('depth', 0)}  timeout={timeout}s")
        return await self.bus.send(task, timeout=timeout)

    async def run(self, query: str) -> dict[str, Any]:
        self._reset_counters()
        t0 = time.monotonic()

        complexity       = detect_complexity(query)
        delegate_timeout = float(_budget(complexity, "timeout"))
        self._apply_complexity_policy(complexity)

        print(f"\n{'=' * 72}")
        print(f"  RLM-A  [{complexity.upper()}]  {query[:66]}...")
        print(f"{'=' * 72}")

        if complexity == "simple":
            out   = await self.recursive_think(
                instruction    = query,
                inherited_ctx  = None,
                research_style = False,
                complexity     = complexity,
            )
            final = _normalise(out["result"])
            return {
                "answer":     final,
                "time_s":     round(time.monotonic() - t0, 2),
                "tool_calls": out["tool_calls"],
                "ctx_depth":  out["ctx_depth"],
                "steps":      out["steps"],
                "confidence": _heuristic_confidence(final, out["tool_calls"], out["steps"]),
                "subtasks":   [],
            }

        ctx      = RecursiveContext(agent=self.name, max_depth=_budget(complexity, "depth"))
        subtasks = await self._plan_subtasks(query, complexity)

        ctx.push("input", query,              {"phase": "orchestrator"})
        ctx.push("plan",  f"Subtask-1: {subtasks[0]}", {"phase": "planning"})
        ctx.push("plan",  f"Subtask-2: {subtasks[1]}", {"phase": "planning"})
        snap = ctx.snapshot()

        r_b = await self._delegate("RLM-B", subtasks[0], snap, delegate_timeout)
        r_c = await self._delegate("RLM-C", subtasks[1], snap, delegate_timeout)

        memory_evidence = await asyncio.to_thread(MCPRegistry.memory_list)

        synth_instruction = (
            f"## Original Query\n{query}\n\n"
            f"## Research Findings (Agent B)\n"
            f"{_normalise(r_b.get('result', 'No result'))[:1000]}\n\n"
            f"## Implementation Findings (Agent C)\n"
            f"{_normalise(r_c.get('result', 'No result'))[:1000]}\n\n"
            f"## Shared Memory Evidence\n{_normalise(memory_evidence)[:600]}\n\n"
            "## Your Task\n"
            "Synthesise the above into one complete, evidence-backed final answer. "
            "Use ## headings. Include comparison tables if helpful. "
            "End with ## Conclusion."
        )

        print(f"\n  [RLM-A] Beginning synthesis pass...")
        synth = await self.recursive_think(
            instruction    = synth_instruction,
            inherited_ctx  = None,
            research_style = True,
            complexity     = complexity,
        )

        final = _normalise(synth["result"])
        if len(final) < FINAL_MIN_LEN:
            final = (
                f"## Research (Agent B)\n{_normalise(r_b.get('result', ''))}\n\n"
                f"## Implementation (Agent C)\n{_normalise(r_c.get('result', ''))}"
            )

        total_tool_calls = (
            self._tool_calls
            + r_b.get("tool_calls", 0)
            + r_c.get("tool_calls", 0)
            + synth.get("tool_calls", 0)
        )
        total_steps = (
            self._steps_taken
            + r_b.get("steps", 0)
            + r_c.get("steps", 0)
            + synth.get("steps", 0)
        )
        total_ctx_depth = (
            r_b.get("ctx_depth", 0)
            + r_c.get("ctx_depth", 0)
            + synth.get("ctx_depth", 0)
        )

        elapsed = round(time.monotonic() - t0, 2)
        print(f"\n  [RLM-A] Done — time={elapsed}s  tools={total_tool_calls}  steps={total_steps}")

        return {
            "answer":     final,
            "time_s":     elapsed,
            "tool_calls": total_tool_calls,
            "ctx_depth":  total_ctx_depth,
            "steps":      total_steps,
            "confidence": _heuristic_confidence(final, total_tool_calls, total_steps),
            "subtasks":   subtasks,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Specialist B — Research & Theory
# ──────────────────────────────────────────────────────────────────────────────
class RLMSpecialistB(RLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "RLM-B",
            role = (
                "You are a recursive research specialist with access to web search and memory tools. "
                "You receive an inherited context from the orchestrator. "
                "Go deeper, not wider. Add evidence, analysis, distinctions, and theoretical "
                "grounding that is NOT already in the inherited context. "
                "Always cite evidence from tool results when available. "
                "Produce structured, academically rigorous output."
            ),
            allowed_tools=["search_web", "memory_store", "memory_search", "memory_get"],
        )
        self.bus = bus

    async def run(self) -> None:
        print(f"  [{self.name}] Listening on bus...")
        while True:
            task = await self.bus.listen("RLM-B")
            snap = task.payload.get("ctx_snapshot", {})

            inherited = (
                RecursiveContext.from_snapshot(snap, new_agent=f"{self.name}-inherited")
                if snap.get("stack") else None
            )

            instruction = _normalise(task.payload.get("instruction", ""))
            complexity  = detect_complexity(instruction)
            print(f"\n  [{self.name}] Task | inh_depth={inherited.depth if inherited else 0} | {complexity}")

            out = await self.recursive_think(
                instruction    = instruction,
                inherited_ctx  = inherited,
                research_style = (complexity != "simple"),
                complexity     = complexity,
            )

            self.bus.respond(task, {
                "result":     out["result"],
                "tool_calls": out["tool_calls"],
                "steps":      out["steps"],
                "ctx_depth":  out["ctx_depth"],
            })


# ──────────────────────────────────────────────────────────────────────────────
# Specialist C — Implementation & Systems
# ──────────────────────────────────────────────────────────────────────────────
class RLMSpecialistC(RLMAgent):
    def __init__(self, bus: A2ABus) -> None:
        super().__init__(
            name = "RLM-C",
            role = (
                "You are a recursive implementation and systems specialist. "
                "You receive an inherited context from the orchestrator. "
                "Build on it — do not repeat it. "
                "Provide: concrete code examples, workflow diagrams in text, "
                "benchmark reasoning, system failure analysis, and actionable recommendations. "
                "When the task involves comparison, produce a structured comparison table. "
                "When the task involves code, write clean, runnable Python."
            ),
            allowed_tools=["search_web", "memory_store", "memory_search", "memory_get"],
        )
        self.bus = bus

    async def run(self) -> None:
        print(f"  [{self.name}] Listening on bus...")
        while True:
            task = await self.bus.listen("RLM-C")
            snap = task.payload.get("ctx_snapshot", {})

            inherited = (
                RecursiveContext.from_snapshot(snap, new_agent=f"{self.name}-inherited")
                if snap.get("stack") else None
            )

            instruction = _normalise(task.payload.get("instruction", ""))
            complexity  = detect_complexity(instruction)
            print(f"\n  [{self.name}] Task | inh_depth={inherited.depth if inherited else 0} | {complexity}")

            out = await self.recursive_think(
                instruction    = instruction,
                inherited_ctx  = inherited,
                research_style = (complexity in ("heavy", "multi-hop")),
                complexity     = complexity,
            )

            self.bus.respond(task, {
                "result":     out["result"],
                "tool_calls": out["tool_calls"],
                "steps":      out["steps"],
                "ctx_depth":  out["ctx_depth"],
            })


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
async def run_rlm_system(query: str) -> dict[str, Any]:
    bus  = A2ABus()
    orch = RLMOrchestrator(bus)
    sp_b = RLMSpecialistB(bus)
    sp_c = RLMSpecialistC(bus)

    for name in ("RLM-A", "RLM-B", "RLM-C"):
        bus.register(name)

    t_b = asyncio.create_task(sp_b.run())
    t_c = asyncio.create_task(sp_c.run())

    try:
        return await orch.run(query)
    finally:
        t_b.cancel()
        t_c.cancel()


if __name__ == "__main__":
    _query = (
        "Analyse recursive vs iterative dynamic programming. "
        "Explain theory, strengths, weaknesses, and practical use cases. "
        "Write Python benchmark code comparing both on Fibonacci."
    )
    _result = asyncio.run(run_rlm_system(_query))

    print("\n" + "=" * 72)
    print("RLM FINAL ANSWER")
    print("=" * 72)
    print(_result["answer"])