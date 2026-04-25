"""
rlm_context.py — Recursive Context Stack
==========================================
Maintains a typed, depth-bounded stack of reasoning layers for a single agent.
Key responsibilities:
  - Push typed layers (input, plan, reasoning, tool_result, critique, …)
  - Compress the stack when depth approaches max_depth, preserving high-signal
    layers and summarising the discarded middle.
  - Serialise to OpenAI-format messages for LLM API calls.
  - Snapshot / restore for A2A context inheritance between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Role literals that represent layers added by the agent itself
_AGENT_ROLES = frozenset({
    "plan", "reasoning", "summary", "critique", "verification", "synthesis"
})

# Roles considered high-signal enough to always survive compression
_KEEP_ROLES = frozenset({
    "input", "inherited", "plan", "summary",
    "tool_result", "reasoning", "critique", "verification",
})

# Per-role content length limits when converting to messages
_TRIM_LIMITS: dict[str, int] = {
    "tool_result":  1_200,
    "reasoning":    1_000,
    "summary":      1_000,
    "critique":     1_000,
    "verification": 1_000,
    "plan":         1_000,
    "input":        1_600,
    "inherited":    1_600,
}
_DEFAULT_TRIM = 900


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CtxLayer:
    step:    int
    role:    str
    content: str
    meta:    dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Recursive context
# ──────────────────────────────────────────────────────────────────────────────
class RecursiveContext:
    """
    Depth-bounded context stack for a single RLM agent.

    Parameters
    ----------
    agent:     Human-readable agent name for logging.
    max_depth: Maximum reasoning steps before ``is_done`` returns True.
    """

    def __init__(self, agent: str, max_depth: int = 10) -> None:
        self.agent      = agent
        self.max_depth  = max_depth
        # Start compressing when ~70 % of budget is used
        self._compress_at = max(3, int(max_depth * 0.7))
        self.stack:  list[CtxLayer] = []
        self.step:   int            = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, role: str, content: str, meta: dict | None = None) -> None:
        """Append a new layer; silently drops empty content."""
        text = "" if content is None else str(content).strip()
        if not text:
            return

        if self.step >= self._compress_at and len(self.stack) > 8:
            self._compress()

        self.stack.append(CtxLayer(step=self.step, role=role, content=text, meta=meta or {}))
        self.step += 1

        print(f"    [CTX:{self.agent}] {self.step:02d} | {role:<12} | {text[:80]}...")

    def to_messages(self) -> list[dict[str, str]]:
        """Serialise stack to OpenAI-format messages for an LLM API call."""
        messages: list[dict[str, str]] = []

        for layer in self.stack:
            content = self._trim(layer)
            if not content:
                continue

            if layer.role in ("input", "inherited"):
                messages.append({"role": "user", "content": content})

            elif layer.role == "tool_result":
                tool = layer.meta.get("tool", "?")
                messages.append({"role": "user", "content": f"[Tool:{tool}] {content}"})

            elif layer.role in _AGENT_ROLES:
                messages.append({"role": "assistant", "content": content})

            else:
                messages.append({"role": "user", "content": content})

        return messages

    def snapshot(self) -> dict:
        """
        Create a compact, serialisable representation for A2A context passing.
        Keeps only high-signal roles; limits recent reasoning layers to 2.
        """
        selected = [l for l in self.stack if l.role in _KEEP_ROLES]

        compact:       list[CtxLayer] = []
        reasoning_seen = 0

        for layer in reversed(selected):
            if layer.role == "reasoning":
                if reasoning_seen >= 2:
                    continue
                reasoning_seen += 1
            compact.append(layer)

        compact.sort(key=lambda x: x.step)

        return {
            "agent": self.agent,
            "depth": self.step,
            "stack": [
                {
                    "step":    l.step,
                    "role":    l.role,
                    "content": l.content[:900] if l.role in ("input", "inherited") else l.content[:700],
                    "meta":    l.meta,
                }
                for l in compact
            ],
        }

    @classmethod
    def from_snapshot(cls, snap: dict, new_agent: str) -> "RecursiveContext":
        """Restore a context from a snapshot produced by ``snapshot()``."""
        ctx = cls(agent=new_agent)
        for item in snap.get("stack", []):
            ctx.stack.append(CtxLayer(
                step    = item.get("step",    len(ctx.stack)),
                role    = item.get("role",    "input"),
                content = item.get("content", ""),
                meta    = item.get("meta",    {}),
            ))
        ctx.step = snap.get("depth", len(ctx.stack))
        return ctx

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_relevant(self, keyword: str) -> list[CtxLayer]:
        """Return layers whose content contains ``keyword`` (case-insensitive)."""
        kw = keyword.lower().strip()
        return [l for l in self.stack if kw in l.content.lower()] if kw else []

    def reasoning_depth(self) -> int:
        """Count how many reasoning steps have been recorded."""
        return sum(1 for l in self.stack if l.role == "reasoning")

    def get_stats(self) -> dict:
        counts: dict[str, int] = {}
        for l in self.stack:
            counts[l.role] = counts.get(l.role, 0) + 1
        return {
            "total_depth":      self.step,
            "role_breakdown":   counts,
            "reasoning_steps":  self.reasoning_depth(),
            "has_compression": any(
                isinstance(l.meta, dict) and l.meta.get("compressed")
                for l in self.stack
            ),
        }

    @property
    def depth(self) -> int:
        return self.step

    def is_done(self) -> bool:
        return self.reasoning_depth() >= self.max_depth

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compress(self) -> None:
        """
        Discard low-priority middle layers, replacing them with a bullet-point
        summary. High-signal layers from ``_KEEP_ROLES`` are always retained.
        """
        if len(self.stack) < 8:
            return

        keep = (
            [l for l in self.stack if l.role == "input"][:1]
            + [l for l in self.stack if l.role == "inherited"][:2]
            + [l for l in self.stack if l.role == "plan"][:2]
            + [l for l in self.stack if l.role == "summary"][-1:]
            + [l for l in self.stack if l.role == "tool_result"][-3:]
            + [l for l in self.stack if l.role == "reasoning"][-3:]
            + [l for l in self.stack if l.role in ("critique", "verification")][-2:]
        )

        keep_ids = {id(l) for l in keep}
        dropped  = [l for l in self.stack if id(l) not in keep_ids]

        bullets = [
            f"- {l.role}: {l.content.replace(chr(10), ' ').strip()[:120]}"
            for l in dropped[-10:]
        ]

        new_stack = list(keep)
        if bullets:
            new_stack.append(CtxLayer(
                step    = self.step,
                role    = "summary",
                content = "Compressed context summary:\n" + "\n".join(bullets),
                meta    = {"compressed": True},
            ))

        new_stack.sort(key=lambda x: x.step)
        self.stack = new_stack
        print(f"    [CTX:{self.agent}] ⚠ Context compressed ({len(dropped)} layers dropped)")

    def _trim(self, layer: CtxLayer) -> str:
        """Apply per-role content length limits."""
        text  = layer.content.strip()
        limit = _TRIM_LIMITS.get(layer.role, _DEFAULT_TRIM)
        if len(text) > limit:
            return text[:limit] + ("...[trimmed]" if layer.role == "tool_result" else "")
        return text