"""
a2a_bus.py — Agent-to-Agent (A2A) Message Bus
===============================================
An asyncio-native, in-process message bus for multi-agent coordination.

Design:
  - Each registered agent owns a private asyncio.Queue inbox.
  - The sender awaits completion via an asyncio.Event on the task object,
    eliminating polling and reducing unnecessary context switches.
  - Tasks carry a full audit trail (status, timestamps, sender/receiver).
  - Bus-level stats are available for post-benchmark reporting.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class A2ATask:
    """
    A unit of work passed between agents.

    ``_event`` is set by ``A2ABus.respond``; the sender unblocks as soon as
    the receiver calls respond, with no polling required.
    """

    task_id:      str              = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender:       str              = ""
    receiver:     str              = ""
    payload:      dict[str, Any]   = field(default_factory=dict)

    response:     dict | None      = None
    status:       str              = "pending"   # pending | running | done | failed
    created_at:   float            = field(default_factory=time.monotonic)
    completed_at: float | None     = None

    # Internal completion signal — not part of the public interface
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def latency(self) -> float | None:
        """Round-trip latency in seconds, or None if not yet completed."""
        if self.completed_at is None:
            return None
        return round(self.completed_at - self.created_at, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Bus
# ──────────────────────────────────────────────────────────────────────────────
class A2ABus:
    """
    Asynchronous in-process message bus.

    Usage::

        bus = A2ABus()
        bus.register("agent-a")
        bus.register("agent-b")

        # sender:
        result = await bus.send(A2ATask(sender="agent-a", receiver="agent-b",
                                        payload={"msg": "hello"}))

        # receiver (runs as a background task):
        task = await bus.listen("agent-b")
        bus.respond(task, {"reply": "world"})
    """

    def __init__(self) -> None:
        self._inbox:    dict[str, asyncio.Queue[A2ATask]] = {}
        self._task_log: list[dict[str, Any]]              = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str) -> None:
        if name in self._inbox:
            raise ValueError(f"Agent '{name}' is already registered on this bus.")
        self._inbox[name] = asyncio.Queue()

    def is_registered(self, name: str) -> bool:
        return name in self._inbox

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, task: A2ATask, timeout: float = 15.0) -> dict[str, Any]:
        """
        Dispatch ``task`` to the receiver's inbox and await completion.

        Returns the response dict on success, or an error dict on timeout or
        unregistered receiver.
        """
        if task.receiver not in self._inbox:
            return {"error": f"Receiver '{task.receiver}' is not registered.", "task_id": task.task_id}

        task.status = "pending"
        self._log(task, "sent")
        await self._inbox[task.receiver].put(task)

        try:
            await asyncio.wait_for(task._event.wait(), timeout=timeout)
            self._log(task, "completed")
            return task.response or {"error": "Receiver returned no response.", "task_id": task.task_id}

        except asyncio.TimeoutError:
            task.status = "failed"
            self._log(task, "timeout")
            return {"error": f"Task timed out after {timeout:.1f}s.", "task_id": task.task_id}

    # ── Receive ───────────────────────────────────────────────────────────────

    async def listen(self, name: str) -> A2ATask:
        """Block until a task arrives in ``name``'s inbox."""
        if name not in self._inbox:
            raise ValueError(f"Agent '{name}' is not registered on this bus.")

        task        = await self._inbox[name].get()
        task.status = "running"
        self._log(task, "received")
        return task

    # ── Respond ───────────────────────────────────────────────────────────────

    def respond(self, task: A2ATask, response: dict[str, Any]) -> None:
        """Complete a task and unblock the sender's ``await send(...)``."""
        task.response     = response
        task.status       = "done"
        task.completed_at = time.monotonic()
        task._event.set()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        total     = len(self._task_log)
        completed = sum(1 for e in self._task_log if e["event"] == "completed")
        timeouts  = sum(1 for e in self._task_log if e["event"] == "timeout")

        by_agent: dict[str, int] = {}
        for e in self._task_log:
            receiver = e["receiver"]
            by_agent[receiver] = by_agent.get(receiver, 0) + 1

        return {
            "total_events":    total,
            "completed":       completed,
            "timeouts":        timeouts,
            "events_by_agent": by_agent,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log(self, task: A2ATask, event: str) -> None:
        self._task_log.append({
            "task_id":   task.task_id,
            "event":     event,
            "sender":    task.sender,
            "receiver":  task.receiver,
            "status":    task.status,
            "timestamp": time.monotonic(),
        })