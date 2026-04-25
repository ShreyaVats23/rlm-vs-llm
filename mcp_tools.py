from __future__ import annotations

import json
import time
import hashlib
import threading
import urllib.parse
import urllib.request
from typing import Any
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CACHE_TTL = 600
MAX_MEMORY = 200
MAX_RESULT_LEN = 1500

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
_MCP_MEMORY: dict[str, str] = {}
_MCP_LOG: list[dict] = []
_CACHE: dict[str, dict] = {}

_LOCK = threading.RLock()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _now():
    return datetime.now(timezone.utc).isoformat()

def _truncate(x: str, n: int = MAX_RESULT_LEN):
    return x if len(x) <= n else x[:n] + "...[truncated]"

def _hash(x: str):
    return hashlib.md5(x.lower().encode()).hexdigest()

def _cache_get(key: str):
    with _LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > CACHE_TTL:
            del _CACHE[key]
            return None
        return entry["value"]

def _cache_set(key: str, value: str):
    with _LOCK:
        _CACHE[key] = {"value": value, "ts": time.time()}

def _log(agent, tool, args, result, latency, cache=False):
    with _LOCK:
        _MCP_LOG.append({
            "ts": _now(),
            "agent": agent,
            "tool": tool,
            "latency": round(latency, 3),
            "cache": cache,
        })

def reset_logs() -> None:
    with _LOCK:
        _MCP_LOG.clear()

def get_logs() -> list[dict]:
    with _LOCK:
        return list(_MCP_LOG)

# ─────────────────────────────────────────────
# CORE MCP
# ─────────────────────────────────────────────
class MCPRegistry:

    @staticmethod
    def call(name: str, args: dict, agent="unknown"):
        if name == "search_web":
            return MCPRegistry.search_web(args.get("query", ""), agent)
        if name == "memory_store":
            return MCPRegistry.memory_store(args.get("key"), args.get("value"))
        if name == "memory_get":
            return MCPRegistry.memory_get(args.get("key"))
        if name == "memory_search":
            return MCPRegistry.memory_search(args.get("query"))
        if name == "memory_list":
            return MCPRegistry.memory_list()
        return "Unknown tool"

    @staticmethod
    def schemas():
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information on a given query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_store",
                    "description": "Store a key-value pair in shared agent memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key":   {"type": "string", "description": "Memory key."},
                            "value": {"type": "string", "description": "Content to store."},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_get",
                    "description": "Retrieve a value from shared agent memory by key.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Memory key to retrieve."}
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search shared agent memory for entries matching a query string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search term to match against keys and values."}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_list",
                    "description": "List all entries currently stored in shared agent memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    @staticmethod
    def search_web(query: str, agent="unknown"):
        t0 = time.time()
        key = "search:" + _hash(query)

        cached = _cache_get(key)
        if cached:
            _log(agent, "search_web", {}, cached, time.time() - t0, True)
            return cached

        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            with urllib.request.urlopen(url, timeout=4) as r:
                data = json.loads(r.read())

            text = data.get("AbstractText") or "No result"

            result = f"""
Query: {query}

Summary:
{text}

Use this as evidence. Verify uncertain claims.
"""

        except Exception:
            result = "Search failed"

        result = _truncate(result)
        _cache_set(key, result)
        _log(agent, "search_web", {}, result, time.time() - t0)

        return result

    @staticmethod
    def memory_store(key: str, value: str):
        if not key:
            return "Error: key required"

        with _LOCK:
            if len(_MCP_MEMORY) > MAX_MEMORY:
                _MCP_MEMORY.pop(next(iter(_MCP_MEMORY)))

            _MCP_MEMORY[key] = _truncate(value, 800)

        return "stored"

    @staticmethod
    def memory_get(key: str):
        return _MCP_MEMORY.get(key, "Not found")

    @staticmethod
    def memory_search(query: str):
        q = (query or "").lower()
        results = {
            k: v for k, v in _MCP_MEMORY.items()
            if q in k.lower() or q in v.lower()
        }
        return json.dumps(results, indent=2) if results else "No matches"

    @staticmethod
    def memory_list():
        return json.dumps(_MCP_MEMORY, indent=2)