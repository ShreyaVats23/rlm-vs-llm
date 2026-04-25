"""
llm_provider.py — Unified LLM Provider Abstraction
====================================================
Drop-in shim that presents a single interface regardless of which
API backend is active.  Switch providers by editing .env only:

    API_PROVIDER=groq          # groq | gemini | openai
    API_KEY=your_key_here
    # MODEL is read from config.py — set it there per-provider.

Returned message objects always expose:
    .content     str   — text of the response
    .tool_calls  list | None — list of _ToolCall objects (or None)

Each _ToolCall exposes:
    .id                str
    .function.name     str
    .function.arguments  str  (JSON)
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_PROVIDER = os.environ.get("API_PROVIDER", "groq").lower().strip()
_API_KEY  = os.environ.get("API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────────
# Shared compat types
# ──────────────────────────────────────────────────────────────────────────────

class _FunctionInfo:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str) -> None:
        self.name      = name
        self.arguments = arguments


class _ToolCall:
    __slots__ = ("id", "function")

    def __init__(self, id_: str, name: str, arguments: str) -> None:
        self.id       = id_
        self.function = _FunctionInfo(name, arguments)


class LLMMessage:
    """Normalised response returned by every provider."""

    __slots__ = ("content", "tool_calls")

    def __init__(self, content: str, tool_calls: list[_ToolCall] | None) -> None:
        self.content    = content
        self.tool_calls = tool_calls or None   # keep None when empty


# ──────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ──────────────────────────────────────────────────────────────────────────────

class _GroqProvider:
    """Wraps groq.Groq with the common interface."""

    def __init__(self) -> None:
        from groq import Groq  # type: ignore
        self._client = Groq(api_key=_API_KEY)

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMMessage:
        # Groq doesn't accept a 'system' message-role in the array for all
        # models, but injecting it as the first message works fine.
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            # Groq expects OpenAI-style tool dicts
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        msg  = resp.choices[0].message

        content    = msg.content or ""
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                _ToolCall(
                    id_       = tc.id,
                    name      = tc.function.name,
                    arguments = tc.function.arguments or "{}",
                )
                for tc in msg.tool_calls
            ]

        return LLMMessage(content, tool_calls)


class _GeminiProvider:
    """Wraps google-generativeai with the common interface."""

    def __init__(self) -> None:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=_API_KEY)
        self._genai = genai

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMMessage:
        genai = self._genai

        gen_cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Build Gemini tool declarations from OpenAI-style tool dicts
        tool_declarations = None
        if tools:
            tool_declarations = [
                genai.types.Tool(function_declarations=[
                    genai.types.FunctionDeclaration(name=t["function"]["name"])
                    for t in tools
                ])
            ]

        # Split messages into history + final prompt
        history: list[dict] = []
        prompt = ""
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content") or ""
            if role == "system":
                continue  # handled via system_instruction at model level
            elif role in ("user", "tool"):
                if content:
                    history.append({"role": "user",  "parts": [str(content)]})
                    prompt = str(content)
            elif role == "assistant":
                if content:
                    history.append({"role": "model", "parts": [str(content)]})

        mdl_kwargs: dict[str, Any] = dict(
            model_name=model,
            generation_config=gen_cfg,
        )
        if system:
            mdl_kwargs["system_instruction"] = system
        if tool_declarations:
            mdl_kwargs["tools"] = tool_declarations

        mdl  = genai.GenerativeModel(**mdl_kwargs)
        chat = mdl.start_chat(history=history[:-1] if len(history) > 1 else [])
        resp = chat.send_message(prompt)

        # Extract text
        try:
            content_str = resp.text
        except Exception:
            content_str = ""

        # Extract function calls
        tool_calls: list[_ToolCall] | None = None
        try:
            tcs: list[_ToolCall] = []
            for part in resp.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tcs.append(_ToolCall(
                        id_       = fc.name,
                        name      = fc.name,
                        arguments = json.dumps(dict(fc.args)),
                    ))
            if tcs:
                tool_calls = tcs
        except Exception:
            pass

        return LLMMessage(content_str, tool_calls)


class _OpenAIProvider:
    """Wraps openai.OpenAI with the common interface (also works for any
    OpenAI-compatible endpoint, e.g. Together, Mistral, etc.)"""

    def __init__(self) -> None:
        from openai import OpenAI  # type: ignore
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=_API_KEY, base_url=base_url)

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMMessage:
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        msg  = resp.choices[0].message

        content    = msg.content or ""
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                _ToolCall(
                    id_       = tc.id,
                    name      = tc.function.name,
                    arguments = tc.function.arguments or "{}",
                )
                for tc in msg.tool_calls
            ]

        return LLMMessage(content, tool_calls)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_provider():
    if _PROVIDER == "gemini":
        return _GeminiProvider()
    if _PROVIDER in ("openai", "together", "mistral"):
        return _OpenAIProvider()
    # Default / "groq"
    return _GroqProvider()


# Module-level singleton — imported by agents
provider = _build_provider()


def llm_complete(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
    tools: list[dict] | None = None,
    system: str | None = None,
) -> LLMMessage:
    """
    Single entry-point for all LLM calls.  Agents import this function only.

    Parameters
    ----------
    model       : model string from config.py
    messages    : OpenAI-style list of {"role": ..., "content": ...} dicts
    max_tokens  : upper token budget for this call
    temperature : sampling temperature
    tools       : OpenAI-style tool schema list (provider-translated internally)
    system      : system prompt / role string

    Returns
    -------
    LLMMessage with .content (str) and .tool_calls (list | None)
    """
    return provider.complete(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        system=system,
    )