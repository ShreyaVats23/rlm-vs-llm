"""
benchmark.py — LLM vs RLM Comparative Evaluation Framework
===========================================================
Evaluates four query categories across two agent architectures:
  LLM — Linear multi-agent pipeline (LLM-A / B / C)
  RLM — Recursive multi-agent system (RLM-A / B / C)

Scoring model (weights configured in config.py):
  task_quality   0–10   Semantic richness of the answer
  ctx_depth      0–10   Contextual depth accumulated during reasoning
  efficiency     0–10   Tool usage relative to task complexity
  latency        0–10   Normalised response time (lower is better — penalty)
  final_score    0–10   Weighted composite

  final = W_QUALITY*quality + W_CTX*ctx_norm + W_EFFICIENCY*eff_norm
          - W_LATENCY*latency_penalty
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any

from config import (
    BENCHMARK_SAVE_PATH,
    CTX_DEPTH_CAP,
    LATENCY_CAP,
    RUNS_PER_QUERY,
    W_CTX,
    W_EFFICIENCY,
    W_LATENCY,
    W_QUALITY,
)
from llm_agents import run_llm_system
from mcp_tools import reset_logs
from rlm_agents import run_rlm_system


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation queries
# ──────────────────────────────────────────────────────────────────────────────
QUERIES: list[dict[str, Any]] = [
    {
        "id":                "Q1",
        "label":             "Simple",
        "query":             "What is the difference between a list and a tuple in Python?",
        "expected_keywords": ["mutable", "immutable", "tuple", "list", "hashable"],
        "complexity":        "simple",
    },
    {
        "id":                "Q2",
        "label":             "Medium",
        "query":             "Explain transformer attention mechanisms and compare LLM vs RLM agents.",
        "expected_keywords": ["attention", "transformer", "context", "recursive", "comparison", "agent"],
        "complexity":        "medium",
    },
    {
        "id":    "Q3",
        "label": "Heavy context",
        "query": (
            "Analyse the limitations of standard LLM systems, the complexity "
            "of MCP/A2A integrations, and the advantages of RLM architectures "
            "for long-horizon reasoning tasks."
        ),
        "expected_keywords": ["limitations", "context", "recursive", "mcp", "a2a", "advantage", "reasoning"],
        "complexity":        "heavy",
    },
    {
        "id":    "Q4",
        "label": "Multi-hop",
        "query": (
            "Explain the full workflow, context flow, and failure modes in "
            "LLM vs RLM multi-agent systems, including how each handles "
            "tool use, memory, and cross-agent delegation."
        ),
        "expected_keywords": ["workflow", "context flow", "failure", "tool", "memory", "delegation", "comparison"],
        "complexity":        "multi-hop",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────
def _normalise(val: float, cap: float) -> float:
    """Min-max normalise val against cap, clipped to [0, 1]."""
    return min(max(val / cap, 0.0), 1.0)


def _score_answer(answer: str, expected_keywords: list[str], complexity: str) -> int:
    """
    Multi-dimensional quality scorer (0–10).

    Dimensions
    ----------
    1. Length sufficiency  — thresholds scaled by complexity.
    2. Structural richness — headings, lists, code blocks.
    3. Keyword coverage    — domain-specific expected terms.
    4. Analytical depth    — comparison, analysis, evidence language.
    5. Completeness        — conclusion or summary present.
    """
    if not answer or len(answer.strip()) < 30:
        return 1

    a     = answer.lower()
    score = 0

    # 1. Length sufficiency
    length_thresholds: dict[str, tuple[int, int]] = {
        "simple":    (80,   250),
        "medium":    (200,  600),
        "heavy":     (400,  900),
        "multi-hop": (500, 1_100),
    }
    lo, hi = length_thresholds.get(complexity, (150, 500))
    score += (1 if len(answer) >= lo else 0) + (1 if len(answer) >= hi else 0)

    # 2. Structural richness
    score += 1 if any(h in answer for h in ("##", "**", "---")) else 0
    score += 1 if ("1." in answer or "- " in answer) else 0
    score += 1 if "```" in answer else 0

    # 3. Keyword coverage
    kw_hits = sum(1 for kw in expected_keywords if kw.lower() in a)
    score  += min(kw_hits, 2)

    # 4. Analytical depth
    depth_markers = [
        "comparison", "analysis", "trade-off", "advantage", "limitation",
        "however", "in contrast", "on the other hand", "evidence", "therefore",
        "furthermore", "specifically",
    ]
    depth_hits = sum(1 for m in depth_markers if m in a)
    score     += min(depth_hits // 2, 2)

    # 5. Completeness
    score += 1 if any(c in a for c in ("conclusion", "summary", "in summary", "to conclude")) else 0

    return min(score, 10)


def _compute_final_score(run: dict[str, Any]) -> float:
    """
    Composite score formula:
      final = W_QUALITY*quality + W_CTX*ctx_norm + W_EFFICIENCY*eff_norm - W_LATENCY*lat_penalty
    Clipped to [1.0, 10.0].
    """
    quality  = run["task_quality"]
    ctx_norm = _normalise(run["ctx_depth"], CTX_DEPTH_CAP) * 10
    lat_norm = _normalise(run["time_s"],    LATENCY_CAP)   * 10
    eff_norm = min(run["tool_calls"] * 0.6 + ctx_norm * 0.25, 10)

    final = (
        W_QUALITY    * quality
        + W_CTX      * ctx_norm
        + W_EFFICIENCY * eff_norm
        - W_LATENCY  * lat_norm
    )
    return round(max(1.0, min(final, 10.0)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────
async def _run_single(system_fn, name: str, q: dict) -> dict[str, Any]:
    """Execute one system on one query and return a scored result dict."""
    reset_logs()
    t0      = time.monotonic()
    result  = await system_fn(q["query"])
    elapsed = round(time.monotonic() - t0, 2)

    quality = _score_answer(
        result.get("answer", ""),
        q.get("expected_keywords", []),
        q.get("complexity", "medium"),
    )
    run = {
        "system":       name,
        "time_s":       result.get("time_s", elapsed),
        "ctx_depth":    result.get("ctx_depth", 0),
        "tool_calls":   result.get("tool_calls", 0),
        "task_quality": quality,
        "answer_len":   len(result.get("answer", "")),
        "confidence":   result.get("confidence", 0.0),
        "steps":        result.get("steps", 0),
    }
    run["final_score"] = _compute_final_score(run)
    return run


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ──────────────────────────────────────────────────────────────────────────────
async def benchmark() -> dict[str, Any]:
    results: list[dict] = []

    for q in QUERIES:
        print(f"\n{'=' * 72}")
        print(f"  {q['id']} [{q['label']}]  complexity={q['complexity']}")
        print(f"  Query: {q['query'][:80]}...")
        print(f"{'=' * 72}")

        # Average over RUNS_PER_QUERY for statistical stability
        llm_runs = [await _run_single(run_llm_system, "LLM", q) for _ in range(RUNS_PER_QUERY)]
        rlm_runs = [await _run_single(run_rlm_system, "RLM", q) for _ in range(RUNS_PER_QUERY)]

        # Aggregate multi-run results by averaging numeric fields
        def _avg_run(runs: list[dict]) -> dict:
            if len(runs) == 1:
                return runs[0]
            numeric_keys = ["time_s", "ctx_depth", "tool_calls", "task_quality",
                            "answer_len", "confidence", "steps", "final_score"]
            avg = dict(runs[0])
            for k in numeric_keys:
                avg[k] = round(statistics.mean(r[k] for r in runs), 4)
            return avg

        llm_res = _avg_run(llm_runs)
        rlm_res = _avg_run(rlm_runs)

        delta   = round(rlm_res["final_score"] - llm_res["final_score"], 2)
        winner  = "RLM" if delta > 0 else ("LLM" if delta < 0 else "TIE")

        # Per-query table
        print(f"\n  {'Metric':<22} {'LLM':>8} {'RLM':>8} {'Delta':>8} {'Winner':>8}")
        print(f"  {'-' * 56}")
        metric_rows = [
            ("Final Score",   llm_res["final_score"],  rlm_res["final_score"]),
            ("Task Quality",  llm_res["task_quality"], rlm_res["task_quality"]),
            ("Context Depth", llm_res["ctx_depth"],    rlm_res["ctx_depth"]),
            ("Tool Calls",    llm_res["tool_calls"],   rlm_res["tool_calls"]),
            ("Latency (s)",   llm_res["time_s"],       rlm_res["time_s"]),
            ("Ans. Length",   llm_res["answer_len"],   rlm_res["answer_len"]),
        ]
        for label, lv, rv in metric_rows:
            dv     = rv - lv
            dv_str = f"{dv:+.2f}"
            print(f"  {label:<22} {lv:>8} {rv:>8} {dv_str:>8}")
        print(f"  {'Winner':<22} {'':>8} {'':>8} {'':>8} {winner:>8}")

        results.append({
            "id":     q["id"],
            "label":  q["label"],
            "query":  q["query"],
            "llm":    llm_res,
            "rlm":    rlm_res,
            "delta":  delta,
            "winner": winner,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 72}")
    print("  FINAL BENCHMARK SUMMARY — LLM vs RLM")
    print(f"{'=' * 72}")
    print(f"  {'ID':<6} {'Type':<16} {'LLM':>8} {'RLM':>8} {'Delta':>8} {'Winner':>8}")
    print(f"  {'-' * 58}")

    def _col(field: str, results: list[dict], system: str) -> list[float]:
        return [r[system][field] for r in results]

    for row in results:
        l, r = row["llm"], row["rlm"]
        print(
            f"  {row['id']:<6} {row['label']:<16}"
            f"{l['final_score']:>8.2f} {r['final_score']:>8.2f}"
            f"{row['delta']:>+8.2f} {row['winner']:>8}"
        )

    print(f"  {'-' * 58}")

    avg_llm = round(statistics.mean(_col("final_score", results, "llm")), 2)
    avg_rlm = round(statistics.mean(_col("final_score", results, "rlm")), 2)
    improvement = round(((avg_rlm - avg_llm) / avg_llm) * 100, 2) if avg_llm else 0.0

    rlm_wins = sum(1 for r in results if r["winner"] == "RLM")
    llm_wins = sum(1 for r in results if r["winner"] == "LLM")
    ties     = sum(1 for r in results if r["winner"] == "TIE")

    print(f"\n  Average final score    LLM: {avg_llm}/10  |  RLM: {avg_rlm}/10")
    print(f"  RLM improvement        {improvement:+.2f}%")
    print(f"  Avg task quality       LLM: {round(statistics.mean(_col('task_quality', results, 'llm')), 2)}"
          f"   |  RLM: {round(statistics.mean(_col('task_quality', results, 'rlm')), 2)}")
    print(f"  Avg context depth      LLM: {round(statistics.mean(_col('ctx_depth', results, 'llm')), 2)}"
          f"   |  RLM: {round(statistics.mean(_col('ctx_depth', results, 'rlm')), 2)}")
    print(f"  Avg tool calls         LLM: {round(statistics.mean(_col('tool_calls', results, 'llm')), 2)}"
          f"   |  RLM: {round(statistics.mean(_col('tool_calls', results, 'rlm')), 2)}")
    print(f"  Avg latency            LLM: {round(statistics.mean(_col('time_s', results, 'llm')), 2)}s"
          f"  |  RLM: {round(statistics.mean(_col('time_s', results, 'rlm')), 2)}s")
    print(f"  Query wins             LLM: {llm_wins}  |  RLM: {rlm_wins}  |  Ties: {ties}")
    print(f"{'=' * 72}")

    # ── Persist results ───────────────────────────────────────────────────────
    output = {
        "meta": {
            "date":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":     "llama-3.1-8b-instant",
            "framework": "LLM vs RLM Comparative Evaluation v2.0",
            "scoring": {
                "w_quality":     W_QUALITY,
                "w_ctx":         W_CTX,
                "w_efficiency":  W_EFFICIENCY,
                "w_latency":     W_LATENCY,
                "ctx_depth_cap": CTX_DEPTH_CAP,
                "latency_cap":   LATENCY_CAP,
                "formula":       (
                    "final = 0.50*quality + 0.25*ctx_norm "
                    "+ 0.15*efficiency - 0.10*latency_penalty"
                ),
            },
        },
        "summary": {
            "avg_llm_score":   avg_llm,
            "avg_rlm_score":   avg_rlm,
            "rlm_improvement": f"{improvement:+.2f}%",
            "llm_wins":        llm_wins,
            "rlm_wins":        rlm_wins,
            "ties":            ties,
        },
        "results": results,
    }

    with open(BENCHMARK_SAVE_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n  Saved -> {BENCHMARK_SAVE_PATH}")
    return output


if __name__ == "__main__":
    asyncio.run(benchmark())