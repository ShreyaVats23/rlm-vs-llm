# RLM vs LLM — Multi-Agent Context Propagation Study

> **Does passing the full reasoning context between agents produce better answers than passing just the task string?**
> This repository answers that empirically.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Model](https://img.shields.io/badge/Model-Llama--3.1--8B--Instant-FF6B35?style=flat-square)](https://groq.com)
[![Provider](https://img.shields.io/badge/API-Groq%20%7C%20Gemini%20%7C%20OpenAI-00A67E?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-B31B1B?style=flat-square)](https://github.com/ShreyaVats23/rlm-vs-llm)

---

## What This Is

This project implements and benchmarks two complete multi-agent AI architectures that share **identical infrastructure** but differ in one key design decision: **what gets passed when one agent delegates a task to another**.

| System | What gets passed in delegation |
|--------|-------------------------------|
| **LLM** (baseline) | Task string only — specialist starts fresh |
| **RLM** (proposed) | Task string + full context snapshot — specialist inherits orchestrator's reasoning |

Both systems use the same model (Llama-3.1-8B-Instant), the same MCP tools (web search, shared memory), and the same A2A message bus. The only variable is context propagation.

---

## Architecture

```
LLM Multi-Agent (Baseline)                RLM Multi-Agent (Proposed)
──────────────────────────                ──────────────────────────
User Query                                User Query
    │                                         │
    ▼                                         ▼
LLM-A  Orchestrator                      RLM-A  Orchestrator
  · One-shot task plan                     · detect_complexity()
  · ctx_depth = 1 (fixed)                  · RecursiveContext stack
  · Dispatches task STRING only            · ctx.snapshot() → A2ATask payload
    │              │                           │                  │
    ▼              ▼                           ▼ (sequential)     ▼
LLM-B          LLM-C                      RLM-B              RLM-C
Research       Implementation             Research           Implementation
Starts fresh   Starts fresh               Inherits ctx       Inherits ctx
    │              │                           │                  │
    └──────────────┘                           └──────────────────┘
           │                                          │
           ▼                                          ▼
    One-shot synthesis                    recursive_think() synthesis
    ctx_depth = 1                         ctx_depth = 4–19
    tool_calls = 0                        tool_calls = 0–4
```

### Key Files

| File | Purpose |
|------|---------|
| `rlm_agents.py` | RLM orchestrator + specialists with `recursive_think()` loop |
| `llm_agents.py` | LLM baseline — linear 4-step pipeline |
| `rlm_context.py` | `RecursiveContext` — typed layer stack, compression, snapshot/restore |
| `a2a_bus.py` | asyncio-native A2A message bus with `asyncio.Event` completion |
| `mcp_tools.py` | MCP tool registry: `search_web`, `memory_store/get/search/list` |
| `llm_provider.py` | Unified provider shim: Groq / Gemini / OpenAI via single `llm_complete()` |
| `benchmark.py` | Evaluation framework: scoring, averaging, JSON output |
| `config.py` | All constants: token budgets, weights, TOOL_BUDGET tiers |
| `generate_report_charts.py` | 6 publication-quality charts from benchmark results |

---

## Benchmark Results

Both systems evaluated on 4 query complexity tiers using Llama-3.1-8B-Instant.

### Composite Scores (0–10)

```
Query              LLM     RLM     Delta    Winner
─────────────────────────────────────────────────
Q1  Simple         3.23    4.54    +1.31    RLM  ✓
Q2  Medium         3.20    5.38    +2.18    RLM  ✓
Q3  Heavy ctx      3.21    5.80    +2.59    RLM  ✓
Q4  Multi-hop      2.19    4.88    +2.69    RLM  ✓
─────────────────────────────────────────────────
Average            2.96    5.15    +2.19    RLM (4/4)
```

### Secondary Metrics (LLM / RLM)

```
Metric              Q1           Q2            Q3            Q4
────────────────────────────────────────────────────────────────
Quality          6  /  7      6  /  7       6  /  7       4  /  6
Context depth    1  /  4      1  / 14       1  / 16       1  / 19
Tool calls       0  /  1      0  /  0       0  /  4       0  /  0
Steps            4  /  1      4  / 18       4  / 20       4  / 26
Latency (s)    2.76/ 2.41  18.73/630.57  12.39/450.35  23.55/864.58
```

**Key finding:** The RLM advantage grows monotonically with complexity (+1.31 → +2.69), confirming that recursive context propagation earns its overhead in direct proportion to task difficulty. On Q1 (simple), RLM is actually *faster* (2.41s vs 2.76s) because the `simple` complexity tier skips delegation entirely.

---

## Complexity Routing

Queries are automatically routed to one of four tiers by `detect_complexity()` in `rlm_agents.py`:

| Tier | Depth | Steps | Tool calls | Timeout |
|------|-------|-------|-----------|---------|
| `simple` | 3 | 2 | 1 | 60s |
| `medium` | 6 | 6 | 2 | 60s |
| `heavy` | 9 | 10 | 2 | 60s |
| `multi-hop` | 12 | 15 | 2 | 60s |

Detection uses keyword signals from `config.py`:
- **Multi-hop signals:** `workflow`, `multi-agent`, `pipeline`, `end-to-end`, ...
- **Heavy signals:** `analyse`, `limitations`, `compare`, `evaluation`, `reasoning`, ...

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ShreyaVats23/rlm-vs-llm.git
cd rlm-vs-llm
pip install -r requirements.txt
```

### 2. Configure API

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
API_PROVIDER=groq          # groq | gemini | openai
API_KEY=your_key_here

# Optional — only for OpenAI-compatible endpoints
# OPENAI_BASE_URL=https://api.together.xyz/v1
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — it's the fastest option and what the benchmark was run on.

### 3. Run the benchmark

```bash
python benchmark.py
```

Results are saved to `benchmark_results.json`.

### 4. Generate charts

```bash
python generate_report_charts.py
```

Outputs 6 charts:
- `final_score.png` — composite score comparison
- `quality.png` — task quality scores
- `context_depth.png` — context utilisation
- `latency.png` — response latency
- `ctx_vs_quality_scatter.png` — context depth vs performance scatter
- `radar_summary.png` — system profile radar

### 5. Run a single query (quick test)

```bash
# LLM system
python llm_agents.py

# RLM system
python rlm_agents.py
```

---

## Scoring Formula

```
final = 0.50 × quality
      + 0.25 × ctx_norm          (ctx_depth / CTX_DEPTH_CAP=12, normalised)
      + 0.15 × efficiency        (tool_calls × 0.6 + ctx_norm × 0.25)
      − 0.10 × latency_penalty   (latency / LATENCY_CAP=480s, normalised)
```

Task quality (0–10) is assessed across five dimensions:
1. **Length sufficiency** — complexity-scaled thresholds
2. **Structural richness** — `##` headings, lists, code blocks
3. **Keyword coverage** — query-specific expected terms
4. **Analytical depth** — depth marker vocabulary
5. **Completeness** — conclusion / summary presence

---

## Switching Providers

No code changes needed. Just update `.env`:

```env
# Groq (default — fastest)
API_PROVIDER=groq
API_KEY=gsk_...

# Google Gemini
API_PROVIDER=gemini
API_KEY=AIza...

# OpenAI
API_PROVIDER=openai
API_KEY=sk-...

# OpenAI-compatible (Together, Mistral, etc.)
API_PROVIDER=openai
API_KEY=your_key
OPENAI_BASE_URL=https://api.together.xyz/v1
```

Model is set in `config.py` — change `MODEL_A` to match your provider.

---

## Known Limitations

- **Single-run design** — `RUNS_PER_QUERY=1` means variance is uncharacterised. The averaging infrastructure in `_avg_run()` exists but was not activated due to cost constraints.
- **Token budget asymmetry** — `LLM_MAX_TOKENS=200` vs `RLM_MAX_TOKENS=400` may partially explain quality differences beyond architecture.
- **LLM tool calls = 0** — MCP tool schemas are not wired to LLM specialist agents in the current implementation. The LLM/RLM comparison is therefore partially tool-free vs tool-enabled.
- **Q2 latency anomaly** — RLM Q2 (630s) exceeds Q3 (450s) despite lower complexity, caused by `_build_directive()` FINAL: signal fragility on multi-domain queries.

---

## Paper

This repository accompanies the research paper:

> **Improving AI Agent Context with Recursive Language Models and Model Context Protocol: A Comparative Study of LLM and RLM Multi-Agent Systems**
> Shreya Vats, Dr. Narjis Fatima
> Institute of Information Technology and Management, New Delhi
> April 2026

---

## License

MIT © 2026 Shreya Vats — see [LICENSE](LICENSE)
