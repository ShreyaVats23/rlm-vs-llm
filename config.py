# ─────────────────────────────────────────────────────────────
# Model / API
# ─────────────────────────────────────────────────────────────
MODEL_A = "llama-3.1-8b-instant"  
MODEL    = MODEL_A          # unified alias used by both LLM and RLM agents

LLM_MAX_TOKENS = 200
RLM_MAX_TOKENS = 400


# ─────────────────────────────────────────────────────────────
# Retry / Backoff
# ─────────────────────────────────────────────────────────────
MAX_RETRIES   = 3
BACKOFF_BASE  = 1.2
BACKOFF_FACTOR = 2.0
BACKOFF_MAX   = 8.0


# ─────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────
BENCHMARK_SAVE_PATH = "benchmark_results.json"

RUNS_PER_QUERY = 1   

CTX_DEPTH_CAP = 12
LATENCY_CAP   = 480

# Scoring weights
W_QUALITY    = 0.50
W_CTX        = 0.25
W_EFFICIENCY = 0.15
W_LATENCY    = 0.10


# ─────────────────────────────────────────────────────────────
# RLM Complexity Signals
# ─────────────────────────────────────────────────────────────
MULTIHOP_SIGNALS = [
    "workflow",
    "multi-agent",
    "multi hop",
    "pipeline",
    "full system",
    "end-to-end",
]

HEAVY_SIGNALS = [
    "analyse",
    "analysis",
    "limitations",
    "advantages",
    "compare",
    "evaluation",
    "reasoning",
]


# ─────────────────────────────────────────────────────────────
# RLM Tool + Depth Budgeting
# ─────────────────────────────────────────────────────────────
TOOL_BUDGET = {
    "simple": {
        "calls":   1,
        "rounds":  1,
        "depth":   3,
        "steps":   2,
        "delay":   0.05,
        "timeout": 60,
    },
    "medium": {
        "calls":   2,
        "rounds":  2,
        "depth":   6,
        "steps":   6,
        "delay":   0.08,
        "timeout": 60,
    },
    "heavy": {
        "calls":   2,
        "rounds":  2,
        "depth":   9,
        "steps":   10,
        "delay":   0.10,
        "timeout": 60,
    },
    "multi-hop": {
        "calls":   2,
        "rounds":  4,
        "depth":   12,
        "steps":   15,
        "delay":   0.12,
        "timeout": 60,
    },
}


# ─────────────────────────────────────────────────────────────
# RLM Output Constraints
# ─────────────────────────────────────────────────────────────
FINAL_MIN_LEN = 200


# ─────────────────────────────────────────────────────────────
# MCP (Tool System)
# ─────────────────────────────────────────────────────────────
CACHE_TTL_S           = 600
HTTP_TIMEOUT_S        = 4
HTTP_POOL_WORKERS     = 4

CODE_TIMEOUT_S        = 8

MAX_EVIDENCE_ITEMS    = 5
MAX_SEARCH_RESULT_LEN = 1600

MAX_MEMORY_ITEMS      = 200
MAX_MEMORY_VALUE_LEN  = 1200