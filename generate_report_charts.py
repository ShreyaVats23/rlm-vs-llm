"""
generate_report_charts.py — Publication-Quality Benchmark Visualisations
=========================================================================
Generates six professional charts from benchmark_results.json:

  1. final_score.png          — Composite score comparison (grouped bar)
  2. quality.png              — Raw task quality (grouped bar)
  3. context_depth.png        — Context utilisation (grouped bar)
  4. latency.png              — Latency comparison (grouped bar)
  5. ctx_vs_quality_scatter.png — Context depth vs final score (scatter)
  6. radar_summary.png        — Radar chart: 4-metric system profile

All charts use a consistent dark-themed palette suitable for academic reports.
"""

import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444466",
    "axes.labelcolor":   "#e0e0ff",
    "axes.titlecolor":   "#ffffff",
    "axes.titlesize":    14,
    "axes.labelsize":    11,
    "xtick.color":       "#aaaacc",
    "ytick.color":       "#aaaacc",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.color":        "#2a2a4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "legend.facecolor":  "#1a1a3a",
    "legend.edgecolor":  "#444466",
    "legend.labelcolor": "#e0e0ff",
    "legend.fontsize":   10,
    "text.color":        "#e0e0ff",
    "font.family":       "DejaVu Sans",
})

C_LLM  = "#5b9bd5"   # steel blue
C_RLM  = "#ed7d31"   # vibrant orange
C_WIN  = "#70ad47"   # win indicator green

WIDTH  = 0.32
DPI    = 160
FIGW   = 9
FIGH   = 5


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
with open("benchmark_results.json", "r") as f:
    raw = json.load(f)

# Support both old (list) and new (dict with meta/results) formats
if isinstance(raw, dict) and "results" in raw:
    items   = raw["results"]
    meta    = raw.get("meta", {})
    summary = raw.get("summary", {})
else:
    items   = raw if isinstance(raw, list) else raw.get("results", [])
    meta    = {}
    summary = {}

queries       = [it["id"]                       for it in items]
labels        = [f"{it['id']}\n{it['label']}"   for it in items]
llm_final     = [it["llm"]["final_score"]        for it in items]
rlm_final     = [it["rlm"]["final_score"]        for it in items]
llm_quality   = [it["llm"]["task_quality"]       for it in items]
rlm_quality   = [it["rlm"]["task_quality"]       for it in items]
llm_time      = [it["llm"]["time_s"]             for it in items]
rlm_time      = [it["rlm"]["time_s"]             for it in items]
llm_ctx       = [it["llm"]["ctx_depth"]          for it in items]
rlm_ctx       = [it["rlm"]["ctx_depth"]          for it in items]
llm_tools     = [it["llm"].get("tool_calls", 0)  for it in items]
rlm_tools     = [it["rlm"].get("tool_calls", 0)  for it in items]
winners       = [it.get("winner", "TIE")         for it in items]

n  = len(queries)
x  = np.arange(n)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: annotate bars
# ──────────────────────────────────────────────────────────────────────────────
def annotate_bars(ax, bars, fmt="{:.2f}", color="#ffffff", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.06,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
        )


def winner_band(ax, winners, x, ymax, alpha=0.06):
    """Shade background green where RLM wins."""
    for i, w in enumerate(winners):
        if w == "RLM":
            ax.axvspan(x[i] - 0.45, x[i] + 0.45, color=C_WIN, alpha=alpha, zorder=0)


def shared_legend(ax):
    ax.legend(
        handles=[
            mpatches.Patch(color=C_LLM, label="LLM — Linear Pipeline"),
            mpatches.Patch(color=C_RLM, label="RLM — Recursive System"),
        ],
        loc="upper left",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. FINAL SCORE
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
b1 = ax.bar(x - WIDTH / 2, llm_final, WIDTH, color=C_LLM, label="LLM", zorder=3)
b2 = ax.bar(x + WIDTH / 2, rlm_final, WIDTH, color=C_RLM, label="RLM", zorder=3)
winner_band(ax, winners, x, max(max(llm_final), max(rlm_final)))
annotate_bars(ax, b1)
annotate_bars(ax, b2)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Query"); ax.set_ylabel("Composite Score (0-10)")
ax.set_title("LLM vs RLM — Composite Final Score"); ax.set_ylim(0, 12)
ax.grid(axis="y", zorder=0)
shared_legend(ax)
fig.tight_layout()
fig.savefig("final_score.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 2. TASK QUALITY
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
b1 = ax.bar(x - WIDTH / 2, llm_quality, WIDTH, color=C_LLM, label="LLM", zorder=3)
b2 = ax.bar(x + WIDTH / 2, rlm_quality, WIDTH, color=C_RLM, label="RLM", zorder=3)
winner_band(ax, winners, x, max(max(llm_quality), max(rlm_quality)))
annotate_bars(ax, b1, fmt="{:.0f}")
annotate_bars(ax, b2, fmt="{:.0f}")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Query"); ax.set_ylabel("Task Quality Score (0-10)")
ax.set_title("LLM vs RLM — Answer Quality"); ax.set_ylim(0, 12)
ax.grid(axis="y", zorder=0)
shared_legend(ax)
fig.tight_layout()
fig.savefig("quality.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CONTEXT DEPTH
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
b1 = ax.bar(x - WIDTH / 2, llm_ctx, WIDTH, color=C_LLM, label="LLM", zorder=3)
b2 = ax.bar(x + WIDTH / 2, rlm_ctx, WIDTH, color=C_RLM, label="RLM", zorder=3)
annotate_bars(ax, b1, fmt="{:.0f}")
annotate_bars(ax, b2, fmt="{:.0f}")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Query"); ax.set_ylabel("Context Depth (layers)")
ax.set_title("Context Utilisation — LLM vs RLM"); ax.set_ylim(0, max(max(rlm_ctx)+3, 5))
ax.grid(axis="y", zorder=0)
shared_legend(ax)
fig.tight_layout()
fig.savefig("context_depth.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 4. LATENCY
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
b1 = ax.bar(x - WIDTH / 2, llm_time, WIDTH, color=C_LLM, label="LLM", zorder=3)
b2 = ax.bar(x + WIDTH / 2, rlm_time, WIDTH, color=C_RLM, label="RLM", zorder=3)
annotate_bars(ax, b1, fmt="{:.1f}")
annotate_bars(ax, b2, fmt="{:.1f}")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Query"); ax.set_ylabel("Latency (seconds)")
ax.set_title("Response Latency — LLM vs RLM"); ax.set_ylim(0, max(max(llm_time), max(rlm_time)) * 1.25)
ax.grid(axis="y", zorder=0)
shared_legend(ax)
fig.tight_layout()
fig.savefig("latency.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 5. SCATTER: Context Depth vs Final Score
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(FIGW, FIGH))
jitter_llm = rng.uniform(-0.12, 0.12, n)
jitter_rlm = rng.uniform(-0.12, 0.12, n)

ax.scatter(np.array(llm_ctx) + jitter_llm, llm_final,
           color=C_LLM, s=90, zorder=4, label="LLM", edgecolors="#ffffff", linewidths=0.5)
ax.scatter(np.array(rlm_ctx) + jitter_rlm, rlm_final,
           color=C_RLM, s=90, zorder=4, label="RLM", edgecolors="#ffffff", linewidths=0.5)

for i, q in enumerate(queries):
    ax.annotate(q, (llm_ctx[i] + jitter_llm[i], llm_final[i]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=8, color=C_LLM, alpha=0.85)
    ax.annotate(q, (rlm_ctx[i] + jitter_rlm[i], rlm_final[i]),
                textcoords="offset points", xytext=(5, -10),
                fontsize=8, color=C_RLM, alpha=0.85)

# Trend lines
for vals_x, vals_y, color in [(llm_ctx, llm_final, C_LLM), (rlm_ctx, rlm_final, C_RLM)]:
    if len(set(vals_x)) > 1:
        m, b = np.polyfit(vals_x, vals_y, 1)
        xs = np.linspace(min(vals_x) - 0.5, max(vals_x) + 0.5, 50)
        ax.plot(xs, m * xs + b, color=color, linestyle="--", linewidth=1, alpha=0.5)

ax.set_xlabel("Context Depth (layers)")
ax.set_ylabel("Composite Final Score (0-10)")
ax.set_title("Context Depth vs Performance — Correlation")
ax.grid(zorder=0)
shared_legend(ax)
fig.tight_layout()
fig.savefig("ctx_vs_quality_scatter.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6. RADAR / SPIDER — System Profile
# ──────────────────────────────────────────────────────────────────────────────
categories = ["Avg Quality", "Avg Ctx Depth", "Avg Tools", "Speed\n(inv. latency)", "Final Score"]
N_cat = len(categories)

def safe_norm(vals, cap):
    m = sum(vals) / len(vals) if vals else 0
    return min(m / cap, 1.0) * 10

avg_llm_q    = safe_norm(llm_quality, 10)
avg_rlm_q    = safe_norm(rlm_quality, 10)
avg_llm_ctx  = safe_norm(llm_ctx,     12)
avg_rlm_ctx  = safe_norm(rlm_ctx,     12)
avg_llm_tool = safe_norm(llm_tools,   8)
avg_rlm_tool = safe_norm(rlm_tools,   8)
max_lat = max(max(llm_time), max(rlm_time), 1)
avg_llm_spd  = (1 - min(sum(llm_time)/len(llm_time)/max_lat, 1)) * 10
avg_rlm_spd  = (1 - min(sum(rlm_time)/len(rlm_time)/max_lat, 1)) * 10
avg_llm_fs   = safe_norm(llm_final,  10)
avg_rlm_fs   = safe_norm(rlm_final,  10)

llm_vals = [avg_llm_q, avg_llm_ctx, avg_llm_tool, avg_llm_spd, avg_llm_fs]
rlm_vals = [avg_rlm_q, avg_rlm_ctx, avg_rlm_tool, avg_rlm_spd, avg_rlm_fs]

angles   = [n_i / N_cat * 2 * math.pi for n_i in range(N_cat)]
angles  += angles[:1]
llm_vals += llm_vals[:1]
rlm_vals += rlm_vals[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

ax.plot(angles, llm_vals, color=C_LLM, linewidth=2, linestyle="solid", label="LLM")
ax.fill(angles, llm_vals, color=C_LLM, alpha=0.18)
ax.plot(angles, rlm_vals, color=C_RLM, linewidth=2, linestyle="solid", label="RLM")
ax.fill(angles, rlm_vals, color=C_RLM, alpha=0.18)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color="#e0e0ff", fontsize=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#888899", fontsize=8)
ax.grid(color="#2a2a5a", linestyle="--", alpha=0.8)
ax.spines["polar"].set_color("#444466")
ax.set_title("System Profile — LLM vs RLM\n(Normalised 0-10)", color="#ffffff",
             fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
          handles=[
              mpatches.Patch(color=C_LLM, label="LLM — Linear Pipeline"),
              mpatches.Patch(color=C_RLM, label="RLM — Recursive System"),
          ])
fig.tight_layout()
fig.savefig("radar_summary.png", dpi=DPI)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
print("\nCharts generated:")
for f in ["final_score.png", "quality.png", "context_depth.png",
          "latency.png", "ctx_vs_quality_scatter.png", "radar_summary.png"]:
    print(f"  -> {f}")