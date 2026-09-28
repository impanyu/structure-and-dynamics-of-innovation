"""Paper 2 nested k-sweep figure (3 seeds per k), from runs/p2_forum/nested_summary.json.

Run after: uv run python scripts/p2_forum/ksweep_summary.py --nested
Writes paper/p2_forum/nested_ksweep.png: (A) uptake of a visible own vs teammate
post, (B) cross-agent citations per run, (C) ideas posted per run. Each seed is a
small dot; the line/bar is the mean over seeds. Same palette as plot_ksweep.py.

Run: uv run python scripts/p2_forum/plot_nested.py
"""
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#6b6a66"
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

rows = json.load(open("runs/p2_forum/nested_summary.json"))
by = collections.defaultdict(list)
for r in rows:
    by[r["k"]].append(r)
KS = sorted(by)
XI = list(range(len(KS)))
seeds = sorted({r["seed"] for r in rows})
steps = rows[0]["steps"]


def mean(k, f):
    v = [r[f] for r in by[k]]
    return sum(v) / len(v)


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(XI)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k  (topics each agent is interested in)")


def dots(ax, f, color, dx=0.0):
    for i, k in enumerate(KS):
        v = [r[f] for r in by[k]]
        ax.scatter([i + dx] * len(v), v, s=14, color=color, alpha=0.45, lw=0, zorder=2)


fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.0, 3.7), facecolor=SURF)
for ax in (a, b, c):
    style(ax)

for f, col, lab in (("p_own", BLUE, "own post in view"), ("p_mate", ORANGE, "teammate's post in view")):
    dots(a, f, col)
    a.plot(XI, [mean(k, f) for k in KS], color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, zorder=3, label=lab)
a.set_ylim(0, 0.75)
a.set_ylabel("P(cited | in 20-step memory)")
a.set_title("A  Uptake of a post the agent can see", loc="left", fontsize=10.5)
a.legend(frameon=False, loc="upper right", fontsize=8.5)

b.bar(XI, [mean(k, "cross") for k in KS], width=0.62, color=ORANGE, alpha=0.35, edgecolor=SURF, linewidth=2, zorder=1)
dots(b, "cross", ORANGE)
for i, k in enumerate(KS):
    b.text(i, max(r["cross"] for r in by[k]) + 0.35, f"{mean(k, 'cross'):.1f}", ha="center", fontsize=8.5, color=INK2)
b.set_ylim(0, max(r["cross"] for r in rows) * 1.25 + 1)
b.set_ylabel("cross-agent citations per run")
b.set_title("B  Citations of a teammate's post", loc="left", fontsize=10.5)

c.bar(XI, [mean(k, "posts") for k in KS], width=0.62, color=NEUTRAL, alpha=0.35, edgecolor=SURF, linewidth=2, zorder=1)
dots(c, "posts", NEUTRAL)
c.set_ylim(0, max(r["posts"] for r in rows) * 1.2)
c.set_ylabel("ideas posted per run (whole team)")
c.set_title("C  Ideas produced", loc="left", fontsize=10.5)

fig.text(0.01, -0.03, f"N = 10 agents, {steps // 10} rounds ({steps} steps), nested topic sets, "
         f"{len(seeds)} seeds per k. Dots: individual seeds; line/bar: mean.", fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig("paper/p2_forum/nested_ksweep.png", dpi=200, bbox_inches="tight", facecolor=SURF)
print("wrote paper/p2_forum/nested_ksweep.png")
