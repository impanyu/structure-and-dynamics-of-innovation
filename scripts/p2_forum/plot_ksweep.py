"""Paper 2 k-sweep figures, from runs/p2_forum/.

Run after scripts/p2_forum/ksweep_summary.py (which writes ksweep_summary.json).
Writes paper/p2_forum/ksweep.png (uptake) and ksweep_production.png (output).

Palette: the dataviz reference instance. Own/teammate use categorical slots 1/2
(validated, CVD dE 24.7). k is ordinal, but an 8-step blue ramp fails the
adjacent-lightness check, so the cumulative panel shows k=2..64 as grey context
and highlights only the two endpoints (validated 2-step ordinal pair).

Run: uv run python scripts/p2_forum/plot_ksweep.py
"""
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
XI = list(range(len(KS)))  # equal spacing; the grid switches from doubling to +16 after 32
BREAK = KS.index(32) + 0.5
SURF, INK, INK2, GRID, CTX = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"
BLUE, ORANGE, LIGHT, DARK = "#2a78d6", "#eb6834", "#86b6ef", "#0d366b"
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


rows = json.load(open("runs/p2_forum/ksweep_summary.json"))
steps = rows[0]["steps"]
rounds = steps // 10
foot = f"N = 10 agents, {rounds} rounds ({steps} steps), one seed per k."

# ---- figure 1: uptake -------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 3.6), facecolor=SURF)
for ax in (a, b):
    style(ax)
    ax.set_xticks(XI)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k  (topics each agent is interested in)")
    ax.axvline(BREAK, color=GRID, lw=1, ls=(0, (3, 3)), zorder=0)
po = [r["p_own"] for r in rows]
pm = [r["p_mate"] for r in rows]
a.plot(XI, po, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label="own post in view")
a.plot(XI, pm, color=ORANGE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label="teammate's post in view")
a.set_ylim(0, 0.55)
a.set_ylabel("P(cited | in 20-step memory)")
a.set_title("A  Uptake of a post the agent can see", loc="left", fontsize=10.5)
a.text(XI[-1] + 0.3, po[-1], "own", color=INK2, va="center", fontsize=9)
a.text(XI[-1] + 0.3, pm[-1], "teammate", color=INK2, va="center", fontsize=9)
a.legend(frameon=False, loc="upper left", fontsize=8.5)
a.set_xlim(-0.5, XI[-1] + 2.2)
a.text(BREAK, 0.005, " doubling | +16 ", ha="center", va="bottom", fontsize=7.5, color=INK2, backgroundcolor=SURF)
cross = [r["cross"] for r in rows]
b.bar(XI, cross, width=0.62, color=ORANGE, edgecolor=SURF, linewidth=2)
for x, y in zip(XI, cross):
    b.text(x, y + 0.3, str(y), ha="center", fontsize=8.5, color=INK2)
b.set_ylim(0, max(cross) * 1.2 + 1)
b.set_ylabel("cross-agent citations (count)")
b.set_title("B  Citations of a teammate's post, per run", loc="left", fontsize=10.5)
b.set_xlim(-0.6, XI[-1] + 0.6)
fig.text(0.01, -0.02, foot + " Rates are over the posts visible in each agent's "
         "20-step memory at writing time.", fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig("paper/p2_forum/ksweep.png", dpi=200, bbox_inches="tight", facecolor=SURF)

# ---- figure 2: production ---------------------------------------------------
tot, per_agent, cum = {}, {}, {}
for k in KS:
    ev = [json.loads(line) for line in open(f"runs/p2_forum/forum-k{k}/events.jsonl")]
    posts = [e for e in ev if e["action"] == "generate" and "node_id" in e["result"]]
    tot[k] = len(posts)
    c = collections.Counter(e["agent_id"] for e in posts)
    per_agent[k] = [c.get(f"a{i}", 0) for i in range(10)]
    post_rounds = [e["step"] // 10 + 1 for e in posts]
    cum[k] = [sum(1 for r in post_rounds if r <= R) for R in range(1, rounds + 1)]
fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 3.6), facecolor=SURF)
for ax in (a, b):
    style(ax)
x = range(len(KS))
top = max(tot.values())
a.bar(x, [tot[k] for k in KS], width=0.62, color=BLUE, edgecolor=SURF, linewidth=2, zorder=2)
for i, k in enumerate(KS):
    a.text(i, tot[k] + top * 0.02, str(tot[k]), ha="center", fontsize=8.5, color=INK2)
a.set_xticks(list(x))
a.set_xticklabels(KS)
a.axvline(BREAK, color=GRID, lw=1, ls=(0, (3, 3)), zorder=0)
a.set_xlabel("k  (topics each agent is interested in)")
a.set_ylabel("ideas posted (whole team)")
a.set_ylim(0, top * 1.18)
a.set_title("A  Ideas produced per run", loc="left", fontsize=10.5)
R = range(1, rounds + 1)
for k in KS:
    if k not in (1, 128):
        b.plot(R, cum[k], color=CTX, lw=1.2, zorder=1)
b.plot(R, cum[1], color=LIGHT, lw=2, zorder=3, label="k = 1")
b.plot(R, cum[128], color=DARK, lw=2, zorder=3, label="k = 128")
b.plot([], [], color=CTX, lw=1.2, label="k = 2 … 112")
b.text(rounds + 1.5, cum[1][-1] - 1.5, "k=1", fontsize=8.5, color=INK2, va="center")
b.text(rounds + 1.5, cum[128][-1] + 1.5, "k=128", fontsize=8.5, color=INK2, va="center")
b.set_xlim(0, rounds * 1.15)
b.set_ylim(0, top * 1.1)
b.set_xlabel("round  (each of the 10 agents acts once)")
b.set_ylabel("ideas posted so far")
b.set_title("B  Cumulative production over the run", loc="left", fontsize=10.5)
b.legend(frameon=False, loc="upper left", fontsize=8.5)
fig.text(0.01, -0.02, foot, fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig("paper/p2_forum/ksweep_production.png", dpi=200, bbox_inches="tight", facecolor=SURF)
print("wrote paper/p2_forum/ksweep.png and ksweep_production.png")
