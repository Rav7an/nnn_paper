"""
Figure F1: 2D Folded Ladder Architecture Diagram
Thesis: Inductive Biases in Representation Learning for DNA Thermodynamic Prediction
Output: out/figures/folded_ladder_diagram.png  (300 dpi)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ─── colour palette (thesis standard + base colours) ──────────────────────────
GREY   = "#7f8c8d"   # GNN (thesis palette, reused for neutral)
BLUE   = "#3498db"   # 1D CNN
RED    = "#e74c3c"   # 2D CNN
BG     = "#f8f9fa"
LINE   = "#2c3e50"

BASE_COLOR = {
    "A": "#27ae60",   # green
    "T": "#2980b9",   # blue
    "G": "#c0392b",   # red
    "C": "#f39c12",   # orange
    "-": "#ecf0f1",   # padding / empty
}
LOOP_ALPHA   = 0.35
PAIR_COLOR   = "#8e44ad"   # purple for H-bond row
BACKBONE_COL = "#1abc9c"   # teal backbone marker

# ─── example hairpin ─────────────────────────────────────────────────────────
SEQ    = list("GCGCTTTTGCGC")   # 12-nt hairpin
STRUCT = list("((((....))))")   # 4-bp stem, 4-nt TTTT loop

# parse pairs from dot-bracket
def parse_pairs(struct):
    stack, pairs = [], {}
    for i, c in enumerate(struct):
        if c == "(": stack.append(i)
        elif c == ")":
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    return pairs

PAIRS = parse_pairs(STRUCT)
N = len(SEQ)

# fold at loop midpoint
loop_pos  = [i for i, c in enumerate(STRUCT) if c == "."]
stem_len  = sum(1 for c in STRUCT if c == "(")   # = 4
half_loop = len(loop_pos) // 2                    # = 2
width     = stem_len + half_loop                  # = 6

top_idx = list(range(stem_len)) + loop_pos[:half_loop]          # [0,1,2,3,4,5]
bot_idx = [PAIRS[i] for i in range(stem_len - 1, -1, -1)] + loop_pos[half_loop:][::-1]  # reversed 3' arm + right loop reversed

# H-bond flags: 1 if both top & bot are paired with each other, else 0
hbond = [1 if j < stem_len else 0 for j, _ in enumerate(top_idx)]

# ─── layout constants ─────────────────────────────────────────────────────────
CELL = 0.72      # cell size (inches in data units)
PAD  = 0.18      # gap between rows
ROW_H = CELL     # row height = cell width (square cells)

FIG_W = 13.0
FIG_H = 7.8

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

# ── We use two sub-axes arranged vertically:
#    ax_top  – linear sequence (row 1 of 3 in figure)
#    ax_bot  – 2D ladder + channel annotation

ax_top = fig.add_axes([0.04, 0.58, 0.92, 0.34], facecolor=BG)
ax_bot = fig.add_axes([0.04, 0.04, 0.92, 0.48], facecolor=BG)

for ax in [ax_top, ax_bot]:
    ax.set_xlim(-0.5, N + 0.5)
    ax.axis("off")

ax_top.set_ylim(-1.2, 2.8)
ax_bot.set_ylim(-2.2, 4.0)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A  –  Linear sequence with structure annotation
# ═══════════════════════════════════════════════════════════════════════════════
ax_top.text(-0.4, 2.55, "A.  Input: DNA Hairpin (linear representation)",
            fontsize=11, fontweight="bold", color=LINE, va="center")

# 5' / 3' labels
ax_top.text(-0.35, 1.5, "5′", fontsize=9, color=LINE, ha="right", va="center", style="italic")
ax_top.text(N - 0.65, 1.5, "3′", fontsize=9, color=LINE, ha="left", va="center", style="italic")

for i, (base, sym) in enumerate(zip(SEQ, STRUCT)):
    is_loop = (sym == ".")
    fc = BASE_COLOR[base]
    alpha = LOOP_ALPHA if is_loop else 1.0

    # base circle
    circ = plt.Circle((i, 1.5), 0.38, color=fc, alpha=alpha, zorder=3)
    ax_top.add_patch(circ)
    ax_top.text(i, 1.5, base, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # position label
    ax_top.text(i, 0.85, str(i), ha="center", va="center",
                fontsize=6.5, color=GREY)

    # structure character
    sym_col = RED if sym == "(" else (RED if sym == ")" else GREY)
    ax_top.text(i, 2.18, sym, ha="center", va="center",
                fontsize=9, color=sym_col, fontfamily="monospace")

# backbone line connecting adjacent bases
for i in range(N - 1):
    ax_top.plot([i + 0.38, i + 1 - 0.38], [1.5, 1.5],
                color=LINE, lw=1.2, zorder=2, alpha=0.5)

# hydrogen-bond arcs for paired bases (above)
for i, j in PAIRS.items():
    if i < j:
        mid = (i + j) / 2
        height = 1.05 + (j - i) * 0.14
        arc = mpatches.Arc((mid, 1.5), j - i, height,
                            angle=0, theta1=0, theta2=180,
                            color=PAIR_COLOR, lw=1.3, linestyle="--", zorder=2)
        ax_top.add_patch(arc)

# legend for panel A
legend_items = [
    mpatches.Patch(color=BASE_COLOR["A"], label="A"),
    mpatches.Patch(color=BASE_COLOR["T"], label="T"),
    mpatches.Patch(color=BASE_COLOR["G"], label="G"),
    mpatches.Patch(color=BASE_COLOR["C"], label="C"),
    mpatches.Patch(color=BASE_COLOR["-"], alpha=0.35, label="Loop (unpaired)"),
    plt.Line2D([0], [0], color=PAIR_COLOR, lw=1.3, linestyle="--", label="H-bond (paired)"),
]
ax_top.legend(handles=legend_items, loc="upper right",
              fontsize=7.5, framealpha=0.85, ncol=6,
              bbox_to_anchor=(1.0, 0.22))

# fold arrow annotation
fold_x = (loop_pos[half_loop - 1] + loop_pos[half_loop]) / 2
ax_top.annotate("", xy=(fold_x, 0.25), xytext=(fold_x, 0.82),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
ax_top.text(fold_x, 0.06, "fold at\nloop midpoint",
            ha="center", va="top", fontsize=7.5, color=RED)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B  –  2D Folded Ladder
# ═══════════════════════════════════════════════════════════════════════════════
ax_bot.text(-0.4, 3.75, "B.  2D Folded Ladder Encoding  (3 rows × 6 columns)",
            fontsize=11, fontweight="bold", color=LINE, va="center")

ROW_Y = {0: 2.5, 1: 1.5, 2: 0.5}   # top / hbond / bottom row y-centres
ROW_LABEL = {0: "Row 0\n5′ arm + left loop", 1: "Row 1\nH-bond indicator",
             2: "Row 2\n3′ arm (rev.) + right loop"}

# draw cell grid
for col in range(width):
    # top row
    top_b   = SEQ[top_idx[col]]
    is_loop_top = (STRUCT[top_idx[col]] == ".")
    fc = BASE_COLOR[top_b]
    rect = FancyBboxPatch((col - 0.45, ROW_Y[0] - 0.42), 0.9, 0.84,
                          boxstyle="round,pad=0.04", fc=fc,
                          alpha=LOOP_ALPHA if is_loop_top else 1.0,
                          ec=LINE, lw=0.8, zorder=3)
    ax_bot.add_patch(rect)
    ax_bot.text(col, ROW_Y[0], top_b, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    # H-bond row
    hb_val = hbond[col]
    hb_fc  = PAIR_COLOR if hb_val else "#ecf0f1"
    hb_sym = "▌" if hb_val else "○"
    rect2 = FancyBboxPatch((col - 0.45, ROW_Y[1] - 0.42), 0.9, 0.84,
                           boxstyle="round,pad=0.04", fc=hb_fc,
                           alpha=0.85 if hb_val else 0.4,
                           ec=LINE, lw=0.8, zorder=3)
    ax_bot.add_patch(rect2)
    ax_bot.text(col, ROW_Y[1], ("1" if hb_val else "0"), ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if hb_val else GREY, zorder=4)

    # bottom row
    bot_b   = SEQ[bot_idx[col]]
    is_loop_bot = (STRUCT[bot_idx[col]] == ".")
    fc_b = BASE_COLOR[bot_b]
    rect3 = FancyBboxPatch((col - 0.45, ROW_Y[2] - 0.42), 0.9, 0.84,
                           boxstyle="round,pad=0.04", fc=fc_b,
                           alpha=LOOP_ALPHA if is_loop_bot else 1.0,
                           ec=LINE, lw=0.8, zorder=3)
    ax_bot.add_patch(rect3)
    ax_bot.text(col, ROW_Y[2], bot_b, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    # column index
    ax_bot.text(col, ROW_Y[2] - 0.72, str(col), ha="center", va="center",
                fontsize=7, color=GREY)

    # vertical brace connecting paired cells
    if hbond[col]:
        ax_bot.plot([col, col], [ROW_Y[0] - 0.43, ROW_Y[1] + 0.43],
                    color=PAIR_COLOR, lw=1.5, zorder=2, linestyle=":")
        ax_bot.plot([col, col], [ROW_Y[1] - 0.43, ROW_Y[2] + 0.43],
                    color=PAIR_COLOR, lw=1.5, zorder=2, linestyle=":")

# row labels on the left
for row, label in ROW_LABEL.items():
    ax_bot.text(-0.6, ROW_Y[row], label,
                ha="right", va="center", fontsize=8, color=LINE,
                multialignment="right")

# column header
ax_bot.text(width / 2 - 0.5, ROW_Y[0] + 0.72, "Column index →",
            ha="center", va="center", fontsize=8, color=GREY)

# ── channel annotation panel (right side) ────────────────────────────────────
CHAN_X = width + 1.5
ax_bot.text(CHAN_X, 3.55, "Channel\nEncoding (C = 6)",
            ha="center", va="center", fontsize=9, fontweight="bold", color=LINE)

channels = [
    ("#27ae60", "Ch 0: is_A"),
    ("#2980b9", "Ch 1: is_T"),
    ("#c0392b", "Ch 2: is_G"),
    ("#f39c12", "Ch 3: is_C"),
    (PAIR_COLOR, "Ch 4: is_paired"),
    (BACKBONE_COL, "Ch 5: is_backbone"),
]
for k, (col_c, lbl) in enumerate(channels):
    y = 2.9 - k * 0.48
    rect = FancyBboxPatch((CHAN_X - 1.1, y - 0.18), 0.38, 0.36,
                          boxstyle="round,pad=0.03", fc=col_c, ec=LINE, lw=0.6, zorder=3)
    ax_bot.add_patch(rect)
    ax_bot.text(CHAN_X - 0.60, y, lbl, va="center", fontsize=8, color=LINE)

# ── tensor shape annotation ───────────────────────────────────────────────────
ax_bot.text(CHAN_X, -0.18,
            "Input tensor shape:\n(C=6, R=3, W≤15)",
            ha="center", va="center", fontsize=8.5, color=LINE,
            bbox=dict(boxstyle="round,pad=0.4", fc="#dfe6e9", ec=GREY, lw=0.8))

# ── sequence labels below bottom row ─────────────────────────────────────────
ax_bot.text(stem_len / 2 - 0.5, ROW_Y[2] - 1.15,
            "← stem (4 bp) →",
            ha="center", va="center", fontsize=8, color=LINE)
ax_bot.text(stem_len + half_loop / 2 - 0.5, ROW_Y[2] - 1.15,
            "← ½ loop →",
            ha="center", va="center", fontsize=8, color=GREY)

ax_bot.annotate("", xy=(stem_len - 0.45, ROW_Y[2] - 0.9),
                xytext=(-0.45, ROW_Y[2] - 0.9),
                arrowprops=dict(arrowstyle="<->", color=LINE, lw=1.1))
ax_bot.annotate("", xy=(width - 0.55, ROW_Y[2] - 0.9),
                xytext=(stem_len + 0.45, ROW_Y[2] - 0.9),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.1))

# ── global title ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.975, "Figure F1  —  2D Folded Ladder Encoding for DNA Hairpin Thermodynamics",
         ha="center", va="top", fontsize=13, fontweight="bold", color=LINE)
fig.text(0.5, 0.955,
         "Example: 12-nt hairpin  GCGCTTTTGCGC  with structure  ((((....))))  →  3 × 6 feature matrix",
         ha="center", va="top", fontsize=9, color=GREY)

import os
os.makedirs("out/figures", exist_ok=True)
plt.savefig("out/figures/folded_ladder_diagram.png", dpi=300, bbox_inches="tight",
            facecolor=BG)
print("Saved: out/figures/folded_ladder_diagram.png")
plt.close()
