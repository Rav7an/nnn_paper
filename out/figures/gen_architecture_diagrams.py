"""
gen_architecture_diagrams.py
Compact horizontal pipeline diagrams for E0-E5 thesis figures.

Design principles
-----------------
* Horizontal left-to-right flow (compact height, wide figure)
* Max 2-line label per box — no text overflow
* figsize ~ (11, 2.6) for linear models; (11, 3.8) for Hybrid
* White background, 300 dpi
* Colour-coded border per model (matches thesis palette)

Run from repo root:
    conda run -n nnn_win_torch python out/figures/gen_architecture_diagrams.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "out", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Model accent colours
CLR = {
    "E0": "#5d7a7e",
    "E1": "#2980b9",
    "E2": "#c0392b",
    "E3": "#8e44ad",
    "E4": "#d35400",
    "E5": "#16a085",
}

# ── drawing primitives ─────────────────────────────────────────────────────────

def _box(ax, cx, cy, bw, bh, text, color, fs=9.5):
    """Rounded box centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle="round,pad=0.05",
        facecolor=color + "22",
        edgecolor=color,
        linewidth=2.0,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy, text,
        ha="center", va="center",
        fontsize=fs, fontweight="bold",
        color="#1a1a1a", zorder=3,
        multialignment="center",
        linespacing=1.4,
    )


def _arrow(ax, x0, x1, y, color="#888888"):
    """Horizontal arrow from x0 to x1 at height y."""
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
        zorder=1,
    )


# ── linear pipeline helper ────────────────────────────────────────────────────

def linear_arch(labels, color, title, fname,
                figsize=(11, 2.6), bw=1.60, bh=0.76, gap=0.42, fs=9.5):
    """
    Draw a left-to-right pipeline of boxes.
    labels : list of strings (use newline for 2-line labels)
    """
    n    = len(labels)
    step = bw + gap
    fw   = 0.30 + n * step - gap + 0.30   # total data width

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_xlim(0, fw)
    ax.set_ylim(0, figsize[1])
    ax.axis("off")

    yc = figsize[1] / 2   # vertical centre

    for i, lbl in enumerate(labels):
        cx = 0.30 + i * step + bw / 2
        _box(ax, cx, yc, bw, bh, lbl, color, fs=fs)
        if i < n - 1:
            _arrow(ax, cx + bw / 2 + 0.04, cx + bw / 2 + gap - 0.04, yc, color)

    ax.set_title(title, fontsize=11, fontweight="bold",
                 color="#2c3e50", pad=7)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  OK  {} ({} KB)".format(fname, os.path.getsize(out) // 1024))


# ── E0: GNN ───────────────────────────────────────────────────────────────────

linear_arch(
    [
        "Sequence\n+ Structure",
        "Build Graph\n(nodes + edges)",
        "TransformerConv\nx 4  (dim 125)",
        "Set2Set\nPooling",
        "MLP Head\n(3 layers)",
        "Output\n[dH,  Tm]",
    ],
    CLR["E0"],
    "E0 - Graph Neural Network (GNN)",
    "arch_e0_gnn.png",
)

# ── E1: 1D CNN ────────────────────────────────────────────────────────────────

linear_arch(
    [
        "Input\n(7 x L)",
        "Conv1D x 4\nk = 3, 3, 5, 5",
        "Global\nAvg Pool",
        "FC Layers\n(128->64->2)",
        "Output\n[dH,  Tm]",
    ],
    CLR["E1"],
    "E1 - One-Dimensional CNN (1D CNN)",
    "arch_e1_1dcnn.png",
)

# ── E2: 2D CNN ────────────────────────────────────────────────────────────────

linear_arch(
    [
        "Input\nSequence",
        "Fold to 2D\n(6 x 3 x W)",
        "Conv2D x 3\n(k=3x3, 3x5)",
        "Spatial\nAttn Pool",
        "MLP Head\n(128->64->2)",
        "Output\n[dH,  Tm]",
    ],
    CLR["E2"],
    "E2 - 2D Folded Ladder CNN",
    "arch_e2_2dcnn.png",
)

# ── E3: SAT ───────────────────────────────────────────────────────────────────

linear_arch(
    [
        "Input\n(7 x L)",
        "Embed +\nPos Enc",
        "SAT Layer x 4\n(attn + lam*M)",
        "Masked\nMean Pool",
        "MLP Head\n(128->64->2)",
        "Output\n[dH,  Tm]",
    ],
    CLR["E3"],
    "E3 - Structure-Aware Transformer (SAT)",
    "arch_e3_sat.png",
)

# ── E4: PINN ──────────────────────────────────────────────────────────────────

linear_arch(
    [
        "Input\nSequence",
        "Fold to 2D\n(6 x 3 x W)",
        "Conv2D x 3\n+ Attn Pool",
        "MLP Head\n[dH, dS]",
        "Thermo\nLayer",
        "Output\n[dH, dS, Tm]",
    ],
    CLR["E4"],
    "E4 - Physics-Informed Neural Network (PINN)",
    "arch_e4_pinn.png",
)

# ── E5: Hybrid CNN-RNN  (branched layout) ────────────────────────────────────

def hybrid_arch():
    color  = CLR["E5"]
    c_rnn  = CLR["E1"]   # BiGRU branch colour
    c_cnn  = CLR["E2"]   # CNN branch colour

    FW, FH = 12.0, 3.8
    BW, BH = 1.55, 0.68
    GAP    = 0.42
    STEP   = BW + GAP

    fig, ax = plt.subplots(figsize=(FW, FH), facecolor="white")
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    # ── branch y-centres
    y_top = 2.85   # BiGRU branch
    y_bot = 1.10   # CNN branch
    y_mid = (y_top + y_bot) / 2

    # ── branch boxes: 3 boxes each, starting at x=0.3
    branch_labels_top = ["Input (7 x L)", "BiGRU x 2\n(hidden 64)", "128-d\nvector"]
    branch_labels_bot = ["Input (6x3xW)", "Conv2D x 3\n+ Attn Pool", "128-d\nvector"]

    last_cx = 0.0
    for i, (lt, lb) in enumerate(zip(branch_labels_top, branch_labels_bot)):
        cx = 0.30 + i * STEP + BW / 2
        _box(ax, cx, y_top, BW, BH, lt, c_rnn, fs=9)
        _box(ax, cx, y_bot, BW, BH, lb, c_cnn, fs=9)
        if i < 2:
            _arrow(ax, cx + BW/2 + 0.04, cx + BW/2 + GAP - 0.04, y_top, c_rnn)
            _arrow(ax, cx + BW/2 + 0.04, cx + BW/2 + GAP - 0.04, y_bot, c_cnn)
        last_cx = cx

    # ── merge arrows from last branch boxes to concat box
    x_concat = last_cx + BW / 2 + GAP * 1.4 + BW / 2
    ax.annotate("", xy=(x_concat - BW/2 - 0.04, y_mid + 0.12),
                xytext=(last_cx + BW/2 + 0.04, y_top),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4), zorder=1)
    ax.annotate("", xy=(x_concat - BW/2 - 0.04, y_mid - 0.12),
                xytext=(last_cx + BW/2 + 0.04, y_bot),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4), zorder=1)

    # ── merge + output boxes
    merge_labels = ["Concat\n(256-d)", "Fusion MLP\n(3 layers)", "Output\n[dH, Tm]"]
    for j, lbl in enumerate(merge_labels):
        cx = x_concat + j * STEP
        _box(ax, cx, y_mid, BW, BH, lbl, color, fs=9)
        if j < len(merge_labels) - 1:
            _arrow(ax, cx + BW/2 + 0.04, cx + BW/2 + GAP - 0.04, y_mid, color)

    # ── branch labels (left-side tags)
    ax.text(0.30, y_top + BH / 2 + 0.18, "BiGRU branch", ha="left", va="bottom",
            fontsize=8.5, color=c_rnn, fontweight="bold", style="italic")
    ax.text(0.30, y_bot - BH / 2 - 0.08, "CNN branch", ha="left", va="top",
            fontsize=8.5, color=c_cnn, fontweight="bold", style="italic")

    ax.set_title("E5 - Hybrid CNN-RNN", fontsize=11,
                 fontweight="bold", color="#2c3e50", pad=7)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "arch_e5_hybrid.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  OK  arch_e5_hybrid.png ({} KB)".format(os.path.getsize(out) // 1024))


hybrid_arch()

print("\nAll architecture diagrams saved to: {}".format(OUT_DIR))
