"""
gen_metrics_table.py
Generates a comprehensive model comparison table for all 6 models (E0-E5)
plus the SantaLucia classical baseline.

Outputs:
  out/figures/metrics_table_full.png   -- full table (arr val + arr test + lit_uv)
  out/figures/metrics_table_collab.png -- collaboration-ready (Tm-focused, clean)
  out/metrics_table.csv                -- raw CSV for spreadsheet use

Run from repo root:
  python out/figures/gen_metrics_table.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "out", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BG   = "#f8f9fa"
LINE = "#2c3e50"
GREY = "#7f8c8d"

MODEL_COLOR = {
    "E0_GNN":    "#7f8c8d",
    "E1_1DCNN":  "#3498db",
    "E2_2DCNN":  "#e74c3c",
    "E3_SAT":    "#9b59b6",
    "E4_PINN":   "#e67e22",
    "E5_Hybrid": "#1abc9c",
    "SantaLucia":"#2ecc71",
}

MODEL_LABEL = {
    "E0_GNN":    "E0: GNN",
    "E1_1DCNN":  "E1: 1D CNN",
    "E2_2DCNN":  "E2: 2D CNN",
    "E3_SAT":    "E3: SAT",
    "E4_PINN":   "E4: PINN",
    "E5_Hybrid": "E5: Hybrid CNN-RNN",
    "SantaLucia":"SantaLucia NN (classical)",
}

# ── load all run logs ─────────────────────────────────────────────────────────
def load_log(fname):
    p = os.path.join(ROOT, "out", fname)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)

logs = {
    "E0_GNN":    load_log("gnn_run_log.json"),
    "E1_1DCNN":  load_log("cnn1d_run_log.json"),
    "E2_2DCNN":  load_log("cnn2d_run_log.json"),
    "E3_SAT":    load_log("sat_run_log.json"),
    "E4_PINN":   load_log("pinn_run_log.json"),
    "E5_Hybrid": load_log("hybrid_run_log.json"),
    "SantaLucia":load_log("nn_baseline_run_log.json"),
}

ORDER = ["E0_GNN","E1_1DCNN","E2_2DCNN","E3_SAT","E4_PINN","E5_Hybrid","SantaLucia"]

def g(log, section, key, decimals=2):
    """Safely get a metric from a log dict."""
    v = log.get(section, {}).get(key, None)
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"

def g_top(log, key, decimals=2):
    v = log.get(key, None)
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DATA ROWS
# ═══════════════════════════════════════════════════════════════════════════════

rows = []
for mid in ORDER:
    log = logs[mid]
    label = MODEL_LABEL[mid]

    # arr val
    arr_v_Tm   = g(log, "val_metrics", "Tm_mae")
    arr_v_dH   = g(log, "val_metrics", "dH_mae")
    arr_v_dG   = g(log, "val_metrics", "dG_37_mae")
    arr_v_Tm_r = g(log, "val_metrics", "Tm_r2", 3)
    arr_v_dG_r = g(log, "val_metrics", "dG_37_r2", 3)

    # arr test
    arr_t_Tm   = g(log, "test_metrics", "Tm_mae")
    arr_t_dH   = g(log, "test_metrics", "dH_mae")
    arr_t_dG   = g(log, "test_metrics", "dG_37_mae")
    arr_t_Tm_r = g(log, "test_metrics", "Tm_r2", 3)
    arr_t_dG_r = g(log, "test_metrics", "dG_37_r2", 3)

    # lit_uv (only Tm available)
    lit_Tm = g_top(log, "lit_uv_Tm_mae")

    # params
    params = log.get("n_params", None)
    params_str = f"{params:,}" if params else "—"

    rows.append({
        "ID": mid,
        "Model": label,
        # arr val
        "arr val Tm MAE": arr_v_Tm,
        "arr val dH MAE": arr_v_dH,
        "arr val ΔG MAE": arr_v_dG,
        "arr val Tm R²":  arr_v_Tm_r,
        "arr val ΔG R²":  arr_v_dG_r,
        # arr test
        "arr test Tm MAE": arr_t_Tm,
        "arr test dH MAE": arr_t_dH,
        "arr test ΔG MAE": arr_t_dG,
        "arr test Tm R²":  arr_t_Tm_r,
        "arr test ΔG R²":  arr_t_dG_r,
        # OOD
        "lit_uv Tm MAE":  lit_Tm,
        "Params":          params_str,
    })

df = pd.DataFrame(rows)

# ── save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(ROOT, "out", "metrics_table.csv")
df.drop(columns=["ID"]).to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — FULL TABLE (arr val + arr test + lit_uv)
# ═══════════════════════════════════════════════════════════════════════════════

def _float_or_nan(s):
    try: return float(s)
    except: return np.nan

def highlight_best(vals, lower_is_better=True):
    """Return index of best value."""
    nums = [_float_or_nan(v) for v in vals]
    valid = [(i, v) for i, v in enumerate(nums) if not np.isnan(v)]
    if not valid: return -1
    return min(valid, key=lambda x: x[1])[0] if lower_is_better \
           else max(valid, key=lambda x: x[1])[0]

COLS_FULL = [
    "Model",
    "arr val\nTm MAE",
    "arr val\ndH MAE",
    "arr val\nΔG MAE",
    "arr val\nTm R²",
    "arr test\nTm MAE",
    "arr test\ndH MAE",
    "arr test\nΔG MAE",
    "arr test\nTm R²",
    "lit_uv\nTm MAE",
    "Params",
]

# map column header → df column key
COL_KEY = {
    "Model":               "Model",
    "arr val\nTm MAE":     "arr val Tm MAE",
    "arr val\ndH MAE":     "arr val dH MAE",
    "arr val\nΔG MAE":     "arr val ΔG MAE",
    "arr val\nTm R²":      "arr val Tm R²",
    "arr test\nTm MAE":    "arr test Tm MAE",
    "arr test\ndH MAE":    "arr test dH MAE",
    "arr test\nΔG MAE":    "arr test ΔG MAE",
    "arr test\nTm R²":     "arr test Tm R²",
    "lit_uv\nTm MAE":      "lit_uv Tm MAE",
    "Params":              "Params",
}

LOWER_BETTER = {
    "arr val\nTm MAE":   True,
    "arr val\ndH MAE":   True,
    "arr val\nΔG MAE":   True,
    "arr val\nTm R²":    False,
    "arr test\nTm MAE":  True,
    "arr test\ndH MAE":  True,
    "arr test\nΔG MAE":  True,
    "arr test\nTm R²":   False,
    "lit_uv\nTm MAE":    True,
    "Model":             None,
    "Params":            None,
}

def make_full_table():
    n_rows = len(ORDER)
    n_cols = len(COLS_FULL)

    fig_w = max(20, n_cols * 1.85)
    fig_h = 1.4 + n_rows * 0.65
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    cell_data = []
    for mid in ORDER:
        row_dict = next(r for r in rows if r["ID"] == mid)
        cell_data.append([row_dict[COL_KEY[c]] for c in COLS_FULL])

    # determine best values per metric column
    best_idx = {}
    for ci, col in enumerate(COLS_FULL):
        lb = LOWER_BETTER.get(col, None)
        if lb is None:
            best_idx[ci] = -1
            continue
        col_vals = [cell_data[ri][ci] for ri in range(n_rows)]
        best_idx[ci] = highlight_best(col_vals, lb)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=COLS_FULL,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # style header
    for ci in range(n_cols):
        cell = tbl[0, ci]
        cell.set_facecolor(LINE)
        cell.set_text_props(color="white", fontsize=9.5, fontweight="bold")
        cell.set_edgecolor("#ffffff")
        cell.set_height(0.22)

    # style group headers
    group_spans = [
        (1, 4,  "#eaf4fb", "arr val"),
        (5, 8,  "#fef9e7", "arr test"),
        (9, 9,  "#eafaf1", "OOD (lit_uv)"),
    ]

    # style data rows
    for ri, mid in enumerate(ORDER):
        mc = MODEL_COLOR[mid]
        is_santa = (mid == "SantaLucia")
        for ci in range(n_cols):
            cell = tbl[ri+1, ci]
            cell.set_edgecolor("#d0d3d4")

            # row tint
            if ci == 0:
                cell.set_facecolor(mc + "33")
                cell.set_text_props(fontweight="bold", color=LINE, fontsize=9.5)
            elif ci in range(1, 5):   # arr val group
                cell.set_facecolor("#eaf4fb" if not is_santa else "#d5f5e3")
            elif ci in range(5, 9):   # arr test group
                cell.set_facecolor("#fef9e7" if not is_santa else "#d5f5e3")
            elif ci == 9:             # lit_uv
                cell.set_facecolor("#eafaf1" if not is_santa else "#d5f5e3")
            else:
                cell.set_facecolor(BG)

            # highlight best
            if best_idx.get(ci, -1) == ri:
                cell.set_facecolor("#2ecc71" + "55")
                cell.set_text_props(fontweight="bold", color="#1a6b3a")

            cell.set_height(0.13)

    fig.suptitle(
        "Model Performance Comparison — All 6 Neural Network Architectures + SantaLucia Classical Baseline\n"
        "Training: arr partition only  |  OOD evaluation: lit_uv (348 literature UV-melt duplexes)  |  "
        "MAE in °C (Tm) or kcal/mol (dH, ΔG)  |  Green cell = best per column",
        fontsize=11, fontweight="bold", color=LINE, y=0.99, va="top"
    )

    # group label annotations below title
    ax.text(0.22, 1.015, "arr validation set", transform=ax.transAxes,
            ha="center", fontsize=9, color="#1a5276", fontweight="bold")
    ax.text(0.57, 1.015, "arr test set", transform=ax.transAxes,
            ha="center", fontsize=9, color="#784212", fontweight="bold")
    ax.text(0.84, 1.015, "OOD: lit_uv", transform=ax.transAxes,
            ha="center", fontsize=9, color="#1e8449", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "metrics_table_full.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: metrics_table_full.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — COLLABORATION TABLE (Tm-focused, clean, easy to read)
# ═══════════════════════════════════════════════════════════════════════════════

def make_collab_table():
    """
    Simplified 3-section table for sharing with collaborators.
    Focuses on Tm MAE (primary metric) and ΔG MAE.
    Rows: all 6 models + SantaLucia.
    """
    COLS = [
        "Model",
        "Params",
        "arr val Tm MAE (°C)",
        "arr val ΔG MAE (kcal/mol)",
        "arr val ΔG R²",
        "arr test Tm MAE (°C)",
        "arr test ΔG MAE (kcal/mol)",
        "lit_uv Tm MAE (°C)\n[OOD]",
    ]
    COL_K = {
        "Model":                         "Model",
        "Params":                        "Params",
        "arr val Tm MAE (°C)":           "arr val Tm MAE",
        "arr val ΔG MAE (kcal/mol)":     "arr val ΔG MAE",
        "arr val ΔG R²":                 "arr val ΔG R²",
        "arr test Tm MAE (°C)":          "arr test Tm MAE",
        "arr test ΔG MAE (kcal/mol)":    "arr test ΔG MAE",
        "lit_uv Tm MAE (°C)\n[OOD]":    "lit_uv Tm MAE",
    }
    LB = {
        "Model": None, "Params": None,
        "arr val Tm MAE (°C)": True,
        "arr val ΔG MAE (kcal/mol)": True,
        "arr val ΔG R²": False,
        "arr test Tm MAE (°C)": True,
        "arr test ΔG MAE (kcal/mol)": True,
        "lit_uv Tm MAE (°C)\n[OOD]": True,
    }

    n_rows = len(ORDER)
    n_cols = len(COLS)

    cell_data = []
    for mid in ORDER:
        row_dict = next(r for r in rows if r["ID"] == mid)
        cell_data.append([row_dict[COL_K[c]] for c in COLS])

    best_idx = {}
    for ci, col in enumerate(COLS):
        lb = LB.get(col, None)
        if lb is None:
            best_idx[ci] = -1
            continue
        col_vals = [cell_data[ri][ci] for ri in range(n_rows)]
        best_idx[ci] = highlight_best(col_vals, lb)

    fig_w = 18
    fig_h = 1.6 + n_rows * 0.72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=COLS,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    col_widths = [0.25, 0.08, 0.12, 0.14, 0.09, 0.12, 0.14, 0.14]
    for ci, w in enumerate(col_widths):
        for ri in range(n_rows + 1):
            tbl[ri, ci].set_width(w)

    # header row
    header_colors = ["#2c3e50","#2c3e50","#1a5276","#1a5276","#1a5276",
                     "#784212","#784212","#1e8449"]
    for ci, hc in enumerate(header_colors):
        cell = tbl[0, ci]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontsize=10, fontweight="bold")
        cell.set_edgecolor("#ffffff")
        cell.set_height(0.20)

    # data rows
    SANTA_TINT = "#d5f5e3"
    for ri, mid in enumerate(ORDER):
        mc = MODEL_COLOR[mid]
        is_santa = (mid == "SantaLucia")
        for ci in range(n_cols):
            cell = tbl[ri+1, ci]
            cell.set_height(0.14)
            cell.set_edgecolor("#bdc3c7")

            if ci == 0:
                cell.set_facecolor(mc + "44")
                cell.set_text_props(fontweight="bold", fontsize=10.5, color=LINE)
            elif ci in [2, 3, 4]:   # val
                cell.set_facecolor(SANTA_TINT if is_santa else "#eaf4fb")
            elif ci in [5, 6]:      # test
                cell.set_facecolor(SANTA_TINT if is_santa else "#fef9e7")
            elif ci == 7:           # ood
                cell.set_facecolor(SANTA_TINT if is_santa else "#eafaf1")
            else:
                cell.set_facecolor(BG)

            # best highlight
            if best_idx.get(ci, -1) == ri:
                cell.set_facecolor("#27ae60" + "55")
                cell.set_text_props(fontweight="bold", color="#1a6b3a", fontsize=11)

    # separator line between DL and SantaLucia
    santa_row = n_rows  # last row (1-indexed)
    for ci in range(n_cols):
        tbl[santa_row, ci].visible_edges = "open"

    fig.suptitle(
        "DNA Thermodynamic Property Prediction — Model Comparison Table\n"
        "Anant Shrenik Patil · UCR MS Thesis (2026)  |  "
        "All models trained on arr (Array Melt hairpins)  |  "
        "lit_uv = 348 literature UV-melt duplexes (OOD)  |  "
        "Green = best per column",
        fontsize=12, fontweight="bold", color=LINE, y=0.995, va="top",
    )

    # footnotes
    fig.text(0.02, 0.01,
             "MAE = Mean Absolute Error  |  R² = coefficient of determination  |  "
             "ov (2,775 duplex sequences) and uv (19 sequences) evaluations not shown — pending notebook run\n"
             "SantaLucia NN: classical nearest-neighbor model (SantaLucia & Hicks 2004) — no training; "
             "evaluated on lit_uv only (duplex-specific model)",
             fontsize=8, color=GREY, va="bottom")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "metrics_table_collab.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: metrics_table_collab.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Building metrics tables...")
    make_full_table()
    make_collab_table()

    print("\n=== Raw numbers (quick check) ===")
    print(df[["Model","arr val Tm MAE","arr test Tm MAE","lit_uv Tm MAE","Params"]].to_string(index=False))
    print(f"\nNote: ov and uv evaluations not in stored run logs — run 04_Evaluate_Models.ipynb to add them.")
