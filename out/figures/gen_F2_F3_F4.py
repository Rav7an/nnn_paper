"""
gen_F2_F3_F4.py  —  Combined thesis figures for all 6 experiments (E0–E5)

Outputs
-------
F2  out/figures/convergence_curves.png   — overlaid val MAE convergence (E3/E4/E5) +
                                           reference lines for E0/E1/E2
F3  out/figures/scatter_grid.png         — 6×3 predicted vs measured (E0–E5, all models)
F4  out/figures/lit_uv_mae_bar.png       — out-of-distribution Tm MAE (all 6 models)
F7  out/figures/final_comparison_table.png — summary table image (all 6 × all metrics)

Run from repo root:
    cd "e:\\project ms\\nnn_paper"
    conda run -n nnn_win_torch python out/figures/gen_F2_F3_F4.py
"""

import os, json, math, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import r2_score

sns.set_style('whitegrid')

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(ROOT, 'out', 'figures')
DATA_CSV = os.path.join(ROOT, 'data', 'models', 'raw', 'combined_dataset.csv')
SPLIT_JSON = os.path.join(ROOT, 'data', 'models', 'raw', 'combined_data_split.json')

COLORS = {
    'E0_GNN':    '#7f8c8d',
    'E1_1DCNN':  '#3498db',
    'E2_2DCNN':  '#e74c3c',
    'E3_SAT':    '#9b59b6',
    'E4_PINN':   '#e67e22',
    'E5_Hybrid': '#1abc9c',
}
LABELS = {
    'E0_GNN':    'E0: GNN',
    'E1_1DCNN':  'E1: 1D CNN',
    'E2_2DCNN':  'E2: 2D CNN',
    'E3_SAT':    'E3: SAT',
    'E4_PINN':   'E4: PINN',
    'E5_Hybrid': 'E5: Hybrid CNN-RNN',
}

# ── Load history for E0/E1/E2 from standardized run output JSONs ──────────────
baseline_hist_files = {
    'E0_GNN':   os.path.join(ROOT, 'out', 'gnn_history.json'),
    'E1_1DCNN': os.path.join(ROOT, 'out', 'cnn1d_history.json'),
    'E2_2DCNN': os.path.join(ROOT, 'out', 'cnn2d_history.json'),
}
baseline_hist = {}
for k, fp in baseline_hist_files.items():
    if os.path.exists(fp):
        with open(fp) as f:
            baseline_hist[k] = json.load(f)

# ── Baseline final val metrics from run_log JSONs (E0/E1/E2) ─────────────────
def _load_run_log_metrics(fn):
    fp = os.path.join(ROOT, 'out', fn)
    if not os.path.exists(fp): return {}
    with open(fp) as f:
        d = json.load(f)
    vm = d.get('val_metrics', {})
    return {'dH_mae': vm.get('dH_mae',0), 'Tm_mae': vm.get('Tm_mae',0),
            'dG_mae': vm.get('dG_37_mae',0), 'dH_rmse': vm.get('dH_rmse',0),
            'Tm_rmse': vm.get('Tm_rmse',0), 'dG_rmse': vm.get('dG_37_rmse',0),
            'dH_r2': vm.get('dH_r2',0), 'Tm_r2': vm.get('Tm_r2',0),
            'dG_r2': vm.get('dG_37_r2',0),
            'lit_uv_Tm_mae': d.get('lit_uv_Tm_mae', float('nan'))}

BASELINES = {
    'E0_GNN':   _load_run_log_metrics('gnn_run_log.json'),
    'E1_1DCNN': _load_run_log_metrics('cnn1d_run_log.json'),
    'E2_2DCNN': _load_run_log_metrics('cnn2d_run_log.json'),
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load data + sumstats (same logic as each notebook) ────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_CSV, index_col='SEQID')
df.sort_index(inplace=True)
with open(SPLIT_JSON) as f:
    split = json.load(f)
train_df = df.loc[split['train_ind']].dropna(subset=['dH', 'Tm'])
sumstats = {
    'dH_min': float(train_df['dH'].min()), 'dH_max': float(train_df['dH'].max()),
    'Tm_min': float(train_df['Tm'].min()), 'Tm_max': float(train_df['Tm'].max()),
    'dS_min': -0.25, 'dS_max': -0.002,
}

def normalize(v, mn, mx):   return (v - mn) / (mx - mn)
def unnormalize(v, mn, mx): return v * (mx - mn) + mn


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS  (minimal copies for checkpoint loading)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_LEN   = 24
MAX_WIDTH = 15
NT_MAP    = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
BASES     = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

# ── E3: SAT ───────────────────────────────────────────────────────────────────
class StructureBiasedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, lambda_init=1.0):
        super().__init__()
        self.d_model = d_model; self.nhead = nhead; self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.lambda_bias = nn.Parameter(torch.tensor(float(lambda_init)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, M, pad_mask=None):
        B, L, _ = x.shape; H, D = self.nhead, self.head_dim
        Q = self.W_q(x).view(B,L,H,D).transpose(1,2)
        K = self.W_k(x).view(B,L,H,D).transpose(1,2)
        V = self.W_v(x).view(B,L,H,D).transpose(1,2)
        logits = torch.matmul(Q, K.transpose(-2,-1)) / self.scale
        logits = logits + self.lambda_bias * M.unsqueeze(1)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out  = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, L, self.d_model)
        return self.W_o(out), attn

class SATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout=0.1, lambda_init=1.0):
        super().__init__()
        self.attn  = StructureBiasedAttention(d_model, nhead, dropout, lambda_init)
        self.ff    = nn.Sequential(nn.Linear(d_model,ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim,d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, M, pad_mask=None):
        attn_out, _ = self.attn(self.norm1(x), M, pad_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class StructureAwareTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=128, nhead=8, num_layers=4,
                 ff_dim=256, dropout=0.1, max_len=MAX_LEN, lambda_init=1.0):
        super().__init__()
        self.d_model    = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)
        self.pos_drop   = nn.Dropout(dropout)
        self.layers     = nn.ModuleList([SATransformerLayer(d_model,nhead,ff_dim,dropout,lambda_init) for _ in range(num_layers)])
        self.norm       = nn.LayerNorm(d_model)
        self.head       = nn.Sequential(nn.Linear(d_model,d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2,2))
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x, M):
        B, L, _ = x.shape
        pos      = torch.arange(L, device=x.device).unsqueeze(0).expand(B,-1)
        pad_mask = (x.sum(-1) == 0)
        h = self.pos_drop(self.input_proj(x) + self.pos_embed(pos))
        for layer in self.layers:
            h = layer(h, M, pad_mask)
        h = self.norm(h)
        not_pad = (~pad_mask).float().unsqueeze(-1)
        h = (h * not_pad).sum(1) / not_pad.sum(1).clamp(min=1)
        return self.head(h)

# ── E4: PINN ──────────────────────────────────────────────────────────────────
class AttentionPool2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_channels,64,1), nn.Tanh(), nn.Conv2d(64,1,1))

    def forward(self, x):
        B, C, H, W = x.shape
        w = torch.softmax(self.attn(x).view(B,1,-1), dim=-1)
        return (x.view(B,C,-1) * w).sum(-1)

class ThermodynamicsLayer(nn.Module):
    def __init__(self, ss):
        super().__init__()
        for k in ['dH_min','dH_max','dS_min','dS_max','Tm_min','Tm_max']:
            self.register_buffer(k, torch.tensor(ss[k], dtype=torch.float32))

    def forward(self, dH_n, dS_n):
        dH = dH_n * (self.dH_max - self.dH_min) + self.dH_min
        dS = dS_n * (self.dS_max - self.dS_min) + self.dS_min
        dS_safe = torch.where(dS < 0, dS.clamp(max=-1e-4), dS.clamp(min=1e-4))
        Tm_C  = dH / dS_safe - 273.15
        return ((Tm_C - self.Tm_min) / (self.Tm_max - self.Tm_min)).clamp(-2., 3.)

class PINN_2DCNN(nn.Module):
    def __init__(self, ss, in_channels=6, dropout=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels,64,(3,3),padding=(1,1)), nn.BatchNorm2d(64),  nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64,128,(3,3),padding=(1,1)),         nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(128,128,(3,5),padding=(1,2)),        nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
        )
        self.attn_pool = AttentionPool2d(128)
        self.head      = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,2))
        self.thermo    = ThermodynamicsLayer(ss)

    def forward(self, x):
        feat = self.attn_pool(self.backbone(x))
        out  = self.head(feat)
        return {'dH_norm': out[:,0], 'dS_norm': out[:,1], 'Tm_norm': self.thermo(out[:,0], out[:,1])}

# ── E5: Hybrid ────────────────────────────────────────────────────────────────
class BiGRUBranch(nn.Module):
    def __init__(self, input_dim=7, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden, n_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if n_layers>1 else 0.)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        not_pad = (x.sum(-1) != 0).float().unsqueeze(-1)
        out, _  = self.gru(x)
        out     = out * not_pad
        lengths = not_pad.sum(1).clamp(min=1)
        return self.drop(out.sum(1) / lengths)

class CNN2DBranch(nn.Module):
    def __init__(self, in_channels=6, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,64,(3,3),padding=(1,1)), nn.BatchNorm2d(64),  nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64,128,(3,3),padding=(1,1)),         nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(128,128,(3,5),padding=(1,2)),        nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
        )
        self.pool = AttentionPool2d(128)

    def forward(self, x):
        return self.pool(self.conv(x))

class HybridCNNRNN(nn.Module):
    def __init__(self, seq_input_dim=7, gru_hidden=64, gru_layers=2, cnn_in_channels=6, dropout=0.2):
        super().__init__()
        self.rnn_branch  = BiGRUBranch(seq_input_dim, gru_hidden, gru_layers, dropout)
        self.cnn_branch  = CNN2DBranch(cnn_in_channels, dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(gru_hidden*2+128,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,64),               nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,2),
        )

    def forward(self, x_1d, x_2d):
        return self.fusion_head(torch.cat([self.rnn_branch(x_1d), self.cnn_branch(x_2d)], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING HELPERS FOR LIT_UV EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

STRUCTS = {'(': 0, ')': 1, '.': 2}

def _parse_litUV_row(row):
    """Return (seq_str, struct_str) handling list-format RefSeq and '+' struct."""
    refseq = str(row['RefSeq'])
    struct = str(row['TargetStruct'])
    if '[' in refseq:
        try:   strands = eval(refseq)
        except: strands = [refseq]
        seq = ''.join(strands)
    else:
        seq = refseq.replace('+','')
    struct = struct.replace('+','').replace(' ','')
    L = min(len(seq), len(struct))
    return seq[:L], struct[:L]

def encode_sat(seq, struct):
    """1D encoding (L,7) + bias matrix (L,L) for SAT — truncated to MAX_LEN."""
    L = min(len(seq), MAX_LEN)
    x = np.zeros((MAX_LEN, 7), dtype=np.float32)
    for i in range(L):
        if seq[i].upper() in BASES:    x[i, BASES[seq[i].upper()]] = 1.0
        if struct[i] in STRUCTS:       x[i, 4 + STRUCTS[struct[i]]] = 1.0
    M = np.zeros((MAX_LEN, MAX_LEN), dtype=np.float32)
    for i in range(L-1):
        M[i,i+1] = M[i+1,i] = 1.0
    stack = []
    for i, c in enumerate(struct[:L]):
        if c == '(':  stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            M[i,j] = M[j,i] = 1.0
    return torch.tensor(x).unsqueeze(0), torch.tensor(M).unsqueeze(0)   # (1,L,7), (1,L,L)

def encode_2d_hairpin(seq, struct, max_width=MAX_WIDTH):
    n_stem = struct.count('('); n_loop = struct.count('.')
    half_loop = n_loop // 2; has_mid = (n_loop % 2 == 1)
    fold_len  = n_stem + half_loop
    top_seq   = seq[:fold_len]
    mid_nt    = seq[fold_len] if has_mid and fold_len < len(seq) else None
    bot_seq   = seq[fold_len + (1 if has_mid else 0):][::-1]
    hbond     = [1.]*n_stem + [0.]*half_loop
    t = np.zeros((6,3,max_width), dtype=np.float32)
    for i,nt in enumerate(top_seq):
        if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()],0,i]=1.
    for i,nt in enumerate(bot_seq):
        if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()],2,i]=1.
    for i,h in enumerate(hbond): t[4,1,i]=h
    bb=fold_len
    if bb<max_width: t[5,:,bb]=1.
    if has_mid and mid_nt and mid_nt.upper() in NT_MAP and bb<max_width:
        t[NT_MAP[mid_nt.upper()],1,bb]=1.
    return t

def encode_2d_duplex(s1, s2, max_width=MAX_WIDTH):
    t = np.zeros((6,3,max_width), dtype=np.float32)
    for i,nt in enumerate(s1):
        if i<max_width and nt.upper() in NT_MAP: t[NT_MAP[nt.upper()],0,i]=1.
    for i,nt in enumerate(s2[::-1]):
        if i<max_width and nt.upper() in NT_MAP: t[NT_MAP[nt.upper()],2,i]=1.
    for i in range(min(len(s1),max_width)): t[4,1,i]=1.
    return t

def encode_2d_row(refseq_raw, struct_raw, max_width=MAX_WIDTH):
    struct = str(struct_raw)
    if '[' in str(refseq_raw):
        try:   strands = eval(str(refseq_raw))
        except: strands = [str(refseq_raw)]
        s1, s2 = strands[0], strands[1] if len(strands)>1 else strands[0]
        return encode_2d_duplex(s1, s2, max_width)
    elif '+' in struct:
        plus = struct.index('+')
        seq  = str(refseq_raw)
        return encode_2d_duplex(seq[:plus], seq[plus+1:], max_width)
    else:
        seq = str(refseq_raw).replace('+','')
        st  = struct.replace('+','')
        return encode_2d_hairpin(seq, st, max_width)

def encode_1d_row(refseq_raw, struct_raw, max_len=MAX_LEN):
    seq_map = {'A':0,'T':1,'C':2,'G':3}; str_map = {'(':0,')':1,'.':2}
    if '[' in str(refseq_raw):
        try:   strands = eval(str(refseq_raw))
        except: strands = [str(refseq_raw)]
        seq = ''.join(strands)
    else:
        seq = str(refseq_raw).replace('+','')
    struct = str(struct_raw).replace('+','').replace(' ','')
    L = min(len(seq), max_len)
    x = np.zeros((max_len, 7), dtype=np.float32)
    for i in range(L):
        if seq[i].upper() in seq_map: x[i, seq_map[seq[i].upper()]] = 1.
        if i<len(struct) and struct[i] in str_map: x[i, 4+str_map[struct[i]]] = 1.
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# LIT_UV EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

lit_df = df[df['dataset'] == 'lit_uv'].copy()
print(f'lit_uv sequences: {len(lit_df)}  (Tm-only)')

def eval_lit_uv_SAT(model, ss):
    model.eval()
    Tm_preds = []
    with torch.no_grad():
        for _, row in lit_df.iterrows():
            seq, struct = _parse_litUV_row(row)
            x, M = encode_sat(seq, struct)
            x, M = x.to(device), M.to(device)
            out  = model(x, M)   # (1,2) [dH_norm, Tm_norm]
            Tm_n = out[0,1].item()
            Tm_p = unnormalize(Tm_n, ss['Tm_min'], ss['Tm_max'])
            Tm_preds.append(Tm_p)
    Tm_true = lit_df['Tm'].values
    mae = float(np.mean(np.abs(np.array(Tm_preds) - Tm_true)))
    return round(mae, 3)

def eval_lit_uv_PINN(model, ss):
    model.eval()
    Tm_preds = []
    with torch.no_grad():
        for _, row in lit_df.iterrows():
            x2d = encode_2d_row(row['RefSeq'], row['TargetStruct'])
            x   = torch.tensor(x2d).unsqueeze(0).to(device)
            out = model(x)
            Tm_n = out['Tm_norm'][0].item()
            Tm_p = unnormalize(Tm_n, ss['Tm_min'], ss['Tm_max'])
            Tm_preds.append(Tm_p)
    Tm_true = lit_df['Tm'].values
    return round(float(np.mean(np.abs(np.array(Tm_preds) - Tm_true))), 3)

def eval_lit_uv_Hybrid(model, ss):
    model.eval()
    Tm_preds = []
    with torch.no_grad():
        for _, row in lit_df.iterrows():
            x1d = encode_1d_row(row['RefSeq'], row['TargetStruct'])
            x2d = encode_2d_row(row['RefSeq'], row['TargetStruct'])
            x1  = torch.tensor(x1d).unsqueeze(0).to(device)
            x2  = torch.tensor(x2d).unsqueeze(0).to(device)
            out = model(x1, x2)
            Tm_n = out[0,1].item()
            Tm_p = unnormalize(Tm_n, ss['Tm_min'], ss['Tm_max'])
            Tm_preds.append(Tm_p)
    Tm_true = lit_df['Tm'].values
    return round(float(np.mean(np.abs(np.array(Tm_preds) - Tm_true))), 3)


# Load checkpoints
print('\nLoading checkpoints...')
sat_ckpt    = os.path.join(ROOT, 'MyExperiments', 'SAT',    'models', 'best_sat_model.pt')
pinn_ckpt   = os.path.join(ROOT, 'MyExperiments', 'PINN',   'models', 'best_pinn_model.pt')
hybrid_ckpt = os.path.join(ROOT, 'MyExperiments', 'Hybrid', 'models', 'best_hybrid_model.pt')

lit_uv_results = {k: v['lit_uv_Tm_mae'] for k, v in BASELINES.items()}   # E0/E1/E2

for name, ckpt, ModelClass, eval_fn, kwargs in [
    ('E3_SAT',    sat_ckpt,    StructureAwareTransformer, eval_lit_uv_SAT,
     dict(input_dim=7, d_model=128, nhead=8, num_layers=4, ff_dim=256, dropout=0.1, max_len=MAX_LEN)),
    ('E4_PINN',   pinn_ckpt,   PINN_2DCNN, eval_lit_uv_PINN,
     dict(ss=sumstats, in_channels=6, dropout=0.2)),
    ('E5_Hybrid', hybrid_ckpt, HybridCNNRNN, eval_lit_uv_Hybrid,
     dict(seq_input_dim=7, gru_hidden=64, gru_layers=2, cnn_in_channels=6, dropout=0.2)),
]:
    if os.path.exists(ckpt):
        m = ModelClass(**kwargs).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        mae = eval_fn(m, sumstats)
        lit_uv_results[name] = mae
        print(f'  {name} lit_uv Tm MAE: {mae}')
    else:
        print(f'  {name}: checkpoint not found at {ckpt}')
        lit_uv_results[name] = None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F2 — Convergence Curves
# ═══════════════════════════════════════════════════════════════════════════════
print('\nGenerating F2...')

hist = {
    'E3_SAT':    json.load(open(os.path.join(ROOT,'out','sat_history.json'))),
    'E4_PINN':   json.load(open(os.path.join(ROOT,'out','pinn_history.json'))),
    'E5_Hybrid': json.load(open(os.path.join(ROOT,'out','hybrid_history.json'))),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#f8f9fa')
targets = [
    ('val_dH_mae',  'Val ΔH MAE (kcal/mol)', 'ΔH'),
    ('val_Tm_mae',  'Val Tm MAE (°C)',         'Tm'),
    ('val_dG_mae',  'Val ΔG₃₇ MAE (kcal/mol)', 'ΔG₃₇'),
]
bl_keys = [('dH_mae','E0_GNN'), ('dH_mae','E1_1DCNN'), ('dH_mae','E2_2DCNN'),
           ('Tm_mae','E0_GNN'), ('Tm_mae','E1_1DCNN'), ('Tm_mae','E2_2DCNN'),
           ('dG_mae','E0_GNN'), ('dG_mae','E1_1DCNN'), ('dG_mae','E2_2DCNN')]

for ax, (key, ylabel, title) in zip(axes, targets):
    # E0/E1/E2 — full curves from standardized history JSONs
    for exp_id, h in baseline_hist.items():
        if key in h:
            ep = range(1, len(h[key]) + 1)
            ax.plot(ep, h[key], color=COLORS[exp_id], lw=2,
                    label=LABELS[exp_id], alpha=0.9)

    # E3/E4/E5 — full curves
    for exp_id, h in hist.items():
        ep = range(1, len(h[key]) + 1)
        ax.plot(ep, h[key], color=COLORS[exp_id], lw=2,
                label=LABELS[exp_id], alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    sns.despine(ax=ax)

# Single shared legend
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.12), frameon=True)
fig.suptitle('Figure F2 — Validation MAE Convergence (All 6 Models)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'convergence_curves.png'), dpi=300,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print('  Saved: convergence_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F3 — Scatter Grids (E0–E5, 6 rows × 3 cols)
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating F3...')

AXIS_LIMITS = {'dH': (-55,-5), 'Tm': (20,60), 'dG_37': (-7,5)}
csv_map = {
    'E0_GNN':    os.path.join(ROOT, 'out', 'gnn_val_eval.csv'),
    'E1_1DCNN':  os.path.join(ROOT, 'out', 'cnn1d_val_eval.csv'),
    'E2_2DCNN':  os.path.join(ROOT, 'out', 'cnn2d_val_eval.csv'),
    'E3_SAT':    os.path.join(ROOT, 'out', 'sat_val_eval.csv'),
    'E4_PINN':   os.path.join(ROOT, 'out', 'pinn_val_eval.csv'),
    'E5_Hybrid': os.path.join(ROOT, 'out', 'hybrid_val_eval.csv'),
}
all_exps = ['E0_GNN', 'E1_1DCNN', 'E2_2DCNN', 'E3_SAT', 'E4_PINN', 'E5_Hybrid']

fig, axes = plt.subplots(6, 3, figsize=(18, 34), facecolor='#f8f9fa')
plt.subplots_adjust(hspace=0.35, wspace=0.30)

COL_TITLES = {'dH': 'ΔH (kcal/mol)', 'Tm': 'Tm (°C)', 'dG_37': 'ΔG₃₇ (kcal/mol)'}

for row_i, exp_id in enumerate(all_exps):
    ev = pd.read_csv(csv_map[exp_id])
    for col_i, (pred_col, true_col, tag, unit) in enumerate([
        ('dH_pred', 'dH_true', 'dH', 'kcal/mol'),
        ('Tm_pred', 'Tm_true', 'Tm', '°C'),
        ('dG_pred', 'dG_true', 'dG_37', 'kcal/mol'),
    ]):
        ax = axes[row_i, col_i]
        p = ev[pred_col].values
        t = ev[true_col].values
        lim = AXIS_LIMITS[tag]
        mask = np.isfinite(p) & np.isfinite(t)
        ax.scatter(t[mask], p[mask], s=6, alpha=0.40, color=COLORS[exp_id], rasterized=True)
        ax.plot(lim, lim, 'k--', alpha=0.30, lw=1.8)
        mae = np.mean(np.abs(p[mask] - t[mask]))
        r2 = r2_score(t[mask], p[mask])
        ax.text(0.04, 0.96, f'MAE = {mae:.3f}\nR²  = {r2:.3f}',
                transform=ax.transAxes, fontsize=11, va='top', linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88,
                          edgecolor='#cccccc', lw=0.8))
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.tick_params(axis='both', labelsize=10)
        if row_i == 5:
            ax.set_xlabel(f'Measured {tag} ({unit})', fontsize=12, labelpad=6)
        if col_i == 0:
            ax.set_ylabel(f'{LABELS[exp_id]}\nPredicted {unit}', fontsize=11, labelpad=8)
        if row_i == 0:
            ax.set_title(COL_TITLES[tag], fontsize=14, fontweight='bold', pad=10)
        sns.despine(ax=ax)

fig.suptitle('Figure F3 — Predicted vs Measured (Validation, All 6 Models)',
             fontsize=16, fontweight='bold', y=1.002)
plt.savefig(os.path.join(OUT_DIR, 'scatter_grid.png'), dpi=300,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print('  Saved: scatter_grid.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F4 — Out-of-Distribution Generalization (lit_uv Tm MAE)
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating F4...')

order   = ['E0_GNN','E1_1DCNN','E2_2DCNN','E3_SAT','E4_PINN','E5_Hybrid']
f4_vals = [lit_uv_results.get(k) for k in order]
f4_cols = [COLORS[k] for k in order]
f4_lbls = [LABELS[k] for k in order]

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f8f9fa')
xs = np.arange(len(order))
bars = ax.bar(xs, [v if v is not None else 0 for v in f4_vals],
              color=f4_cols, width=0.55, edgecolor='white', linewidth=1.3)
for bar, val in zip(bars, f4_vals):
    if val is not None:
        ax.text(bar.get_x()+bar.get_width()/2, val+0.015,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(f4_lbls, fontsize=9)
ax.set_ylabel('Tm MAE on lit_uv (°C)', fontsize=11)
ax.set_title('Figure F4 — Zero-Shot Generalization to Literature UV Dataset\n'
             '(348 sequences unseen during training)', fontsize=11, fontweight='bold')
ax.axhline(min(v for v in f4_vals if v), color='green', lw=1.2, linestyle=':', alpha=0.7,
           label=f'Best: {min(v for v in f4_vals if v):.2f} °C')
ax.legend(fontsize=9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'lit_uv_mae_bar.png'), dpi=300,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print('  Saved: lit_uv_mae_bar.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F7 — Final Comparison Table (all 6 models × all metrics)
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating F7...')

# Load E3/E4/E5 run logs for full metric set
run_logs = {}
for exp_id, fname in [('E3_SAT','sat_run_log.json'),('E4_PINN','pinn_run_log.json'),('E5_Hybrid','hybrid_run_log.json')]:
    p = os.path.join(ROOT,'out',fname)
    if os.path.exists(p):
        run_logs[exp_id] = json.load(open(p))

table_data = []
for exp_id in order:
    row = {'Model': LABELS[exp_id]}
    if exp_id in BASELINES:
        b = BASELINES[exp_id]
        row.update({
            'dH MAE': f"{b['dH_mae']:.2f}", 'dH R²': f"{b['dH_r2']:.3f}",
            'Tm MAE': f"{b['Tm_mae']:.2f}", 'Tm R²':  f"{b['Tm_r2']:.3f}",
            'dG MAE': f"{b['dG_mae']:.2f}", 'dG R²':  f"{b['dG_r2']:.3f}",
            'lit_uv Tm': f"{b['lit_uv_Tm_mae']:.2f}",
        })
    elif exp_id in run_logs:
        vm = run_logs[exp_id]['val_metrics']
        row.update({
            'dH MAE': f"{vm['dH_mae']:.2f}", 'dH R²': f"{vm['dH_r2']:.3f}",
            'Tm MAE': f"{vm['Tm_mae']:.2f}", 'Tm R²':  f"{vm['Tm_r2']:.3f}",
            'dG MAE': f"{vm['dG_37_mae']:.2f}", 'dG R²': f"{vm['dG_37_r2']:.3f}",
            'lit_uv Tm': f"{lit_uv_results.get(exp_id, 'N/A')}",
        })
    table_data.append(row)

cols = ['Model','dH MAE','dH R²','Tm MAE','Tm R²','dG MAE','dG R²','lit_uv Tm']
cell_vals = [[r.get(c,'—') for c in cols] for r in table_data]

fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='#f8f9fa')
ax.axis('off')
tbl = ax.table(cellText=cell_vals, colLabels=cols,
               cellLoc='center', loc='center',
               bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    else:
        exp_id_row = order[r-1]
        cell.set_facecolor(COLORS[exp_id_row] + '22')   # light tint
    if c == 0 and r > 0:
        cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('#cccccc')

# Highlight best values in each metric column
metric_cols = list(range(1, len(cols)))
for ci in metric_cols:
    vals = []
    for ri in range(1, len(table_data)+1):
        try:   vals.append((ri, float(cell_vals[ri-1][ci])))
        except: pass
    if not vals: continue
    is_r2 = 'R²' in cols[ci]
    best_r, _ = max(vals, key=lambda x: x[1]) if is_r2 else min(vals, key=lambda x: x[1])
    tbl[best_r, ci].set_facecolor('#2ecc71' + '55')
    tbl[best_r, ci].set_text_props(fontweight='bold')

fig.suptitle('Figure F7 — Final Model Comparison (Validation Set + lit_uv Generalization)\n'
             'Green = best per column. Metrics on arr validation set except lit_uv Tm.',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'final_comparison_table.png'), dpi=300,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print('  Saved: final_comparison_table.png')

print('\n=== All figures complete ===')
print(f'Output directory: {OUT_DIR}')
for fn in ['convergence_curves.png','scatter_grid.png','lit_uv_mae_bar.png','final_comparison_table.png']:
    fp = os.path.join(OUT_DIR, fn)
    if os.path.exists(fp):
        print(f'  OK  {fn}  ({os.path.getsize(fp)//1024} KB)')
    else:
        print(f'  ✗  {fn}  (missing)')
