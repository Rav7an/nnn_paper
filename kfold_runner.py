"""
kfold_runner.py
===============
Single-file K=5 stratified cross-validation for all 6 DNA Tm models (E0–E5).

Usage (from repo root, nnn_win_torch env):
    python kfold_runner.py

Outputs (all in out/new/ — nothing in out/ is touched):
  out/new/splits/kfold5_arr_indices.json
  out/new/logs/E{x}_fold{k}_log.json        (per-epoch train_loss + val metrics)
  out/new/checkpoints/E{x}/fold{k}/best_model.pt
  out/new/preds/oof_E{x}.csv                (out-of-fold predictions)
  out/new/summary/kfold_summary.csv
  out/new/figures/learning_curves_E{x}.png  (train + val loss curves, all folds)
  out/new/figures/scatter_kfold_oof_a.png   (E0-E2, 3x3 dH/Tm/dG_37)
  out/new/figures/scatter_kfold_oof_b.png   (E3-E5, 3x3 dH/Tm/dG_37)
  out/new/figures/kfold_mae_bar.png
  out/new/figures/kfold_r2_bar.png
  out/new/figures/kfold_fold_variance.png
"""

import os, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import TransformerConv, Set2Set

sns.set_style('whitegrid')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(ROOT, 'data', 'models', 'raw')
OUT_NEW   = os.path.join(ROOT, 'out', 'new')

SPLIT_DIR = os.path.join(OUT_NEW, 'splits')
LOG_DIR   = os.path.join(OUT_NEW, 'logs')
CKPT_DIR  = os.path.join(OUT_NEW, 'checkpoints')
PRED_DIR  = os.path.join(OUT_NEW, 'preds')
SUM_DIR   = os.path.join(OUT_NEW, 'summary')
FIG_DIR   = os.path.join(OUT_NEW, 'figures')

for d in [SPLIT_DIR, LOG_DIR, CKPT_DIR, PRED_DIR, SUM_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
K_FOLDS        = 5
N_EPOCH        = 200
BATCH_SIZE     = 256
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
GRAD_CLIP      = 1.0
ALPHA_PINN     = 0.1     # PINN physics penalty weight
MAX_LEN        = 24
MAX_WIDTH      = 15
RANDOM_STATE   = 42
LOG_EVERY      = 20      # print progress every N epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

COLORS = {
    'E0': '#7f8c8d', 'E1': '#3498db', 'E2': '#e74c3c',
    'E3': '#9b59b6', 'E4': '#e67e22', 'E5': '#1abc9c',
}
LABELS = {
    'E0': 'E0: GNN',    'E1': 'E1: 1D CNN', 'E2': 'E2: 2D CNN',
    'E3': 'E3: SAT',    'E4': 'E4: PINN',   'E5': 'E5: Hybrid',
}
ALL_MODELS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']

NT_MAP  = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
ST_MAP  = {'(': 0, ')': 1, '.': 2}

# ── Data ───────────────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(os.path.join(RAW_DIR, 'combined_dataset.csv'), index_col='SEQID')
df.sort_index(inplace=True)

# Use all arr sequences with valid dH + Tm
arr_df = df[(df['dataset'] == 'arr')].dropna(subset=['dH', 'Tm']).copy()
print(f'arr pool: {len(arr_df):,} sequences')

def normalize(v, mn, mx):   return (v - mn) / (mx - mn)
def unnormalize(v, mn, mx): return v * (mx - mn) + mn

def make_sumstats(sub_df):
    return {
        'dH_min': float(sub_df['dH'].min()), 'dH_max': float(sub_df['dH'].max()),
        'Tm_min': float(sub_df['Tm'].min()), 'Tm_max': float(sub_df['Tm'].max()),
        'dS_min': -0.25, 'dS_max': -0.002,
    }

# ── Encoding helpers ───────────────────────────────────────────────────────────
def _clean(row):
    raw_seq  = str(row['RefSeq'])
    raw_st   = str(row['TargetStruct'])
    if '[' in raw_seq:
        try:    strands = eval(raw_seq)
        except: strands = [raw_seq]
        seq = ''.join(strands)
    else:
        seq = raw_seq.replace('+', '')
    struct = raw_st.replace('+', '').replace(' ', '')
    L = min(len(seq), len(struct))
    return seq[:L], struct[:L]

def enc_1d(row):
    seq, struct = _clean(row)
    L = min(len(seq), MAX_LEN)
    x = np.zeros((MAX_LEN, 7), dtype=np.float32)
    for i in range(L):
        if seq[i].upper() in NT_MAP:              x[i, NT_MAP[seq[i].upper()]] = 1.
        if i < len(struct) and struct[i] in ST_MAP: x[i, 4 + ST_MAP[struct[i]]] = 1.
    return x  # (MAX_LEN, 7)

def enc_2d(row):
    seq, struct = _clean(row)
    raw_st = str(row['TargetStruct'])
    if '+' in raw_st:
        plus = raw_st.index('+'); s1 = seq[:plus]; s2 = seq[plus:]
        t = np.zeros((6, 3, MAX_WIDTH), dtype=np.float32)
        for i, nt in enumerate(s1[:MAX_WIDTH]):
            if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()], 0, i] = 1.
        for i, nt in enumerate(s2[::-1][:MAX_WIDTH]):
            if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()], 2, i] = 1.
        for i in range(min(len(s1), MAX_WIDTH)): t[4, 1, i] = 1.
        return t
    ns  = struct.count('('); nl = struct.count('.'); hl = nl // 2; fl = ns + hl
    top = seq[:fl]; bot = seq[fl + (1 if nl % 2 == 1 else 0):][::-1]
    hb  = [1.] * ns + [0.] * hl
    t   = np.zeros((6, 3, MAX_WIDTH), dtype=np.float32)
    for i, nt in enumerate(top[:MAX_WIDTH]):
        if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()], 0, i] = 1.
    for i, nt in enumerate(bot[:MAX_WIDTH]):
        if nt.upper() in NT_MAP: t[NT_MAP[nt.upper()], 2, i] = 1.
    for i, h in enumerate(hb[:MAX_WIDTH]): t[4, 1, i] = h
    if fl < MAX_WIDTH: t[5, :, fl] = 1.
    return t

def enc_gnn(row):
    seq, struct = _clean(row)
    N = len(seq)
    x = np.zeros((N, 4), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        if nt in NT_MAP: x[i, NT_MAP[nt]] = 1.
    bb = [[i, i+1] for i in range(N - 1)]; hb = []
    stk = []
    for i, c in enumerate(struct):
        if c == '(':  stk.append(i)
        elif c == ')' and stk: j = stk.pop(); hb.append([j, i])
    edges = bb + [e[::-1] for e in bb] + hb + [e[::-1] for e in hb]
    ea    = np.zeros((len(edges), 3), dtype=np.float32)
    nb    = len(bb)
    ea[:nb, 0] = 1; ea[nb:2*nb, 1] = 1; ea[2*nb:, 2] = 1
    ei = (torch.tensor(np.array(edges).T, dtype=torch.long)
          if edges else torch.zeros((2, 0), dtype=torch.long))
    return Data(x=torch.tensor(x), edge_index=ei, edge_attr=torch.tensor(ea))

def enc_M(row):
    seq, struct = _clean(row)
    L = min(len(seq), MAX_LEN)
    M = np.zeros((MAX_LEN, MAX_LEN), dtype=np.float32)
    for i in range(L - 1): M[i, i+1] = M[i+1, i] = 1.
    stk = []
    for i, c in enumerate(struct[:L]):
        if c == '(':  stk.append(i)
        elif c == ')' and stk: j = stk.pop(); M[i, j] = M[j, i] = 1.
    return M

# ── Dataset classes ────────────────────────────────────────────────────────────
class MultiEncDataset(Dataset):
    """Pre-encodes all samples; stores 1d, 2d, M, and targets."""
    def __init__(self, sub_df, ss):
        self.rows = sub_df.reset_index(drop=False)
        self.ss   = ss
        print(f'    Encoding {len(self.rows)} sequences...', end=' ', flush=True)
        self.x1d = np.stack([enc_1d(r) for _, r in self.rows.iterrows()])   # (N,L,7)
        self.x2d = np.stack([enc_2d(r) for _, r in self.rows.iterrows()])   # (N,6,3,W)
        self.xM  = np.stack([enc_M(r)  for _, r in self.rows.iterrows()])   # (N,L,L)
        self.gnn = [enc_gnn(r) for _, r in self.rows.iterrows()]
        ys = np.stack([
            [normalize(float(r['dH']), ss['dH_min'], ss['dH_max']),
             normalize(float(r['Tm']), ss['Tm_min'], ss['Tm_max'])]
            for _, r in self.rows.iterrows()
        ], axis=0).astype(np.float32)
        self.y = ys
        print('done.')

    def __len__(self):  return len(self.rows)
    def __getitem__(self, i):
        return (self.x1d[i], self.x2d[i], self.xM[i], self.y[i],
                self.rows.iloc[i]['SEQID'])


def collate_fn(batch):
    x1d = torch.tensor(np.stack([b[0] for b in batch]))   # (B,L,7)
    x2d = torch.tensor(np.stack([b[1] for b in batch]))   # (B,6,3,W)
    xM  = torch.tensor(np.stack([b[2] for b in batch]))   # (B,L,L)
    y   = torch.tensor(np.stack([b[3] for b in batch]))   # (B,2)
    ids = [b[4] for b in batch]
    return x1d, x2d, xM, y, ids


# ── Model definitions ──────────────────────────────────────────────────────────
class GNN(nn.Module):
    def __init__(self, hidden=125, n_conv=4, lin_hidden=128,
                 conv_drop=0.0127, lin_drop=0.25, s2s_steps=10):
        super().__init__()
        self.conv_drop = conv_drop; self.lin_drop = lin_drop
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(4, hidden, heads=1, edge_dim=3, dropout=conv_drop))
        for _ in range(n_conv - 1):
            self.convs.append(TransformerConv(hidden, hidden, heads=4, edge_dim=3,
                                              dropout=conv_drop, concat=False))
        self.pool = Set2Set(hidden, processing_steps=s2s_steps)
        self.lin1 = nn.Linear(2*hidden, lin_hidden)
        self.lin2 = nn.Linear(lin_hidden, lin_hidden)
        self.lin3 = nn.Linear(lin_hidden, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = F.dropout(F.leaky_relu(conv(x, edge_index, edge_attr)),
                          p=self.conv_drop, training=self.training)
        x = self.pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.lin_drop, training=self.training)
        return self.lin3(F.relu(self.lin2(x)))

class DNA_1DCNN(nn.Module):
    def __init__(self, in_channels=7, dropout=0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64,  3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64,          128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(128,         128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(128,         128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):  return self.head(self.pool(self.conv_block(x)).squeeze(-1))

class AttentionPool2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(c, 64, 1), nn.Tanh(), nn.Conv2d(64, 1, 1))
    def forward(self, x):
        B, C, H, W = x.shape
        w = torch.softmax(self.attn(x).view(B, 1, -1), dim=-1)
        return (x.view(B, C, -1) * w).sum(-1)

def _bb2d(ic, dr):
    return nn.Sequential(
        nn.Conv2d(ic,  64,  (3,3), padding=(1,1)), nn.BatchNorm2d(64),  nn.ReLU(), nn.Dropout2d(dr),
        nn.Conv2d(64,  128, (3,3), padding=(1,1)), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dr),
        nn.Conv2d(128, 128, (3,5), padding=(1,2)), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dr),
    )

class DNA_2DCNN(nn.Module):
    def __init__(self, in_channels=6, dropout=0.2):
        super().__init__()
        self.conv_block = _bb2d(in_channels, dropout)
        self.pool       = AttentionPool2d(128)
        self.head       = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):  return self.head(self.pool(self.conv_block(x)))

class StructureBiasedAttention(nn.Module):
    def __init__(self, d, nh, dr=0.1, li=1.0):
        super().__init__()
        self.nhead = nh; self.hd = d // nh; self.scale = math.sqrt(self.hd)
        self.Wq = nn.Linear(d, d, bias=False); self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False); self.Wo = nn.Linear(d, d)
        self.lam = nn.Parameter(torch.tensor(float(li))); self.drop = nn.Dropout(dr)
    def forward(self, x, M, pm=None):
        B, L, d = x.shape; H, D = self.nhead, self.hd
        Q = self.Wq(x).view(B,L,H,D).transpose(1,2)
        K = self.Wk(x).view(B,L,H,D).transpose(1,2)
        V = self.Wv(x).view(B,L,H,D).transpose(1,2)
        lg = torch.matmul(Q, K.transpose(-2,-1)) / self.scale + self.lam * M.unsqueeze(1)
        if pm is not None: lg = lg.masked_fill(pm.unsqueeze(1).unsqueeze(2), float('-inf'))
        return self.Wo(torch.matmul(self.drop(torch.softmax(lg, dim=-1)), V)
                       .transpose(1,2).contiguous().view(B, L, d)), None

class SATransformerLayer(nn.Module):
    def __init__(self, d, nh, ff, dr=0.1, li=1.0):
        super().__init__()
        self.attn = StructureBiasedAttention(d, nh, dr, li)
        self.ff   = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dr), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.drop = nn.Dropout(dr)
    def forward(self, x, M, pm=None):
        ao, _ = self.attn(self.n1(x), M, pm); x = x + self.drop(ao)
        return x + self.drop(self.ff(self.n2(x)))

class StructureAwareTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=128, nhead=8, num_layers=4,
                 ff_dim=256, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.proj  = nn.Linear(input_dim, d_model)
        self.pos   = nn.Embedding(max_len, d_model)
        self.drop  = nn.Dropout(dropout)
        self.layers = nn.ModuleList([SATransformerLayer(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)])
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, 2))
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, x, M):
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pm  = (x.sum(-1) == 0)
        h   = self.drop(self.proj(x) + self.pos(pos))
        for ly in self.layers: h = ly(h, M, pm)
        h = self.norm(h); npad = (~pm).float().unsqueeze(-1)
        return self.head((h * npad).sum(1) / npad.sum(1).clamp(min=1))

class ThermodynamicsLayer(nn.Module):
    def __init__(self, ss):
        super().__init__()
        for k in ['dH_min','dH_max','dS_min','dS_max','Tm_min','Tm_max']:
            self.register_buffer(k, torch.tensor(ss[k], dtype=torch.float32))
    def forward(self, dH_n, dS_n):
        dH = dH_n * (self.dH_max - self.dH_min) + self.dH_min
        dS = dS_n * (self.dS_max - self.dS_min) + self.dS_min
        dS = torch.where(dS < 0, dS.clamp(max=-1e-4), dS.clamp(min=1e-4))
        return ((dH/dS - 273.15 - self.Tm_min) / (self.Tm_max - self.Tm_min)).clamp(-2., 3.)

class PINN_2DCNN(nn.Module):
    def __init__(self, ss, in_channels=6, dropout=0.2):
        super().__init__()
        self.backbone  = _bb2d(in_channels, dropout)
        self.attn_pool = AttentionPool2d(128)
        self.head      = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
        self.thermo    = ThermodynamicsLayer(ss)
    def forward(self, x):
        o = self.head(self.attn_pool(self.backbone(x)))
        return {'dH_norm': o[:,0], 'dS_norm': o[:,1], 'Tm_norm': self.thermo(o[:,0], o[:,1])}

class PINNLoss(nn.Module):
    def __init__(self, ss, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.ss    = ss
    def forward(self, out, y):
        dH_n  = out['dH_norm']; Tm_n = out['Tm_norm']
        l_dH  = F.mse_loss(dH_n, y[:, 0])
        l_Tm  = F.mse_loss(Tm_n, y[:, 1])
        ss    = self.ss
        dH    = dH_n.detach() * (ss['dH_max'] - ss['dH_min']) + ss['dH_min']
        dS    = out['dS_norm'].detach() * (ss['dS_max'] - ss['dS_min']) + ss['dS_min']
        p1    = F.relu(dH); p2 = F.relu(dS)
        p3    = F.relu(dH - 310.15 * dS)
        pen   = (p1 + p2 + p3).mean()
        return l_dH + l_Tm + self.alpha * pen, l_dH.item(), l_Tm.item(), pen.item()

class BiGRUBranch(nn.Module):
    def __init__(self, id=7, h=64, nl=2, dr=0.2):
        super().__init__()
        self.gru  = nn.GRU(id, h, nl, batch_first=True, bidirectional=True,
                           dropout=dr if nl > 1 else 0.)
        self.drop = nn.Dropout(dr)
    def forward(self, x):
        np_ = (x.sum(-1) != 0).float().unsqueeze(-1)
        o, _ = self.gru(x)
        return self.drop((o * np_).sum(1) / np_.sum(1).clamp(min=1))

class HybridCNNRNN(nn.Module):
    def __init__(self, seq_input_dim=7, gru_hidden=64, gru_layers=2, cnn_in_channels=6, dropout=0.2):
        super().__init__()
        self.rnn = BiGRUBranch(seq_input_dim, gru_hidden, gru_layers, dropout)
        self.cnn = nn.Sequential(_bb2d(cnn_in_channels, dropout), AttentionPool2d(128))
        self.head = nn.Sequential(
            nn.Linear(gru_hidden*2+128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),              nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
    def forward(self, x1d, x2d):
        cnn_feat = x2d
        for layer in self.cnn: cnn_feat = layer(cnn_feat)
        return self.head(torch.cat([self.rnn(x1d), cnn_feat], dim=-1))

def build_model(exp_id, ss):
    if exp_id == 'E0': return GNN().to(device)
    if exp_id == 'E1': return DNA_1DCNN().to(device)
    if exp_id == 'E2': return DNA_2DCNN().to(device)
    if exp_id == 'E3': return StructureAwareTransformer().to(device)
    if exp_id == 'E4': return PINN_2DCNN(ss).to(device)
    if exp_id == 'E5': return HybridCNNRNN().to(device)
    raise ValueError(exp_id)

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(pred_n, true_n, ss):
    """pred_n, true_n: numpy (N,2) normalized [dH, Tm]"""
    dH_p = pred_n[:,0] * (ss['dH_max']-ss['dH_min']) + ss['dH_min']
    Tm_p = pred_n[:,1] * (ss['Tm_max']-ss['Tm_min']) + ss['Tm_min']
    dH_t = true_n[:,0] * (ss['dH_max']-ss['dH_min']) + ss['dH_min']
    Tm_t = true_n[:,1] * (ss['Tm_max']-ss['Tm_min']) + ss['Tm_min']
    dG_p = dH_p * (1. - 310.15 / (273.15 + Tm_p))
    dG_t = dH_t * (1. - 310.15 / (273.15 + Tm_t))
    out  = {}
    for tag, p, t in [('dH', dH_p, dH_t), ('Tm', Tm_p, Tm_t), ('dG_37', dG_p, dG_t)]:
        m = np.isfinite(p) & np.isfinite(t)
        d = p[m] - t[m]
        out[f'{tag}_mae']  = float(np.mean(np.abs(d)))
        out[f'{tag}_rmse'] = float(np.sqrt(np.mean(d**2)))
        out[f'{tag}_r2']   = float(r2_score(t[m], p[m])) if m.sum() > 1 else float('nan')
    return out, dH_p, Tm_p, dH_t, Tm_t, dG_p, dG_t

# ── Evaluate on a DataLoader (non-GNN models) ──────────────────────────────────
@torch.no_grad()
def eval_model(exp_id, model, loader, ss):
    model.eval()
    all_pred, all_true = [], []
    for x1d, x2d, xM, y, _ in loader:
        x1d = x1d.to(device); x2d = x2d.to(device)
        xM  = xM.to(device);  y   = y.to(device)
        if exp_id == 'E1':
            pred = model(x1d.permute(0,2,1))   # (B,7,L)
        elif exp_id == 'E2':
            pred = model(x2d)
        elif exp_id == 'E3':
            pred = model(x1d, xM)
        elif exp_id == 'E4':
            o    = model(x2d)
            pred = torch.stack([o['dH_norm'], o['Tm_norm']], dim=1)
        elif exp_id == 'E5':
            pred = model(x1d, x2d)
        else:
            raise ValueError(f'eval_model: {exp_id} not handled here')
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    pred_n = np.concatenate(all_pred, axis=0)
    true_n = np.concatenate(all_true, axis=0)
    return compute_metrics(pred_n, true_n, ss), pred_n, true_n

@torch.no_grad()
def eval_gnn(model, gnn_list, y_true_n, ss, batch_size=256):
    model.eval()
    all_pred = []
    loader   = GeoDataLoader(gnn_list, batch_size=batch_size, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_pred.append(out.cpu().numpy())
    pred_n = np.concatenate(all_pred, axis=0)
    return compute_metrics(pred_n, y_true_n, ss), pred_n, y_true_n

# ── One training run ───────────────────────────────────────────────────────────
def train_one_fold(exp_id, fold_idx, train_ds, val_ds, ss):
    ckpt_dir  = os.path.join(CKPT_DIR, exp_id, f'fold{fold_idx}')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    log_path  = os.path.join(LOG_DIR, f'{exp_id}_fold{fold_idx}_log.json')
    oof_path  = os.path.join(PRED_DIR, f'oof_{exp_id}_fold{fold_idx}.csv')

    # ── Resume: skip if this fold already completed ───────────────────────────
    if os.path.exists(ckpt_path) and os.path.exists(log_path) and os.path.exists(oof_path):
        print(f'  [{exp_id} fold{fold_idx}] Already done — loading saved OOF predictions.')
        with open(log_path) as f:
            saved_log = json.load(f)
        oof_df = pd.read_csv(oof_path)
        last_vm = {k: saved_log['history'][f'val_{k}'][-1]
                   for k in ['dH_mae','Tm_mae','dG_37_mae','dH_r2','Tm_r2','dG_37_r2','Tm_rmse']
                   if f'val_{k}' in saved_log['history']}
        return oof_df, saved_log['history'], last_vm

    model     = build_model(exp_id, ss)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCH, eta_min=1e-5)
    criterion = PINNLoss(ss, ALPHA_PINN) if exp_id == 'E4' else nn.MSELoss()

    # GNN needs its own loader
    if exp_id == 'E0':
        train_gnn_loader = GeoDataLoader(train_ds.gnn, batch_size=BATCH_SIZE, shuffle=True)
        val_gnn_list     = val_ds.gnn
        val_y_true_n     = val_ds.y
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=512,         shuffle=False,
                                  collate_fn=collate_fn, num_workers=0)

    history = {
        'train_loss': [],
        'val_dH_mae': [], 'val_Tm_mae': [], 'val_dG_37_mae': [],
        'val_dH_r2':  [], 'val_Tm_r2':  [], 'val_dG_37_r2':  [],
        'val_Tm_rmse': [],
    }
    if exp_id == 'E4':
        history['train_loss_dH']  = []
        history['train_loss_Tm']  = []
        history['train_penalty']  = []

    best_val_dG = float('inf')
    t0 = time.time()

    for epoch in range(N_EPOCH):
        model.train()
        ep_loss = ep_dH = ep_Tm = ep_pen = 0.
        n_batches = 0

        if exp_id == 'E0':
            for batch in train_gnn_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y.view(-1, 2))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                ep_loss += loss.item(); n_batches += 1
        else:
            for x1d, x2d, xM, y, _ in train_loader:
                x1d = x1d.to(device); x2d = x2d.to(device)
                xM  = xM.to(device);  y   = y.to(device)
                optimizer.zero_grad()

                if exp_id == 'E1':
                    out  = model(x1d.permute(0,2,1))
                    loss = criterion(out, y)
                elif exp_id == 'E2':
                    out  = model(x2d)
                    loss = criterion(out, y)
                elif exp_id == 'E3':
                    out  = model(x1d, xM)
                    loss = criterion(out, y)
                elif exp_id == 'E4':
                    out  = model(x2d)
                    loss, l_dH, l_Tm, pen = criterion(out, y)
                    ep_dH += l_dH; ep_Tm += l_Tm; ep_pen += pen
                elif exp_id == 'E5':
                    out  = model(x1d, x2d)
                    loss = criterion(out, y)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                ep_loss += loss.item(); n_batches += 1

        scheduler.step()
        ep_loss /= max(n_batches, 1)
        history['train_loss'].append(ep_loss)
        if exp_id == 'E4':
            history['train_loss_dH'].append(ep_dH / max(n_batches, 1))
            history['train_loss_Tm'].append(ep_Tm / max(n_batches, 1))
            history['train_penalty'].append(ep_pen / max(n_batches, 1))

        # Validation
        if exp_id == 'E0':
            (vm, *_), _, _ = eval_gnn(model, val_gnn_list, val_y_true_n, ss)
        else:
            (vm, *_), _, _ = eval_model(exp_id, model, val_loader, ss)

        for k in ['dH_mae','Tm_mae','dG_37_mae','dH_r2','Tm_r2','dG_37_r2','Tm_rmse']:
            history[f'val_{k}'].append(vm.get(k, float('nan')))

        if vm['dG_37_mae'] < best_val_dG:
            best_val_dG = vm['dG_37_mae']
            torch.save(model.state_dict(), ckpt_path)

        if (epoch+1) % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            print(f'  [{exp_id} fold{fold_idx}] Ep {epoch+1:3d}/{N_EPOCH} '
                  f'| train_loss {ep_loss:.5f} '
                  f'| val_Tm_mae {vm["Tm_mae"]:.3f} '
                  f'| val_dG_mae {vm["dG_37_mae"]:.4f} '
                  f'| {elapsed:.1f}min')

    # Save log
    with open(log_path, 'w') as f:
        json.dump({'exp_id': exp_id, 'fold': fold_idx, 'best_val_dG': best_val_dG,
                   'history': history}, f)
    print(f'  [{exp_id} fold{fold_idx}] Best val dG MAE = {best_val_dG:.4f} | log → {log_path}')

    # Load best and collect OOF predictions
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if exp_id == 'E0':
        _, pred_n, true_n = eval_gnn(model, val_gnn_list, val_y_true_n, ss)
    else:
        _, pred_n, true_n = eval_model(exp_id, model, val_loader, ss)

    # Convert to physical units
    dH_p = pred_n[:,0] * (ss['dH_max']-ss['dH_min']) + ss['dH_min']
    Tm_p = pred_n[:,1] * (ss['Tm_max']-ss['Tm_min']) + ss['Tm_min']
    dH_t = true_n[:,0] * (ss['dH_max']-ss['dH_min']) + ss['dH_min']
    Tm_t = true_n[:,1] * (ss['Tm_max']-ss['Tm_min']) + ss['Tm_min']
    dG_p = dH_p * (1. - 310.15 / (273.15 + Tm_p))
    dG_t = dH_t * (1. - 310.15 / (273.15 + Tm_t))

    seqids = val_ds.rows['SEQID'].values if 'SEQID' in val_ds.rows.columns else val_ds.rows.index.values
    oof_df = pd.DataFrame({
        'SEQID':    seqids,
        'fold':     fold_idx,
        'dH_true':  dH_t, 'dH_pred':  dH_p,
        'Tm_true':  Tm_t, 'Tm_pred':  Tm_p,
        'dG_37_true': dG_t, 'dG_37_pred': dG_p,
    })
    # Save per-fold OOF CSV (used for resume detection)
    oof_df.to_csv(oof_path, index=False)
    return oof_df, history, vm

# ── K-fold split generation ────────────────────────────────────────────────────
def generate_folds(df):
    split_path = os.path.join(SPLIT_DIR, 'kfold5_arr_indices.json')
    if os.path.exists(split_path):
        print('Loading existing folds from', split_path)
        with open(split_path) as f:
            saved = json.load(f)
        return [(saved[f'fold_{k}']['train'], saved[f'fold_{k}']['val'])
                for k in range(K_FOLDS)]

    print(f'Generating K={K_FOLDS} stratified folds...')
    seqids = df.index.tolist()
    Tm_bins = pd.cut(df['Tm'], bins=5, labels=False).fillna(0).astype(int).values
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    saved = {}
    for k, (tr_idx, va_idx) in enumerate(skf.split(seqids, Tm_bins)):
        tr_ids = [seqids[i] for i in tr_idx]
        va_ids = [seqids[i] for i in va_idx]
        folds.append((tr_ids, va_ids))
        saved[f'fold_{k}'] = {'train': tr_ids, 'val': va_ids}
        print(f'  Fold {k}: train={len(tr_ids):,}  val={len(va_ids):,}')

    with open(split_path, 'w') as f:
        json.dump(saved, f)
    print(f'Folds saved → {split_path}')
    return folds

# ── Figure generation ──────────────────────────────────────────────────────────
def plot_learning_curves(exp_id, fold_histories):
    """One figure per model: train_loss + val_Tm_mae, all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle(f'Learning Curves — {LABELS[exp_id]} (K=5 folds)', fontsize=14, fontweight='bold')

    for fold_idx, hist in enumerate(fold_histories):
        epochs = list(range(1, len(hist['train_loss'])+1))
        alpha  = 0.9 if fold_idx == 0 else 0.55
        lw     = 1.8 if fold_idx == 0 else 1.2
        label  = f'fold {fold_idx}'
        axes[0].plot(epochs, hist['train_loss'], alpha=alpha, lw=lw, label=label, color=plt.cm.tab10(fold_idx))
        axes[1].plot(epochs, hist['val_Tm_mae'],  alpha=alpha, lw=lw, label=label, color=plt.cm.tab10(fold_idx))

    axes[0].set_title('Train Loss (MSE normalized)'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[1].set_title('Val Tm MAE (°C)');             axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Tm MAE (°C)')
    for ax in axes: ax.legend(fontsize=9); sns.despine(ax=ax)

    # If PINN, add a 3rd panel for physics penalty
    if exp_id == 'E4' and 'train_penalty' in fold_histories[0]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5), facecolor='white')
        for fold_idx, hist in enumerate(fold_histories):
            epochs = list(range(1, len(hist['train_penalty'])+1))
            ax2.plot(epochs, hist['train_penalty'], lw=1.4, alpha=0.7,
                     label=f'fold {fold_idx}', color=plt.cm.tab10(fold_idx))
        ax2.set_title('PINN Physics Penalty per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Penalty')
        ax2.legend(fontsize=9); sns.despine(ax=ax2)
        fig2.tight_layout()
        fig2.savefig(os.path.join(FIG_DIR, f'pinn_penalty_curves.png'), dpi=200, bbox_inches='tight')
        plt.close(fig2)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'learning_curves_{exp_id}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(out)}')


def plot_oof_scatter(oof_dfs, exp_ids, fname):
    """3×3 grid (one row per model, cols: dH / Tm / dG_37). exp_ids = list of 3."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), facecolor='#f8f9fa')
    plt.subplots_adjust(hspace=0.38, wspace=0.28)

    cols  = [('dH', 'dH (kcal/mol)'), ('Tm', 'Tm (°C)'), ('dG_37', '\u0394G\u2083\u2087 (kcal/mol)')]
    titles = ['dH (kcal/mol)', 'Tm (\u00b0C)', '\u0394G\u2083\u2087 (kcal/mol)']

    for row_i, exp_id in enumerate(exp_ids):
        df_oof = oof_dfs[exp_id]
        for col_i, (tag, unit) in enumerate(cols):
            ax = axes[row_i, col_i]
            t  = df_oof[f'{tag}_true'].values
            p  = df_oof[f'{tag}_pred'].values
            m  = np.isfinite(t) & np.isfinite(p)
            ax.scatter(t[m], p[m], s=5, alpha=0.35, color=COLORS[exp_id], rasterized=True)
            lo = min(t[m].min(), p[m].min()) - 1; hi = max(t[m].max(), p[m].max()) + 1
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1.5)
            mae = np.mean(np.abs(p[m]-t[m]))
            r2  = r2_score(t[m], p[m]) if m.sum() > 1 else float('nan')
            ax.text(0.04, 0.96, f'MAE = {mae:.3f}\nR\u00b2  = {r2:.3f}',
                    transform=ax.transAxes, fontsize=10, va='top', linespacing=1.6,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='#ccc', lw=0.7))
            ax.tick_params(labelsize=9)
            if row_i == 2: ax.set_xlabel(titles[col_i], fontsize=11)
            if col_i == 0: ax.set_ylabel(f'{LABELS[exp_id]}\nPredicted', fontsize=10)
            if row_i == 0: ax.set_title(titles[col_i], fontsize=12, fontweight='bold', pad=6)
            sns.despine(ax=ax)

    fig.suptitle(f'K-Fold OOF Predictions ({", ".join(exp_ids)})',
                 fontsize=13, fontweight='bold', y=1.01)
    out = os.path.join(FIG_DIR, fname)
    plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f'  Saved: {fname}')


def plot_summary_bars(summary_df):
    orig = {
        'E0': 3.283, 'E1': 1.808, 'E2': 1.568,
        'E3': 2.251, 'E4': 2.754, 'E5': 1.530,
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

    for ax, metric, title in [
        (axes[0], 'Tm_mae', 'Tm MAE (°C) — lower is better'),
        (axes[1], 'Tm_r2',  'Tm R²  — higher is better'),
    ]:
        models = summary_df['exp_id'].tolist()
        means  = summary_df[f'{metric}_mean'].values
        stds   = summary_df[f'{metric}_std'].values
        xs     = np.arange(len(models))
        bars   = ax.bar(xs, means, yerr=stds, capsize=5, width=0.5,
                        color=[COLORS[m] for m in models], alpha=0.85)
        if metric == 'Tm_mae':
            for xi, m in zip(xs, models):
                if m in orig:
                    ax.hlines(orig[m], xi-0.28, xi+0.28, colors='black', linewidths=1.5,
                              linestyles='--', alpha=0.6)

        ax.set_xticks(xs); ax.set_xticklabels([LABELS[m] for m in models], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric); sns.despine(ax=ax)
        if metric == 'Tm_mae':
            ax.text(0.98, 0.97, 'dashed = original thesis val score',
                    transform=ax.transAxes, fontsize=8, ha='right', va='top', color='gray')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'kfold_summary_bars.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: kfold_summary_bars.png')


def plot_fold_variance(all_fold_metrics):
    """all_fold_metrics: {exp_id: [{fold: vm_dict}, ...]}"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    for exp_id in ALL_MODELS:
        fold_maes = [vm['Tm_mae'] for vm in all_fold_metrics[exp_id]]
        xs = np.arange(K_FOLDS)
        ax.plot(xs, fold_maes, 'o-', color=COLORS[exp_id], lw=2, ms=7, label=LABELS[exp_id])
    ax.set_xticks(range(K_FOLDS))
    ax.set_xticklabels([f'Fold {i}' for i in range(K_FOLDS)])
    ax.set_ylabel('Val Tm MAE (°C)', fontsize=12)
    ax.set_title('Per-Fold Val Tm MAE — All Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); sns.despine(ax=ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'kfold_fold_variance.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Saved: kfold_fold_variance.png')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()

    # 1. Generate folds
    folds = generate_folds(arr_df)

    # 2. Train all models × all folds
    oof_dfs           = {exp: [] for exp in ALL_MODELS}
    fold_histories    = {exp: [] for exp in ALL_MODELS}
    fold_val_metrics  = {exp: [] for exp in ALL_MODELS}

    for exp_id in ALL_MODELS:
        print(f'\n{"="*60}')
        print(f'  Model {exp_id} — {LABELS[exp_id]}')
        print(f'{"="*60}')

        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            print(f'\n  --- Fold {fold_idx}/{K_FOLDS-1} ---')
            train_sub = arr_df.loc[arr_df.index.isin(train_ids)]
            val_sub   = arr_df.loc[arr_df.index.isin(val_ids)]

            # Per-fold normalization stats (from training split only)
            ss = make_sumstats(train_sub)

            print(f'  Building datasets (train={len(train_sub):,} val={len(val_sub):,})')
            train_ds = MultiEncDataset(train_sub, ss)
            val_ds   = MultiEncDataset(val_sub,   ss)

            # For GNN: attach y tensors directly to graph objects
            if exp_id == 'E0':
                y_train_n = np.array([
                    [normalize(float(r['dH']), ss['dH_min'], ss['dH_max']),
                     normalize(float(r['Tm']),  ss['Tm_min'], ss['Tm_max'])]
                    for _, r in train_sub.iterrows()
                ], dtype=np.float32)
                y_val_n = np.array([
                    [normalize(float(r['dH']), ss['dH_min'], ss['dH_max']),
                     normalize(float(r['Tm']),  ss['Tm_min'], ss['Tm_max'])]
                    for _, r in val_sub.iterrows()
                ], dtype=np.float32)
                for g, y in zip(train_ds.gnn, y_train_n):
                    g.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)
                val_ds.y = y_val_n  # keep numpy for eval_gnn

            oof_df, hist, vm = train_one_fold(exp_id, fold_idx, train_ds, val_ds, ss)
            oof_dfs[exp_id].append(oof_df)
            fold_histories[exp_id].append(hist)
            fold_val_metrics[exp_id].append(vm)

        # Concatenate OOF predictions for this model
        full_oof      = pd.concat(oof_dfs[exp_id], ignore_index=True)
        full_oof_path = os.path.join(PRED_DIR, f'oof_{exp_id}.csv')
        full_oof.to_csv(full_oof_path, index=False)
        oof_dfs[exp_id] = full_oof
        print(f'  OOF saved → {full_oof_path}  ({len(full_oof):,} rows)')

    # 3. Build summary table
    rows = []
    for exp_id in ALL_MODELS:
        vms = fold_val_metrics[exp_id]
        row = {'exp_id': exp_id}
        for metric in ['dH_mae','Tm_mae','dG_37_mae','Tm_r2','dH_r2']:
            vals = [v[metric] for v in vms if not math.isnan(v.get(metric, float('nan')))]
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std']  = float(np.std(vals))
            row[f'{metric}_min']  = float(np.min(vals))
            row[f'{metric}_max']  = float(np.max(vals))
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values('Tm_mae_mean')
    summary_path = os.path.join(SUM_DIR, 'kfold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary saved → {summary_path}')
    print(summary_df[['exp_id','Tm_mae_mean','Tm_mae_std','Tm_r2_mean']].to_string(index=False))

    # 4. Figures
    print('\nGenerating figures...')

    # Learning curves (one per model)
    for exp_id in ALL_MODELS:
        plot_learning_curves(exp_id, fold_histories[exp_id])

    # OOF scatter plots
    plot_oof_scatter(oof_dfs, ['E0','E1','E2'], 'scatter_kfold_oof_a.png')
    plot_oof_scatter(oof_dfs, ['E3','E4','E5'], 'scatter_kfold_oof_b.png')

    # Summary bars + fold variance
    plot_summary_bars(summary_df)
    plot_fold_variance(fold_val_metrics)

    elapsed = (time.time() - t_total) / 60
    print(f'\nAll done in {elapsed:.1f} min.')
    print(f'Outputs: {OUT_NEW}')


if __name__ == '__main__':
    main()
