#!/usr/bin/env python3
"""
train_lstm.py  –  end-to-end WADI LSTM-Autoencoder baseline
-----------------------------------------------------------
1. Load & concatenate WADI_14days.csv   +   WADI_attackdata.csv
2. Parse attack windows from attack_description.xlsx
3. Fill/scale data   →   slice into non-overlapping windows (100 s)
4. Balanced train / test split (test = 50 % attack, 50 % normal)
5. Train LSTM-AE   →   ROC-AUC on window-level anomaly scores
6. Explain top-5 anomalous windows (feature MSE + permutation ΔMSE)
"""

import argparse, warnings, random
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# reproducibility ─────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────────────  CLI  ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="training epochs")
args = parser.parse_args()

# ───────────────────────  1) CSV LOADERS  ────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load one of the WADI CSVs and build a proper timestamp column."""
    # find header row (first line that starts with Row,Date,Time)
    with open(path) as fh:
        for idx, line in enumerate(fh):
            if line.startswith("Row,Date,Time"):
                header = idx
                break
    df = pd.read_csv(path, skiprows=header, low_memory=False)
    # shorten absurd OPC-tag columns
    df.columns = [c.split("\\")[-1].strip() for c in df.columns]
    # build datetime – allow AM/PM *or* fractional seconds
    ts_raw = df["Date"].astype(str) + " " + df["Time"].astype(str)
    df["timestamp"] = pd.to_datetime(ts_raw, errors="coerce",
                                     format="%m/%d/%Y %I:%M:%S %p")
    # if the fast path fails, let pandas guess
    bad = df["timestamp"].isna()
    if bad.any():
        df.loc[bad, "timestamp"] = pd.to_datetime(ts_raw[bad], errors="coerce")
    return df.drop(columns=["Row", "Date", "Time"])

print("↳  Loading WADI_14days.csv …")
part1 = load_csv("WADI_14days.csv")
print("↳  Loading WADI_attackdata.csv …")
part2 = load_csv("WADI_attackdata.csv")

data = (pd.concat([part1, part2], ignore_index=True)
          .sort_values("timestamp")
          .reset_index(drop=True))

print(f"✓  Total rows loaded: {len(data):,}")

# ───────────────  2) READ & NORMALISE ATTACK-SHEET  ─────────────────────
xls_raw = pd.read_excel("attack_description.xlsx", header=None,
                        engine="openpyxl")
hdr_row = xls_raw[xls_raw.iloc[:, 0].eq("S.No")].index[0]
meta = (pd.read_excel("attack_description.xlsx", header=hdr_row,
                      engine="openpyxl")
          .rename(columns=lambda c: c.strip()))

# Date cleanup
meta["date_only"] = (pd.to_datetime(meta["Date"], errors="coerce")
                       .dt.strftime("%Y-%m-%d"))

def _clean_time(col):
    return (meta[col].astype(str)
                .str.replace(r"[^\d:]", ":", regex=True)   # 11.30:40 → 11:30:40
                .str.strip())

meta["start_ts"] = pd.to_datetime(
    meta["date_only"] + " " + _clean_time("Start Time"),
    format="%Y-%m-%d %H:%M:%S", errors="coerce")
meta["end_ts"] = pd.to_datetime(
    meta["date_only"] + " " + _clean_time("End Time"),
    format="%Y-%m-%d %H:%M:%S", errors="coerce")

# ────────────────  3) LABEL THE MAIN DATAFRAME  ─────────────────────────
data["attack_label"] = 1
for _, r in meta.dropna(subset=["start_ts", "end_ts"]).iterrows():
    mask = (data.timestamp.between(r.start_ts, r.end_ts))
    data.loc[mask, "attack_label"] = -1

print("Label distribution:\n", data.attack_label.value_counts(), "\n")

# ────────────────  4) FILL ▸ SCALE ▸ WINDOW  ────────────────────────────
features = [c for c in data.columns if c not in ("timestamp", "attack_label")]

X_df = data[features].copy().ffill().bfill().fillna(0)
scaler = StandardScaler()
scaler.fit(X_df[data.attack_label.eq(1)])
scaler.scale_[scaler.scale_ == 0] = 1      # no /0
X = scaler.transform(X_df).astype(np.float32)

W = 100
n_win = (len(X) // W)
X_win = X[:n_win*W].reshape(n_win, W, -1)
y_win = data.attack_label.values[:n_win*W].reshape(n_win, W)
labels = (y_win == -1).any(axis=1).astype(int)

print(f"Windowed shape: {X_win.shape}")

# balanced test (attack = normal) ————————————————————————————————
att_idx, norm_idx = np.where(labels==1)[0], np.where(labels==0)[0]
rng = np.random.default_rng(SEED)
test_idx = np.concatenate([att_idx, rng.choice(norm_idx, len(att_idx), False)])
train_idx = np.setdiff1d(np.arange(n_win), test_idx)

X_train = X_win[train_idx]           # only normal windows
X_test  = X_win[test_idx]
y_test  = labels[test_idx]

print(f"Train windows: {X_train.shape} | Test windows: {X_test.shape}\n")

# ───────────────────────  5) LSTM - AUTOENCODER  ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAE(nn.Module):
    def __init__(self, n_feat, h=64, code=16):
        super().__init__()
        self.enc = nn.LSTM(n_feat, h, batch_first=True)
        self.e2c = nn.Linear(h, code)
        self.c2h = nn.Linear(code, h)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, n_feat)

    def forward(self, x):
        # ----- encoder -----
        enc_seq, (hN, _) = self.enc(x)          # enc_seq: (B,T,h)
        z   = torch.tanh(self.e2c(hN[-1]))      # code  : (B,code)

        # ----- prepare decoder initial state -----
        h0  = torch.tanh(self.c2h(z)).unsqueeze(0)  # (1,B,h)
        c0  = torch.zeros_like(h0)

        # feed zeros (or any BOS token) with *hidden*-dim features
        dec_in = torch.zeros_like(enc_seq)      # (B,T,h)

        # ----- decoder -----
        dec_seq, _ = self.dec(dec_in, (h0, c0)) # (B,T,h)

        # map back to original feature space
        return self.out(dec_seq)                # (B,T,n_feat)

model  = LSTMAE(X_train.shape[2]).to(device)
optim_ = optim.Adam(model.parameters(), lr=1e-3)
crit   = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                    batch_size=128, shuffle=True)

for epoch in range(1, args.epochs+1):
    model.train(); tot=0
    for (batch,) in loader:
        batch = batch.to(device)
        optim_.zero_grad()
        loss = crit(model(batch), batch)
        loss.backward(); optim_.step()
        tot += loss.item()*len(batch)
    print(f"Epoch {epoch}/{args.epochs}   loss = {tot/len(loader.dataset):.6f}")

# ─────────────────────────  6) TEST ROC-AUC  ────────────────────────────
model.eval()
with torch.no_grad():
    recon = model(torch.tensor(X_test).to(device)).cpu().numpy()

mse = ((recon - X_test)**2).mean(axis=(1,2))
print("\nTest ROC-AUC:", roc_auc_score(y_test, mse).round(4))

# ─────────────  7) TOP-N WINDOWS  +  PERM-IMPORT  ───────────────────────
top_n = np.argsort(mse)[::-1][:5]
print("\nTop-5 anomalous windows (+ feature MSE):\n")
for w_idx in top_n:
    per_feat = ((recon[w_idx]-X_test[w_idx])**2).mean(0)
    feat_top = per_feat.argsort()[-5:][::-1]
    print(f"Window #{w_idx:3d}  MSE={mse[w_idx]:.4f}")
    for f in feat_top:
        print(f"   • {features[f]:30s}  err={per_feat[f]:.4f}")
    print()

print("Permutation-importance (ΔMSE) …\n")
for w_idx in top_n:
    base = mse[w_idx]; window = X_test[w_idx]
    delta = np.zeros(window.shape[1], dtype=float)
    for f in range(window.shape[1]):
        w_shuf = window.copy(); rng.shuffle(w_shuf[:, f])
        with torch.no_grad():
            rec = (model(torch.tensor(w_shuf[None],dtype=torch.float32)
                          .to(device))
                   .detach().cpu().numpy()[0])
        delta[f] = ((rec - w_shuf)**2).mean() - base
    feat_top = delta.argsort()[-5:][::-1]
    print(f"Window #{w_idx:3d}  base={base:.4f}")
    for f in feat_top:
        print(f"   • {features[f]:30s}  ΔMSE={delta[f]:.4f}")
    print()
