import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pyarrow.dataset as ds 
import pyarrow as pa

DATA_DIR = "stgcn_dataset"
NUM_NODES = 263
SEQ_LEN = 24       
PRED_LEN = 1       
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ============================================================
# DATA LOADING  — only scaler_X, Y stays in original scale
# ============================================================
def load_and_preprocess_data():
    print("--- [1] Loading and Preprocessing Data ---")

    parquet_path = os.path.join(DATA_DIR, "node_features_X.parquet")
    dataset = ds.dataset(parquet_path)

    time_bin_table = dataset.to_table(columns=['time_bin'])
    time_bins = (time_bin_table.column('time_bin').to_pandas()
                 .drop_duplicates().sort_values().values)
    del time_bin_table
    num_timesteps = len(time_bins)

    all_cols     = [f.name for f in dataset.schema]
    targets      = ['demand', 'revenue_total']
    drop_cols    = ['time_bin', 'LocationID', 'node_index']
    datetime_cols = [f.name for f in dataset.schema
                     if pa.types.is_timestamp(f.type) or pa.types.is_date(f.type)]
    feature_cols = [c for c in all_cols
                    if c not in drop_cols and c not in datetime_cols and c not in targets]

    num_features = len(feature_cols)
    print(f"Total timesteps: {num_timesteps}, Features per node: {num_features}")

    years     = pd.to_datetime(time_bins).year
    train_idx = np.where(years == 2023)[0]
    val_idx   = np.where(years == 2024)[0]
    test_idx  = np.where(years == 2025)[0]
    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Fit scaler_X only (no scaler_Y — Tweedie predicts in original scale)
    train_row_start = int(train_idx[0]) * NUM_NODES
    train_row_end   = (int(train_idx[-1]) + 1) * NUM_NODES

    scaler_X = StandardScaler()
    print("Fitting scaler_X on training data (streaming)...")
    scanner  = dataset.scanner(columns=feature_cols + targets,
                                batch_size=NUM_NODES * 168)
    rows_seen = 0
    for batch in scanner.to_batches():
        batch_len   = batch.num_rows
        batch_start = rows_seen
        batch_end   = rows_seen + batch_len
        overlap_s   = max(batch_start, train_row_start)
        overlap_e   = min(batch_end,   train_row_end)
        if overlap_s < overlap_e:
            df_chunk = batch.slice(overlap_s - batch_start,
                                   overlap_e - overlap_s).to_pandas()
            X_chunk  = df_chunk[feature_cols].astype('float32').values
            X_chunk  = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            scaler_X.partial_fit(X_chunk)
        rows_seen += batch_len
        if rows_seen >= train_row_end:
            break

    print("scaler_X fitted.")

    # Save scaler_X and metadata for evaluation script
    joblib.dump(scaler_X, os.path.join(DATA_DIR, "scaler_X.pkl"))
    np.save(os.path.join(DATA_DIR, "num_features.npy"), np.array(num_features))
    np.save(os.path.join(DATA_DIR, "feature_cols.npy"), np.array(feature_cols))
    np.save(os.path.join(DATA_DIR, "targets.npy"),      np.array(targets))

    X_mm, Y_mm = build_memmap(parquet_path, feature_cols, targets,
                               num_timesteps, DATA_DIR, scaler_X)

    return (parquet_path, X_mm, Y_mm, time_bins,
            train_idx, val_idx, test_idx, feature_cols, targets, scaler_X)


def build_memmap(parquet_path, feature_cols, targets, num_timesteps, data_dir, scaler_X):
    """
    Stream parquet → write scaled X and raw Y to memmap.
    X is scaled here so __getitem__ is pure array slicing (no transform overhead).
    Y is stored raw (original scale) — Tweedie loss needs it that way.
    """
    X_path       = os.path.join(data_dir, "X_memmap.dat")
    Y_path       = os.path.join(data_dir, "Y_memmap.dat")
    num_features = len(feature_cols)
    num_targets  = len(targets)
    total_rows   = num_timesteps * NUM_NODES

    if os.path.exists(X_path) and os.path.exists(Y_path):
        print("Reusing existing memmap files.")
        X_mm = np.memmap(X_path, dtype='float32', mode='r',
                         shape=(total_rows, num_features))
        Y_mm = np.memmap(Y_path, dtype='float32', mode='r',
                         shape=(total_rows, num_targets))
        return X_mm, Y_mm

    print("Building memmap (one-time, streaming)...")
    X_mm = np.memmap(X_path, dtype='float32', mode='w+',
                     shape=(total_rows, num_features))
    Y_mm = np.memmap(Y_path, dtype='float32', mode='w+',
                     shape=(total_rows, num_targets))

    scanner = ds.dataset(parquet_path).scanner(
        columns=feature_cols + targets, batch_size=NUM_NODES * 336)

    row = 0
    for batch in scanner.to_batches():
        df_chunk = batch.to_pandas()
        X_chunk  = df_chunk[feature_cols].astype('float32').values
        Y_chunk  = df_chunk[targets].astype('float32').values
        X_chunk  = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        Y_chunk  = np.nan_to_num(Y_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        # Scale X; keep Y raw
        X_chunk  = scaler_X.transform(X_chunk)
        n = len(X_chunk)
        X_mm[row:row+n] = X_chunk
        Y_mm[row:row+n] = Y_chunk
        row += n

    X_mm.flush()
    Y_mm.flush()
    print(f"Memmap built: {row} rows written.")
    return X_mm, Y_mm


# ============================================================
# DATASET  — no transform in __getitem__, memmap already scaled
# ============================================================
class SpatioTemporalDataset(Dataset):
    def __init__(self, X_mm, Y_mm, time_bins, indices, seq_len, pred_len):
        self.X_mm          = X_mm
        self.Y_mm          = Y_mm
        self.seq_len       = seq_len
        self.pred_len      = pred_len
        self.valid_indices = [i for i in indices
                              if i - seq_len >= 0 and i + pred_len <= len(time_bins)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t   = self.valid_indices[idx]
        x_s = (t - self.seq_len) * NUM_NODES
        X   = self.X_mm[x_s : x_s + self.seq_len * NUM_NODES].copy()
        X   = X.reshape(self.seq_len, NUM_NODES, -1)

        y_s = t * NUM_NODES
        Y   = self.Y_mm[y_s : y_s + self.pred_len * NUM_NODES].copy()
        Y   = Y.reshape(self.pred_len, NUM_NODES, -1)

        return torch.FloatTensor(X), torch.FloatTensor(Y).squeeze(0)


# ============================================================
# MODEL
# ============================================================
class FastDenseGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.einsum('ij,btjf->btif', adj, out)
        return torch.relu(out)


class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, eps=1e-8):
        super().__init__()
        assert 1 < p < 2
        self.p   = p
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=0.0)
        loss   = (- y_true * y_pred.pow(1 - self.p) / (1 - self.p)
                  + y_pred.pow(2 - self.p) / (2 - self.p))
        return loss.mean()


class CausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 dilation=dilation, padding=self.padding)
        self.norm    = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)[:, :, :-self.padding]
        x = x.permute(0, 2, 1)
        return torch.relu(self.norm(x))


class FastMultiGraphSTGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_dim):
        super().__init__()
        self.gcn_spatial = FastDenseGCN(num_features, hidden_dim)
        self.gcn_flow    = FastDenseGCN(num_features, hidden_dim)
        self.tcn1        = CausalTCN(hidden_dim * 2, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2        = CausalTCN(hidden_dim,     hidden_dim, kernel_size=3, dilation=4)
        self.fc1         = nn.Linear(hidden_dim, 64)
        self.fc2         = nn.Linear(64, out_dim)
        self.softplus    = nn.Softplus()   # guarantees positive output for Tweedie

    def forward(self, x, adj_spatial, adj_flow):
        B, T, N, F = x.size()
        gcn_out = torch.cat([self.gcn_spatial(x, adj_spatial),
                              self.gcn_flow(x, adj_flow)], dim=-1)
        tcn_in  = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h_last  = self.tcn2(self.tcn1(tcn_in))[:, -1, :]
        return self.softplus(self.fc2(torch.relu(self.fc1(h_last)))).view(B, N, -1)


# ============================================================
# GRAPH LOADING
# ============================================================
def load_dense_graphs():
    A_s = torch.FloatTensor(np.load(os.path.join(DATA_DIR, "adj_spatial.npy"))).to(DEVICE)
    A_f = torch.FloatTensor(np.load(os.path.join(DATA_DIR, "adj_flow.npy"))).to(DEVICE)
    return A_s, A_f


# ============================================================
# METRICS
# ============================================================
def calculate_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    eps  = 1.0
    mask = y_true > eps
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, mape


# ============================================================
# TRAINING
# ============================================================
def train_model():
    adj_spatial, adj_flow = load_dense_graphs()

    (parquet_path, X_mm, Y_mm, time_bins,
     train_idx, val_idx, test_idx,
     feature_cols, targets, scaler_X) = load_and_preprocess_data()

    # Save time_bins for evaluation script
    np.save(os.path.join(DATA_DIR, "time_bins.npy"), time_bins)

    ds_kwargs = dict(X_mm=X_mm, Y_mm=Y_mm, time_bins=time_bins,
                     seq_len=SEQ_LEN, pred_len=PRED_LEN)

    train_ds = SpatioTemporalDataset(indices=train_idx, **ds_kwargs)
    val_ds   = SpatioTemporalDataset(indices=val_idx,   **ds_kwargs)
    test_ds  = SpatioTemporalDataset(indices=test_idx,  **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    num_features = len(feature_cols)
    num_targets  = len(targets)

    model     = FastMultiGraphSTGCN(num_features=num_features,
                                    hidden_dim=64, out_dim=num_targets).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TweedieLoss(p=1.5)

    print("\n--- [2] Starting Training ---")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader,
                                     desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x, adj_spatial, adj_flow)
            loss  = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                val_loss += criterion(model(batch_x, adj_spatial, adj_flow),
                                      batch_y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_stgcn_model_tweedie.pth")
            print("  --> Model saved!")

    # ---- Final test ----
    print("\n--- [3] Final Testing (Year 2025) ---")
    model.load_state_dict(torch.load("best_stgcn_model_tweedie.pth"))
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            preds = model(batch_x.to(DEVICE), adj_spatial, adj_flow)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())

    # Shape: (T*N, num_targets) — predictions already in original scale
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, num_targets)
    all_trues = np.concatenate(all_trues, axis=0).reshape(-1, num_targets)

    demand_mae, demand_rmse, demand_mape = calculate_metrics(
        all_trues[:, 0], all_preds[:, 0])
    rev_mae, rev_rmse, rev_mape = calculate_metrics(
        all_trues[:, 1], all_preds[:, 1])

    print("\n" + "="*52)
    print("  TEST SET PERFORMANCE (Year 2025)")
    print("="*52)
    print(f"[Demand]  MAE={demand_mae:.2f}  RMSE={demand_rmse:.2f}  MAPE={demand_mape:.1f}%")
    print(f"[Revenue] MAE={rev_mae:.2f}  RMSE={rev_rmse:.2f}  MAPE={rev_mape:.1f}%")
    print("="*52)


if __name__ == "__main__":
    train_model()