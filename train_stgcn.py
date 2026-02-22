import os
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

def load_and_preprocess_data():
    print("--- [1] Loading and Preprocessing Data ---")

    parquet_path = os.path.join(DATA_DIR, "node_features_X.parquet")
    dataset = ds.dataset(parquet_path)

    # --- Get time_bins without loading full data ---
    time_bin_table = dataset.to_table(columns=['time_bin'])
    time_bins = time_bin_table.column('time_bin').to_pandas().drop_duplicates().sort_values().values
    del time_bin_table
    num_timesteps = len(time_bins)

    # --- Determine feature/target cols from schema only ---
    all_cols = [f.name for f in dataset.schema]
    targets = ['demand', 'revenue_total']
    drop_cols = ['time_bin', 'LocationID', 'node_index']
    # Also drop datetime cols by checking schema types
    import pyarrow as pa
    datetime_cols = [f.name for f in dataset.schema
                     if pa.types.is_timestamp(f.type) or pa.types.is_date(f.type)]
    feature_cols = [c for c in all_cols
                    if c not in drop_cols and c not in datetime_cols and c not in targets]

    num_features = len(feature_cols)
    print(f"Total timesteps: {num_timesteps}, Features per node: {num_features}")

    # --- Compute train/val/test split indices ---
    years = pd.to_datetime(time_bins).year
    train_idx = np.where(years == 2023)[0]
    val_idx   = np.where(years == 2024)[0]
    test_idx  = np.where(years == 2025)[0]
    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Fit scalers by streaming only train rows ---
    # Train rows = train_idx * NUM_NODES rows (row range in parquet)
    train_row_start = int(train_idx[0]) * NUM_NODES
    train_row_end   = (int(train_idx[-1]) + 1) * NUM_NODES

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    print("Fitting scalers on training data (streaming)...")
    scanner = dataset.scanner(
        columns=feature_cols + targets,
        batch_size=NUM_NODES * 168  # one week at a time
    )

    rows_seen = 0
    for batch in scanner.to_batches():
        batch_len = batch.num_rows
        batch_start = rows_seen
        batch_end   = rows_seen + batch_len

        # Check overlap with train row range
        overlap_start = max(batch_start, train_row_start)
        overlap_end   = min(batch_end,   train_row_end)

        if overlap_start < overlap_end:
            local_start = overlap_start - batch_start
            local_end   = overlap_end   - batch_start
            df_chunk = batch.slice(local_start, local_end - local_start).to_pandas()

            X_chunk = df_chunk[feature_cols].astype('float32').values
            Y_chunk = df_chunk[targets].astype('float32').values

            X_chunk = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            Y_chunk = np.nan_to_num(Y_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            scaler_X.partial_fit(X_chunk)
            scaler_Y.partial_fit(Y_chunk)

        rows_seen += batch_len
        if rows_seen >= train_row_end:
            break  # No need to scan beyond training data

    print("Scalers fitted.")
    X_mm, Y_mm = build_memmap(parquet_path, feature_cols, targets, num_timesteps, DATA_DIR)

    return (parquet_path, X_mm, Y_mm, time_bins, train_idx, val_idx, test_idx,
            feature_cols, targets, scaler_X, scaler_Y)

def load_dense_graphs():
    A_spatial = np.load(os.path.join(DATA_DIR, "adj_spatial.npy"))
    A_flow = np.load(os.path.join(DATA_DIR, "adj_flow.npy"))
    
    A_spatial_t = torch.FloatTensor(A_spatial).to(DEVICE)
    A_flow_t = torch.FloatTensor(A_flow).to(DEVICE)
    
    return A_spatial_t, A_flow_t

def build_memmap(parquet_path, feature_cols, targets, num_timesteps, data_dir):
    """Stream parquet → write to memmap files. Only done once."""
    X_path = os.path.join(data_dir, "X_memmap.dat")
    Y_path = os.path.join(data_dir, "Y_memmap.dat")
    num_features = len(feature_cols)
    num_targets  = len(targets)
    total_rows   = num_timesteps * NUM_NODES

    # If already built, reuse
    if os.path.exists(X_path) and os.path.exists(Y_path):
        print("Reusing existing memmap files.")
        X_mm = np.memmap(X_path, dtype='float32', mode='r', shape=(total_rows, num_features))
        Y_mm = np.memmap(Y_path, dtype='float32', mode='r', shape=(total_rows, num_targets))
        return X_mm, Y_mm

    print("Building memmap from parquet (one-time, streaming)...")
    X_mm = np.memmap(X_path, dtype='float32', mode='w+', shape=(total_rows, num_features))
    Y_mm = np.memmap(Y_path, dtype='float32', mode='w+', shape=(total_rows, num_targets))

    scanner = ds.dataset(parquet_path).scanner(
        columns=feature_cols + targets,
        batch_size=NUM_NODES * 336  # 2 weeks at a time
    )

    row = 0
    for batch in scanner.to_batches():
        df_chunk = batch.to_pandas()
        X_chunk  = df_chunk[feature_cols].astype('float32').values
        Y_chunk  = df_chunk[targets].astype('float32').values
        X_chunk  = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        Y_chunk  = np.nan_to_num(Y_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(X_chunk)
        X_mm[row:row+n] = X_chunk
        Y_mm[row:row+n] = Y_chunk
        row += n

    X_mm.flush()
    Y_mm.flush()
    print(f"Memmap built: {row} rows written.")
    return X_mm, Y_mm


class SpatioTemporalDataset(Dataset):
    def __init__(self, X_mm, Y_mm, time_bins, indices, seq_len, pred_len,
                 feature_cols, targets, scaler_X, scaler_Y):
        self.X_mm      = X_mm        # np.memmap, shape (total_rows, num_features)
        self.Y_mm      = Y_mm        # np.memmap, shape (total_rows, num_targets)
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.scaler_X  = scaler_X
        self.scaler_Y  = scaler_Y
        self.valid_indices = [i for i in indices
                              if i - seq_len >= 0 and i + pred_len <= len(time_bins)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        # Direct O(1) row slice — no scanning, no file open
        x_s = (t - self.seq_len) * NUM_NODES
        x_e = t * NUM_NODES
        X   = self.X_mm[x_s:x_e].copy()   # copy() avoids memmap reference issues
        X   = self.scaler_X.transform(X).reshape(self.seq_len, NUM_NODES, -1)

        y_s = t * NUM_NODES
        y_e = (t + self.pred_len) * NUM_NODES
        Y   = self.Y_mm[y_s:y_e].copy()
        Y   = self.scaler_Y.transform(Y).reshape(self.pred_len, NUM_NODES, -1)

        return torch.FloatTensor(X), torch.FloatTensor(Y).squeeze(0)
    
class FastDenseGCN(nn.Module):
    """
    H = A * (XW)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x shape: [Batch, Time, Nodes, Features]
        # adj shape: [Nodes, Nodes]
        
        # 1. XW -> [Batch, Time, Nodes, Out_Features]
        out = self.linear(x)
        
        # 2. messaging (aggregating neighbors) using einsum for max speed
        # i, j: nodes; b: batch, t: time, f: features
        #  A(i, j) * Out(b, t, j, f)
        out = torch.einsum('ij, btjf -> btif', adj, out)
        
        return torch.relu(out)

class CausalTCN(nn.Module):
    """Dilated causal temporal convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal: pad left only
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: [B*N, T, C] -> conv expects [B*N, C, T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x[:, :, :-self.padding]  # remove future leak from left-padding
        x = x.permute(0, 2, 1)       # back to [B*N, T, C]
        return torch.relu(self.norm(x))


class FastMultiGraphSTGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_dim):
        super().__init__()
        self.gcn_spatial = FastDenseGCN(num_features, hidden_dim)
        self.gcn_flow    = FastDenseGCN(num_features, hidden_dim)

        # Stack two TCN layers with increasing dilation to capture hourly + daily patterns
        self.tcn1 = CausalTCN(hidden_dim * 2, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalTCN(hidden_dim,     hidden_dim, kernel_size=3, dilation=4)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj_spatial, adj_flow):
        B, T, N, F = x.size()

        out_spatial = self.gcn_spatial(x, adj_spatial)  # [B, T, N, H]
        out_flow    = self.gcn_flow(x, adj_flow)
        gcn_out     = torch.cat([out_spatial, out_flow], dim=-1)  # [B, T, N, 2H]

        # Reshape for temporal conv: merge B and N into batch dim
        tcn_in = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, 2H]

        tcn_out = self.tcn1(tcn_in)          # [B*N, T, H]
        tcn_out = self.tcn2(tcn_out)         # [B*N, T, H]
        h_last  = tcn_out[:, -1, :]          # take last timestep: [B*N, H]

        out = self.fc2(self.relu(self.fc1(h_last)))
        return out.view(B, N, -1)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return mae, rmse

def train_model():

    adj_spatial, adj_flow = load_dense_graphs()
    
    parquet_path, X_mm, Y_mm, time_bins, train_idx, val_idx, test_idx, \
        feature_cols, targets, scaler_X, scaler_Y = load_and_preprocess_data()

    ds_kwargs = dict(
        X_mm=X_mm, Y_mm=Y_mm,
        time_bins=time_bins,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        feature_cols=feature_cols, targets=targets,
        scaler_X=scaler_X, scaler_Y=scaler_Y,
    )
    train_ds = SpatioTemporalDataset(indices=train_idx, **ds_kwargs)
    val_ds   = SpatioTemporalDataset(indices=val_idx,   **ds_kwargs)
    test_ds  = SpatioTemporalDataset(indices=test_idx,  **ds_kwargs)

    num_workers = 0 # min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers)

    num_features = len(feature_cols)
    num_targets  = len(targets)
    
    model = FastMultiGraphSTGCN(num_features=num_features, hidden_dim=64, out_dim=num_targets).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\n--- [2] Starting Fast Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            # 极速前向传播
            preds = model(batch_x, adj_spatial, adj_flow)
            
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x, adj_spatial, adj_flow)
                val_loss += criterion(preds, batch_y).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_stgcn_model.pth")
            print("  --> Model saved!")

    print("\n--- [3] Final Testing (Year 2025) ---")
    model.load_state_dict(torch.load("best_stgcn_model.pth"))
    model.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(DEVICE)
            preds = model(batch_x, adj_spatial, adj_flow)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, num_targets)
    all_trues = np.concatenate(all_trues, axis=0).reshape(-1, num_targets)
    
    all_preds_real = scaler_Y.inverse_transform(all_preds)
    all_trues_real = scaler_Y.inverse_transform(all_trues)
    
    demand_mae, demand_rmse = calculate_metrics(all_trues_real[:, 0], all_preds_real[:, 0])
    rev_mae, rev_rmse = calculate_metrics(all_trues_real[:, 1], all_preds_real[:, 1])
    
    print("\n" + "="*50)
    print("📈 TEST SET PERFORMANCE (Year 2025)")
    print("="*50)
    print(f"[Target 1] DEMAND (Trips per hour per node)")
    print(f"  --> MAE : {demand_mae:.2f} trips")
    print(f"  --> RMSE: {demand_rmse:.2f} trips")
    print("-" * 50)
    print(f"[Target 2] REVENUE ($ per hour per node)")
    print(f"  --> MAE : ${rev_mae:.2f}")
    print(f"  --> RMSE: ${rev_rmse:.2f}")
    print("="*50)

if __name__ == "__main__":
    train_model()