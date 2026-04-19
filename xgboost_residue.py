#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "dataset"
PARQUET_PATH = os.path.join(DATA_DIR, "node_features_X_with_airport.parquet")
NUM_NODES = 263
BATCH_SIZE = 256
EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 128
DEPTH = 2
DROPOUT = 0.10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OOF settings
OOF_N_FOLDS = 5
OOF_MIN_TRAIN_FRAC = 0.40   # first 40% of 2023 only used as initial history for OOF rolling folds
EARLY_STOP_VAL_FRAC = 0.10  # internal tail split inside each training window

RAW_X_MEMMAP = os.path.join(DATA_DIR, "X_raw_memmap.dat")
Y_MEMMAP = os.path.join(DATA_DIR, "Y_memmap.dat")
SCALER_PATH = os.path.join(DATA_DIR, "xgb_graph_residual_scaler.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "best_xgb_graph_residual_oof.pth")
XGB_D_PATH = os.path.join(DATA_DIR, "xgb_demand_full.json")
XGB_P_PATH = os.path.join(DATA_DIR, "xgb_price_full.json")

ID_COLS = {"time_bin", "LocationID", "node_index"}
TARGET_COLS = ["demand", "revenue_total"]

STATIC_COLS = [
    "zone_area_sqkm",
    "dist_to_center_km",
    "is_airport_zone",
]

GLOBAL_EXACT = {
    "temperature", "wind_speed", "precipitation",
    "is_holiday", "hour", "weekday", "day_of_month", "month", "day_of_year", "year",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_24", "temp_lag_168",
    "wind_lag_1", "wind_lag_2", "wind_lag_3", "wind_lag_24", "wind_lag_168",
    "precip_lag_1", "precip_lag_2", "precip_lag_3", "precip_lag_24", "precip_lag_168",
}
GLOBAL_PREFIXES = ("ap_ewr_", "ap_jfk_", "ap_lga_", "ap_city_")

print(f"Using device: {DEVICE}")

# ============================================================
# FEATURE GROUPING
# ============================================================
def split_feature_groups(feature_cols):
    static_cols = [c for c in STATIC_COLS if c in feature_cols]
    global_cols = []
    for c in feature_cols:
        if c in static_cols:
            continue
        if c in GLOBAL_EXACT or c.startswith(GLOBAL_PREFIXES):
            global_cols.append(c)
    local_cols = [c for c in feature_cols if c not in set(static_cols + global_cols)]
    return local_cols, global_cols, static_cols


# ============================================================
# DATA PREP
# ============================================================
def get_schema_and_timebins():
    dataset = ds.dataset(PARQUET_PATH)
    time_table = dataset.to_table(columns=["time_bin"])
    time_bins = (
        time_table.column("time_bin").to_pandas()
        .drop_duplicates().sort_values().values
    )
    all_cols = [f.name for f in dataset.schema]
    feature_cols = [c for c in all_cols if c not in ID_COLS]
    return dataset, time_bins, feature_cols


def get_splits(time_bins):
    ts = pd.to_datetime(time_bins)

    sample_t = np.arange(len(ts) - 1)
    target_ts = ts[1:]

    train_start = pd.Timestamp("2023-01-08 00:00:00")
    train_end = pd.Timestamp("2025-04-30 23:00:00")
    val_start = pd.Timestamp("2025-05-01 00:00:00")
    val_end = pd.Timestamp("2025-07-15 23:00:00")
    test_start = pd.Timestamp("2025-07-16 00:00:00")

    train_t = sample_t[(target_ts >= train_start) & (target_ts <= train_end)]
    val_t = sample_t[(target_ts >= val_start) & (target_ts <= val_end)]
    test_t = sample_t[(target_ts >= test_start)]

    print(f"Samples -> Train: {len(train_t)}, Val: {len(val_t)}, Test: {len(test_t)}")
    return train_t, val_t, test_t


def build_raw_memmaps(dataset, feature_cols, total_timesteps):
    total_rows = total_timesteps * NUM_NODES
    feat_dim = len(feature_cols)

    if os.path.exists(RAW_X_MEMMAP) and os.path.exists(Y_MEMMAP):
        print("Reusing raw memmaps...")
        X_mm = np.memmap(RAW_X_MEMMAP, dtype="float32", mode="r", shape=(total_rows, feat_dim))
        Y_mm = np.memmap(Y_MEMMAP, dtype="float32", mode="r", shape=(total_rows, 2))
        return X_mm, Y_mm

    print("Building raw memmaps...")
    X_mm = np.memmap(RAW_X_MEMMAP, dtype="float32", mode="w+", shape=(total_rows, feat_dim))
    Y_mm = np.memmap(Y_MEMMAP, dtype="float32", mode="w+", shape=(total_rows, 2))

    scanner = dataset.scanner(columns=list(dict.fromkeys(feature_cols + TARGET_COLS)), batch_size=NUM_NODES * 336)
    row = 0
    for batch in tqdm(scanner.to_batches()):
        df = batch.to_pandas()
        x = df[feature_cols].astype("float32").values
        y = df[TARGET_COLS].astype("float32").values
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(df)
        X_mm[row:row+n] = x
        Y_mm[row:row+n] = y
        row += n

    X_mm.flush()
    Y_mm.flush()
    print(f"Raw memmaps written: {row} rows")
    return X_mm, Y_mm


def fit_scaler(X_mm, train_t):
    if os.path.exists(SCALER_PATH):
        print("Loading scaler...")
        return joblib.load(SCALER_PATH)

    scaler = StandardScaler()
    start = int(train_t[0]) * NUM_NODES
    end = (int(train_t[-1]) + 1) * NUM_NODES

    chunk = 262144
    print("Fitting scaler on train rows...")
    for s in tqdm(range(start, end, chunk)):
        e = min(s + chunk, end)
        x = np.array(X_mm[s:e], dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scaler.partial_fit(x)

    joblib.dump(scaler, SCALER_PATH)
    return scaler


def rows_for_contiguous_sample_t(sample_t):
    xs = int(sample_t[0]) * NUM_NODES
    xe = (int(sample_t[-1]) + 1) * NUM_NODES
    ys = (int(sample_t[0]) + 1) * NUM_NODES
    ye = (int(sample_t[-1]) + 2) * NUM_NODES
    return xs, xe, ys, ye


def internal_time_split(sample_t, val_frac=EARLY_STOP_VAL_FRAC):
    n = len(sample_t)
    n_val = max(24 * 7, int(n * val_frac))
    n_val = min(n_val, n // 3) if n >= 3 else 1
    cut = n - n_val
    cut = max(cut, 1)
    return sample_t[:cut], sample_t[cut:]


# ============================================================
# METRICS
# ============================================================
def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true > 1.0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else np.nan
    wape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100.0)
    return mae, rmse, mape, wape


def print_metrics(title, d_true, d_pred, r_true, r_pred):
    d_mae, d_rmse, d_mape, d_wape = regression_metrics(d_true, d_pred)
    r_mae, r_rmse, r_mape, r_wape = regression_metrics(r_true, r_pred)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"[Demand ] MAE={d_mae:.3f} RMSE={d_rmse:.3f} MAPE={d_mape:.2f}% WAPE={d_wape:.2f}%")
    print(f"[Revenue] MAE={r_mae:.3f} RMSE={r_rmse:.3f} MAPE={r_mape:.2f}% WAPE={r_wape:.2f}%")
    print("=" * 70)


# ============================================================
# XGBOOST BASE
# ============================================================
def xgb_params():
    return dict(
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=8,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        reg_alpha=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def build_target_arrays(Y):
    y_d = np.log1p(Y[:, 0])
    y_p = np.log1p(Y[:, 1] / np.clip(Y[:, 0], 1.0, None))
    return y_d, y_p


def fit_xgb_pair_on_contiguous_block(X_mm, Y_mm, train_block_t):
    sub_train_t, sub_val_t = internal_time_split(train_block_t)

    tr_xs, tr_xe, tr_ys, tr_ye = rows_for_contiguous_sample_t(sub_train_t)
    va_xs, va_xe, va_ys, va_ye = rows_for_contiguous_sample_t(sub_val_t)

    X_train = X_mm[tr_xs:tr_xe]
    Y_train = Y_mm[tr_ys:tr_ye]
    X_val = X_mm[va_xs:va_xe]
    Y_val = Y_mm[va_ys:va_ye]

    y_train_d, y_train_p = build_target_arrays(Y_train)
    y_val_d, y_val_p = build_target_arrays(Y_val)

    dtrain_d = xgb.QuantileDMatrix(X_train, y_train_d)
    dval_d = xgb.QuantileDMatrix(X_val, y_val_d)
    dtrain_p = xgb.QuantileDMatrix(X_train, y_train_p)
    dval_p = xgb.QuantileDMatrix(X_val, y_val_p)

    params = xgb_params()
    model_d = xgb.train(
        params,
        dtrain_d,
        num_boost_round=1200,
        evals=[(dtrain_d, "train"), (dval_d, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    model_p = xgb.train(
        params,
        dtrain_p,
        num_boost_round=1200,
        evals=[(dtrain_p, "train"), (dval_p, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model_d, model_p


def train_full_xgb_base(X_mm, Y_mm, train_t):
    print("\n--- Training FULL XGBoost base on 2023 ---")
    model_d, model_p = fit_xgb_pair_on_contiguous_block(X_mm, Y_mm, train_t)
    model_d.save_model(XGB_D_PATH)
    model_p.save_model(XGB_P_PATH)
    return model_d, model_p


def predict_pair(model_d, model_p, X):
    pred_log_d = model_d.predict(xgb.QuantileDMatrix(X))
    pred_log_p = model_p.predict(xgb.QuantileDMatrix(X))
    return np.stack([pred_log_d, pred_log_p], axis=-1).astype(np.float32)


def predict_base_for_contiguous_split(model_d, model_p, X_mm, sample_t):
    xs, xe, _, _ = rows_for_contiguous_sample_t(sample_t)
    X = X_mm[xs:xe]
    preds = predict_pair(model_d, model_p, X)
    return preds.reshape(len(sample_t), NUM_NODES, 2)


def make_time_based_oof_predictions(X_mm, Y_mm, train_t, n_folds=OOF_N_FOLDS, min_train_frac=OOF_MIN_TRAIN_FRAC):
    n = len(train_t)
    init_end = max(24 * 14, int(n * min_train_frac))
    init_end = min(init_end, n - n_folds)
    assert init_end > 0 and init_end < n, "Not enough train samples for OOF."

    remain = n - init_end
    fold_sizes = np.full(n_folds, remain // n_folds, dtype=int)
    fold_sizes[:remain % n_folds] += 1

    oof_preds_list = []
    oof_t_list = []
    start = init_end

    print("\n--- Building TIME-BASED OOF base predictions ---")
    print(f"Initial history only block: first {init_end} train samples are not used for residual training")

    for i, fold_size in enumerate(fold_sizes):
        end = start + fold_size
        holdout_t = train_t[start:end]
        prefix_train_t = train_t[:start]
        print(f"Fold {i+1}/{n_folds}: train_prefix={len(prefix_train_t)} holdout={len(holdout_t)}")

        model_d, model_p = fit_xgb_pair_on_contiguous_block(X_mm, Y_mm, prefix_train_t)
        fold_preds = predict_base_for_contiguous_split(model_d, model_p, X_mm, holdout_t)

        oof_preds_list.append(fold_preds)
        oof_t_list.append(holdout_t)
        start = end

    oof_t = np.concatenate(oof_t_list)
    oof_preds = np.concatenate(oof_preds_list, axis=0)
    return oof_t, oof_preds


def eval_xgb_base(base_preds, Y_mm, sample_t, split_name):
    ys = (int(sample_t[0]) + 1) * NUM_NODES
    ye = (int(sample_t[-1]) + 2) * NUM_NODES
    Y = Y_mm[ys:ye].reshape(len(sample_t), NUM_NODES, 2)

    d_true = Y[..., 0].reshape(-1)
    r_true = Y[..., 1].reshape(-1)
    d_pred = np.expm1(base_preds[..., 0]).reshape(-1)
    p_pred = np.expm1(base_preds[..., 1]).reshape(-1)
    r_pred = d_pred * p_pred

    print_metrics(f"XGBoost BASE | {split_name}", d_true, d_pred, r_true, r_pred)


# ============================================================
# RESIDUAL DATASET
# ============================================================
class ResidualSnapshotDataset(Dataset):
    def __init__(self, X_mm, Y_mm, sample_t, base_preds, feature_cols, scaler, local_cols, global_cols, static_cols):
        self.X_mm = X_mm
        self.Y_mm = Y_mm
        self.sample_t = sample_t
        self.base_preds = base_preds
        self.mean_ = scaler.mean_.astype(np.float32)
        self.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

        name_to_idx = {c: i for i, c in enumerate(feature_cols)}
        self.local_idx = np.array([name_to_idx[c] for c in local_cols], dtype=np.int64)
        self.global_idx = np.array([name_to_idx[c] for c in global_cols], dtype=np.int64)
        self.static_idx = np.array([name_to_idx[c] for c in static_cols], dtype=np.int64)

    def __len__(self):
        return len(self.sample_t)

    def __getitem__(self, i):
        t = int(self.sample_t[i])
        x_start = t * NUM_NODES
        y_start = (t + 1) * NUM_NODES

        x = self.X_mm[x_start:x_start + NUM_NODES].copy().reshape(NUM_NODES, -1)
        x = (x - self.mean_) / self.scale_
        y = self.Y_mm[y_start:y_start + NUM_NODES].copy().reshape(NUM_NODES, 2)
        base = self.base_preds[i]

        x_local = x[:, self.local_idx]
        x_static = x[:, self.static_idx]
        x_global = x[0, self.global_idx]
        demand = y[:, 0]
        revenue = y[:, 1]

        return (
            torch.from_numpy(x_local).float(),
            torch.from_numpy(x_global).float(),
            torch.from_numpy(x_static).float(),
            torch.from_numpy(base).float(),
            torch.from_numpy(demand).float(),
            torch.from_numpy(revenue).float(),
        )


# ============================================================
# GRAPH MODEL
# ============================================================
def normalize_adj(A):
    A = A.astype(np.float32)
    np.fill_diagonal(A, 1.0)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return A / row_sum


def load_graphs():
    A_s = normalize_adj(np.load(os.path.join(DATA_DIR, "adj_spatial.npy")))
    A_f = normalize_adj(np.load(os.path.join(DATA_DIR, "adj_flow.npy")))
    return torch.from_numpy(A_s).float().to(DEVICE), torch.from_numpy(A_f).float().to(DEVICE)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualGraphBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, m, c):
        c_rep = c.unsqueeze(1).expand(-1, h.size(1), -1)
        z = torch.cat([h, m, c_rep], dim=-1)
        upd = self.fc2(self.dropout(F.gelu(self.fc1(z))))
        g = torch.sigmoid(self.gate(z))
        return self.norm(h + g * upd)


class XGBGraphResidualNet(nn.Module):
    def __init__(self, num_nodes, local_dim, global_dim, static_dim, hidden_dim=128, depth=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_emb = nn.Embedding(num_nodes, 16)

        self.local_enc = MLP(local_dim + static_dim + 2, hidden_dim, hidden_dim)
        self.global_enc = MLP(global_dim, hidden_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim + 16, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim + 16, hidden_dim)
        self.mix_mlp = nn.Linear(hidden_dim + hidden_dim + 16, 3)
        self.blocks = nn.ModuleList([ResidualGraphBlock(hidden_dim) for _ in range(depth)])

        self.delta_head = MLP(hidden_dim * 2, hidden_dim, 2)

    def learned_adj(self, h, node_id_embed):
        nid = node_id_embed.unsqueeze(0).expand(h.size(0), -1, -1)
        hk = torch.cat([h, nid], dim=-1)
        q = self.q_proj(hk)
        k = self.k_proj(hk)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        return torch.softmax(scores, dim=-1)

    def mix_adj(self, A_s, A_f, A_l, h, c, node_id_embed):
        B, N, _ = h.shape
        c_rep = c.unsqueeze(1).expand(-1, N, -1)
        nid = node_id_embed.unsqueeze(0).expand(B, -1, -1)
        w = torch.softmax(self.mix_mlp(torch.cat([h, c_rep, nid], dim=-1)), dim=-1)
        A_s_b = A_s.unsqueeze(0).expand(B, -1, -1)
        A_f_b = A_f.unsqueeze(0).expand(B, -1, -1)
        A = (
            w[..., 0].unsqueeze(-1) * A_s_b +
            w[..., 1].unsqueeze(-1) * A_f_b +
            w[..., 2].unsqueeze(-1) * A_l
        )
        return A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def forward(self, x_local, x_global, x_static, base_log_preds, A_s, A_f):
        B, N, _ = x_local.shape
        node_ids = torch.arange(N, device=x_local.device)
        nid = self.node_emb(node_ids)

        h = self.local_enc(torch.cat([x_local, x_static, base_log_preds], dim=-1))
        c = self.global_enc(x_global)

        for blk in self.blocks:
            A_l = self.learned_adj(h, nid)
            A = self.mix_adj(A_s, A_f, A_l, h, c, nid)
            m = torch.matmul(A, h)
            h = blk(h, m, c)

        c_rep = c.unsqueeze(1).expand(-1, N, -1)
        delta = self.delta_head(torch.cat([h, c_rep], dim=-1))

        pred_log_d = base_log_preds[..., 0] + delta[..., 0]
        pred_log_p = base_log_preds[..., 1] + delta[..., 1]

        d_hat = torch.expm1(pred_log_d).clamp(min=0.0)
        p_hat = torch.expm1(pred_log_p).clamp(min=0.0)
        r_hat = d_hat * p_hat
        return pred_log_d, pred_log_p, d_hat, p_hat, r_hat


# ============================================================
# LOSS / EVAL
# ============================================================
def compute_loss(pred_log_d, pred_log_p, d_hat, p_hat, r_hat, d_true, r_true):
    true_log_d = torch.log1p(d_true)
    true_p = r_true / d_true.clamp(min=1.0)
    true_log_p = torch.log1p(true_p)
    true_log_r = torch.log1p(r_true)
    pred_log_r = torch.log1p(r_hat)

    d_loss = F.smooth_l1_loss(pred_log_d, true_log_d, beta=0.20)

    price_mask = (d_true > 0).float()
    p_loss_all = F.smooth_l1_loss(pred_log_p, true_log_p, beta=0.20, reduction="none")
    p_loss = (p_loss_all * price_mask).sum() / price_mask.sum().clamp(min=1.0)

    r_loss = F.smooth_l1_loss(pred_log_r, true_log_r, beta=0.20)

    agg_d = F.smooth_l1_loss(torch.log1p(d_hat.sum(dim=1)), torch.log1p(d_true.sum(dim=1)), beta=0.20)
    agg_r = F.smooth_l1_loss(torch.log1p(r_hat.sum(dim=1)), torch.log1p(r_true.sum(dim=1)), beta=0.20)

    return d_loss + 0.5 * p_loss + 0.75 * r_loss + 0.25 * (agg_d + agg_r)


@torch.no_grad()
def eval_hybrid(model, loader, A_s, A_f, split_name):
    model.eval()
    losses = []
    d_true_all, d_pred_all = [], []
    r_true_all, r_pred_all = [], []

    for x_local, x_global, x_static, base, d_true, r_true in loader:
        x_local = x_local.to(DEVICE)
        x_global = x_global.to(DEVICE)
        x_static = x_static.to(DEVICE)
        base = base.to(DEVICE)
        d_true = d_true.to(DEVICE)
        r_true = r_true.to(DEVICE)

        pred_log_d, pred_log_p, d_hat, p_hat, r_hat = model(x_local, x_global, x_static, base, A_s, A_f)
        loss = compute_loss(pred_log_d, pred_log_p, d_hat, p_hat, r_hat, d_true, r_true)
        losses.append(loss.item())

        d_true_all.append(d_true.cpu().numpy())
        d_pred_all.append(d_hat.cpu().numpy())
        r_true_all.append(r_true.cpu().numpy())
        r_pred_all.append(r_hat.cpu().numpy())

    d_true_all = np.concatenate(d_true_all, axis=0).reshape(-1)
    d_pred_all = np.concatenate(d_pred_all, axis=0).reshape(-1)
    r_true_all = np.concatenate(r_true_all, axis=0).reshape(-1)
    r_pred_all = np.concatenate(r_pred_all, axis=0).reshape(-1)

    print_metrics(f"XGB + GRAPH RESIDUAL (OOF) | {split_name}", d_true_all, d_pred_all, r_true_all, r_pred_all)
    return float(np.mean(losses))


# ============================================================
# MAIN
# ============================================================
def main():
    dataset, time_bins, feature_cols = get_schema_and_timebins()
    local_cols, global_cols, static_cols = split_feature_groups(feature_cols)
    train_t, val_t, test_t = get_splits(time_bins)

    X_mm, Y_mm = build_raw_memmaps(dataset, feature_cols, len(time_bins))
    scaler = fit_scaler(X_mm, train_t)

    # 1) Full XGB base for downstream validation / testing
    if os.path.exists(XGB_D_PATH) and os.path.exists(XGB_P_PATH):
        print("Loading full XGBoost base models...")
        xgb_d = xgb.Booster(); xgb_d.load_model(XGB_D_PATH)
        xgb_p = xgb.Booster(); xgb_p.load_model(XGB_P_PATH)
    else:
        xgb_d, xgb_p = train_full_xgb_base(X_mm, Y_mm, train_t)

    base_val = predict_base_for_contiguous_split(xgb_d, xgb_p, X_mm, val_t)
    base_test = predict_base_for_contiguous_split(xgb_d, xgb_p, X_mm, test_t)
    eval_xgb_base(base_val, Y_mm, val_t, "VAL")
    eval_xgb_base(base_test, Y_mm, test_t, "TEST")

    # 2) Time-based OOF base predictions for residual training only
    oof_train_t, base_oof_train = make_time_based_oof_predictions(X_mm, Y_mm, train_t)
    print(f"Residual training samples after OOF warmup drop: {len(oof_train_t)}")

    # 3) Train graph residual on honest OOF train preds, evaluate on full-base val/test preds
    train_ds = ResidualSnapshotDataset(X_mm, Y_mm, oof_train_t, base_oof_train, feature_cols, scaler, local_cols, global_cols, static_cols)
    val_ds = ResidualSnapshotDataset(X_mm, Y_mm, val_t, base_val, feature_cols, scaler, local_cols, global_cols, static_cols)
    test_ds = ResidualSnapshotDataset(X_mm, Y_mm, test_t, base_test, feature_cols, scaler, local_cols, global_cols, static_cols)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    A_s, A_f = load_graphs()
    model = XGBGraphResidualNet(
        num_nodes=NUM_NODES,
        local_dim=len(local_cols),
        global_dim=len(global_cols),
        static_dim=len(static_cols),
        hidden_dim=HIDDEN,
        depth=DEPTH,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    bad_epochs = 0
    patience = 6

    print("\n--- Training graph residual on top of TIME-BASED OOF XGB base ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}")
        for x_local, x_global, x_static, base, d_true, r_true in pbar:
            x_local = x_local.to(DEVICE, non_blocking=True)
            x_global = x_global.to(DEVICE, non_blocking=True)
            x_static = x_static.to(DEVICE, non_blocking=True)
            base = base.to(DEVICE, non_blocking=True)
            d_true = d_true.to(DEVICE, non_blocking=True)
            r_true = r_true.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred_log_d, pred_log_p, d_hat, p_hat, r_hat = model(x_local, x_global, x_static, base, A_s, A_f)
            loss = compute_loss(pred_log_d, pred_log_p, d_hat, p_hat, r_hat, d_true, r_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_loss = eval_hybrid(model, val_loader, A_s, A_f, "VAL")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  --> saved best OOF graph residual model")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print("\n--- Final test for OOF hybrid model ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    eval_hybrid(model, test_loader, A_s, A_f, "TEST")


if __name__ == "__main__":
    main()
