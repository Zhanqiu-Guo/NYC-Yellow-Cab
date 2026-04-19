import os
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import xgboost as xgb

import xgboost_residue as xgr

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "dataset"
PARQUET_PATH = os.path.join(DATA_DIR, "node_features_X.parquet")
NUM_NODES = 263

# Keep eval memory predictable. Override from shell if needed:
#   EVAL_BATCH_SIZE=16 EVAL_PRED_CHUNK_TIMESTEPS=24 python eval_only.py
BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
PRED_CHUNK_TIMESTEPS = int(os.getenv("EVAL_PRED_CHUNK_TIMESTEPS", "168"))

HIDDEN = 128
DEPTH = 2
NUM_WORKERS = int(os.getenv("EVAL_NUM_WORKERS", "0"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(DATA_DIR, "best_xgb_graph_residual_oof_h1_fixed.pth")
XGB_D_PATH = os.path.join(DATA_DIR, "xgb_demand_full_h1_fixed.json")
XGB_P_PATH = os.path.join(DATA_DIR, "xgb_price_full_h1_fixed.json")

BASE_VAL_MEMMAP = os.path.join(DATA_DIR, "eval_base_val_preds.dat")
BASE_TEST_MEMMAP = os.path.join(DATA_DIR, "eval_base_test_preds.dat")

print(f"Using device: {DEVICE}")


def configure_xgboost_residue_globals():
    """The helper functions in xgboost_residue read globals from that module."""
    xgr.DATA_DIR = DATA_DIR
    xgr.PARQUET_PATH = PARQUET_PATH
    xgr.NUM_NODES = NUM_NODES
    xgr.BATCH_SIZE = BATCH_SIZE
    xgr.HIDDEN = HIDDEN
    xgr.DEPTH = DEPTH
    xgr.NUM_WORKERS = NUM_WORKERS
    xgr.DEVICE = DEVICE


def existing_or_fallback_parquet_path():
    if os.path.exists(PARQUET_PATH):
        return PARQUET_PATH

    fallback = os.path.join(DATA_DIR, "node_features_X_with_airport.parquet")
    if os.path.exists(fallback):
        print(f"Parquet not found at {PARQUET_PATH}; using {fallback}")
        return fallback

    raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")


def get_eval_feature_memmap(X_raw_mm, feature_cols, total_timesteps):
    if hasattr(xgr, "build_aligned_memmap"):
        return xgr.build_aligned_memmap(X_raw_mm, feature_cols, total_timesteps)
    return X_raw_mm


def sample_chunks(sample_t, chunk_timesteps=PRED_CHUNK_TIMESTEPS):
    for start in range(0, len(sample_t), chunk_timesteps):
        end = min(start + chunk_timesteps, len(sample_t))
        yield start, end, sample_t[start:end]


def predict_base_to_memmap(model_d, model_p, X_mm, sample_t, out_path, split_name):
    shape = (len(sample_t), NUM_NODES, 2)
    preds_mm = np.memmap(out_path, dtype="float32", mode="w+", shape=shape)

    desc = f"XGB base predict {split_name}"
    total_chunks = math.ceil(len(sample_t) / PRED_CHUNK_TIMESTEPS)
    for out_s, out_e, chunk_t in tqdm(sample_chunks(sample_t), total=total_chunks, desc=desc):
        xs, xe, _, _ = xgr.rows_for_contiguous_sample_t(chunk_t)
        chunk_preds = xgr.predict_pair(model_d, model_p, X_mm[xs:xe])
        preds_mm[out_s:out_e] = chunk_preds.reshape(len(chunk_t), NUM_NODES, 2)

    preds_mm.flush()
    del preds_mm
    return np.memmap(out_path, dtype="float32", mode="r+", shape=shape)


class StreamingRegressionMetrics:
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_true_abs = 0.0
        self.sum_ape = 0.0
        self.n_ape = 0

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        diff = y_true - y_pred
        abs_diff = np.abs(diff)

        self.n += int(y_true.size)
        self.sum_abs += float(abs_diff.sum())
        self.sum_sq += float(np.square(diff).sum())
        self.sum_true_abs += float(np.abs(y_true).sum())

        mask = y_true > 1.0
        if mask.any():
            self.sum_ape += float((abs_diff[mask] / y_true[mask]).sum())
            self.n_ape += int(mask.sum())

    def compute(self):
        if self.n == 0:
            return math.nan, math.nan, math.nan, math.nan

        mae = self.sum_abs / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        mape = self.sum_ape / self.n_ape * 100.0 if self.n_ape else math.nan
        wape = self.sum_abs / (self.sum_true_abs + 1e-8) * 100.0
        return mae, rmse, mape, wape


def print_streaming_metrics(title, demand_metrics, revenue_metrics):
    d_mae, d_rmse, d_mape, d_wape = demand_metrics.compute()
    r_mae, r_rmse, r_mape, r_wape = revenue_metrics.compute()

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"[Demand ] MAE={d_mae:.3f} RMSE={d_rmse:.3f} MAPE={d_mape:.2f}% WAPE={d_wape:.2f}%")
    print(f"[Revenue] MAE={r_mae:.3f} RMSE={r_rmse:.3f} MAPE={r_mape:.2f}% WAPE={r_wape:.2f}%")
    print("=" * 70)


def eval_xgb_base_streaming(base_preds, Y_mm, sample_t, split_name):
    demand_metrics = StreamingRegressionMetrics()
    revenue_metrics = StreamingRegressionMetrics()

    for out_s, out_e, chunk_t in sample_chunks(sample_t):
        ys = (int(chunk_t[0]) + 1) * NUM_NODES
        ye = (int(chunk_t[-1]) + 2) * NUM_NODES
        y = Y_mm[ys:ye].reshape(len(chunk_t), NUM_NODES, 2)
        base = np.asarray(base_preds[out_s:out_e])

        d_true = y[..., 0]
        r_true = y[..., 1]
        d_pred = np.expm1(base[..., 0])
        p_pred = np.expm1(base[..., 1])
        r_pred = d_pred * p_pred

        demand_metrics.update(d_true, d_pred)
        revenue_metrics.update(r_true, r_pred)

    print_streaming_metrics(f"XGBoost BASE | {split_name}", demand_metrics, revenue_metrics)


@torch.inference_mode()
def eval_hybrid_streaming(model, loader, A_s, A_f, split_name):
    model.eval()
    losses = []
    demand_metrics = StreamingRegressionMetrics()
    revenue_metrics = StreamingRegressionMetrics()

    for x_local, x_global, x_static, base, d_true, r_true in tqdm(loader, desc=f"Hybrid eval {split_name}"):
        x_local = x_local.to(DEVICE)
        x_global = x_global.to(DEVICE)
        x_static = x_static.to(DEVICE)
        base = base.to(DEVICE)
        d_true = d_true.to(DEVICE)
        r_true = r_true.to(DEVICE)

        pred_log_d, pred_log_p, d_hat, p_hat, r_hat = model(
            x_local, x_global, x_static, base, A_s, A_f
        )
        loss = xgr.compute_loss(pred_log_d, pred_log_p, d_hat, p_hat, r_hat, d_true, r_true)
        losses.append(loss.item())

        demand_metrics.update(d_true.cpu().numpy(), d_hat.cpu().numpy())
        revenue_metrics.update(r_true.cpu().numpy(), r_hat.cpu().numpy())

    print_streaming_metrics(
        f"XGB + GRAPH RESIDUAL (OOF) | {split_name}",
        demand_metrics,
        revenue_metrics,
    )
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def eval_only():
    configure_xgboost_residue_globals()
    xgr.PARQUET_PATH = existing_or_fallback_parquet_path()

    dataset, time_bins, feature_cols = xgr.get_schema_and_timebins()
    local_cols, global_cols, static_cols = xgr.split_feature_groups(feature_cols)
    train_t, val_t, test_t = xgr.get_splits(time_bins)

    # Build/reuse disk-backed arrays. This scans parquet in batches, not into RAM.
    X_raw_mm, Y_mm = xgr.build_raw_memmaps(dataset, feature_cols, len(time_bins))
    X_eval_mm = get_eval_feature_memmap(X_raw_mm, feature_cols, len(time_bins))
    scaler = xgr.fit_scaler(X_eval_mm, train_t)

    if not (os.path.exists(XGB_D_PATH) and os.path.exists(XGB_P_PATH)):
        raise FileNotFoundError("XGBoost base models not found.")
    xgb_d = xgb.Booster()
    xgb_d.load_model(XGB_D_PATH)
    xgb_p = xgb.Booster()
    xgb_p.load_model(XGB_P_PATH)

    base_val = predict_base_to_memmap(xgb_d, xgb_p, X_eval_mm, val_t, BASE_VAL_MEMMAP, "VAL")
    base_test = predict_base_to_memmap(xgb_d, xgb_p, X_eval_mm, test_t, BASE_TEST_MEMMAP, "TEST")

    print("\n--- EVAL ONLY: XGBoost base ---")
    eval_xgb_base_streaming(base_val, Y_mm, val_t, "VAL")
    eval_xgb_base_streaming(base_test, Y_mm, test_t, "TEST")

    val_ds = xgr.ResidualSnapshotDataset(
        X_eval_mm, Y_mm, val_t, base_val, feature_cols, scaler,
        local_cols, global_cols, static_cols
    )
    test_ds = xgr.ResidualSnapshotDataset(
        X_eval_mm, Y_mm, test_t, base_test, feature_cols, scaler,
        local_cols, global_cols, static_cols
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    A_s, A_f = xgr.load_graphs()
    model = xgr.XGBGraphResidualNet(
        num_nodes=NUM_NODES,
        local_dim=len(local_cols),
        global_dim=len(global_cols),
        static_dim=len(static_cols),
        hidden_dim=HIDDEN,
        depth=DEPTH,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Graph checkpoint not found: {MODEL_PATH}")

    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)

    print("\n--- EVAL ONLY: XGB + graph residual ---")
    eval_hybrid_streaming(model, val_loader, A_s, A_f, "VAL")
    eval_hybrid_streaming(model, test_loader, A_s, A_f, "TEST")


if __name__ == "__main__":
    eval_only()
