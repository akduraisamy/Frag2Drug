"""
=============================================================
 Frag2Drug  |  Step 2 — Feature Engineering, Training & Evaluation
=============================================================
 Author  : Dr. Amudha Kumari Duraisamy
 Input   : results/data_processing/X.npy      raw spectral features (N x 1000)
           results/data_processing/y.npy      labels [logP, QED, MW]
           results/data_processing/meta.csv   compound metadata
 Models  : Random Forest + Feedforward Neural Network
 Output  : results/model_training/models/    saved models (.pkl, .pt, scalers)
           results/model_training/results/   plots, metrics CSV
           mlruns/                           MLflow experiment tracking
 Usage   : python step2_train_evaluate.py           # train only
           python step2_train_evaluate.py --shap    # train + NN SHAP plots
=============================================================
Requirements:
    conda install -c conda-forge rdkit
    pip install torch scikit-learn matchms seaborn joblib tqdm mlflow shap
"""

import os
import warnings

# Must be set before importing torch or numpy's BLAS layer
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"]             = "1"
os.environ["MKL_NUM_THREADS"]             = "1"
os.environ["OPENBLAS_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]         = "1"

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids macOS display segfault

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

# ── Paths ───────────────────────────────────────────────────
DATA_DIR   = Path("results/data_processing")
OUT_DIR    = Path("results/model_training")
MODEL_DIR  = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"
for d in [MODEL_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────
TARGET_NAMES = ["logP", "QED", "MW"]
RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15

# Refilter thresholds
MIN_ACTIVE_BINS = 10      # min non-zero m/z bins per spectrum
LOGP_MIN, LOGP_MAX = -5, 10
MW_MIN,   MW_MAX   = 100, 1000
MZ_MIN,   MZ_MAX   = 50,  1000

# Random Forest
RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = None   # fully grown trees
RF_MIN_SAMPLES   = 2
RF_N_JOBS        = -1

# Neural Network
NN_HIDDEN     = [512, 256, 128]
NN_EPOCHS     = 100
NN_BATCH_SIZE = 256        # smaller = less memory pressure = no segfault
NN_LR         = 1e-3
NN_DROPOUT    = 0.3
NN_PATIENCE   = 15


# ══════════════════════════════════════════════════════════
#  STEP A — REFILTER  (integrated from refilter.py)
# ══════════════════════════════════════════════════════════

def refilter_and_featurise(
    x_path: Path = Path("results/data_processing") / "X.npy",
    y_path: Path = Path("results/data_processing") / "y.npy",
    meta_path: Path = Path("results/data_processing") / "meta.csv",
    force: bool = False,
) -> tuple:
    """
    Load raw X/y from Step 1, apply quality filters, add 6 summary
    spectral features, and return clean arrays.

    Caches result to data/X_clean.npy so re-runs are instant.
    Set force=True to recompute from scratch.

    Filters applied
    ---------------
    1. Min active bins  >= MIN_ACTIVE_BINS  (removes sparse spectra)
    2. logP in [LOGP_MIN, LOGP_MAX]         (drug-like range)
    3. MW   in [MW_MIN,   MW_MAX]           (drug-like range)

    Added features (appended after the 1000 bins)
    -----------------------------------------------
    TIC        total intensity (sum of all bins)
    n_peaks    number of non-zero bins
    mean_mz    intensity-weighted mean m/z
    max_bin    index of most intense bin
    frac_low   fraction of intensity below 200 Da
    frac_high  fraction of intensity above 600 Da
    """
    cache_x = DATA_DIR / "X_clean.npy"
    cache_y = DATA_DIR / "y_clean.npy"
    cache_m = DATA_DIR / "meta_clean.csv"

    if not force and cache_x.exists() and cache_y.exists():
        print("[refilter] Loading cached clean arrays ...")
        X    = np.load(cache_x)
        y    = np.load(cache_y)
        meta = pd.read_csv(cache_m)
        print(f"  X: {X.shape}   y: {y.shape}  (from cache)")
        return X, y, meta

    print("[refilter] Loading raw arrays from Step 1 ...")
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    meta  = pd.read_csv(meta_path)
    print(f"  Raw: {X_raw.shape[0]:,} spectra")

    # ── Filters ───────────────────────────────────────────
    n_active = (X_raw > 0).sum(axis=1)
    m1 = n_active >= MIN_ACTIVE_BINS
    m2 = (y_raw[:, 0] >= LOGP_MIN) & (y_raw[:, 0] <= LOGP_MAX)
    m3 = (y_raw[:, 2] >= MW_MIN)   & (y_raw[:, 2] <= MW_MAX)
    mask = m1 & m2 & m3

    print(f"  Removed sparse (<{MIN_ACTIVE_BINS} bins) : {(~m1).sum():>6,}")
    print(f"  Removed logP outliers             : {(m1 & ~m2).sum():>6,}")
    print(f"  Removed MW outliers               : {(m1 & m2 & ~m3).sum():>6,}")
    print(f"  Kept                              : {mask.sum():>6,}")

    X_f    = X_raw[mask]
    y_f    = y_raw[mask]
    meta_f = meta[mask].reset_index(drop=True)

    # ── Add 6 summary features ────────────────────────────
    n_bins     = X_f.shape[1]
    mz_centres = np.linspace(MZ_MIN, MZ_MAX, n_bins)
    low_mask   = mz_centres < 200
    high_mask  = mz_centres > 600

    tic       = X_f.sum(axis=1, keepdims=True)
    n_peaks   = (X_f > 0).sum(axis=1, keepdims=True).astype(np.float32)
    mean_mz   = (X_f * mz_centres).sum(axis=1, keepdims=True)
    max_bin   = X_f.argmax(axis=1, keepdims=True).astype(np.float32)
    frac_low  = X_f[:, low_mask].sum(axis=1, keepdims=True)
    frac_high = X_f[:, high_mask].sum(axis=1, keepdims=True)

    precursor_mz = pd.to_numeric(meta_f["precursor_mz"], errors="coerce").fillna(0.0).values.reshape(-1, 1).astype(np.float32)

    X_aug = np.hstack([X_f, tic, n_peaks, mean_mz,
                       max_bin, frac_low, frac_high,
                       precursor_mz]).astype(np.float32)
    print(f"  Feature matrix : {X_aug.shape}  (1000 bins + 6 summary + precursor_mz)")

    # ── Label stats ───────────────────────────────────────
    print("\n  Label statistics after filtering:")
    for i, name in enumerate(TARGET_NAMES):
        col = y_f[:, i]
        print(f"    {name:<6} mean={col.mean():7.3f}  std={col.std():6.3f}"
              f"  min={col.min():7.3f}  max={col.max():7.3f}")

    # ── Cache ─────────────────────────────────────────────
    np.save(cache_x, X_aug)
    np.save(cache_y, y_f)
    meta_f.to_csv(cache_m, index=False)
    print(f"\n  Cached → {cache_x}  {cache_y}  {cache_m}")

    return X_aug, y_f, meta_f


# ══════════════════════════════════════════════════════════
#  STEP B — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════

def split_and_scale(X, y):
    """
    70 / 15 / 15 split, then StandardScaler fitted on train only.
    Returns both raw (for RF) and scaled (for NN) versions.
    """
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    print(f"\n[split]  Train:{X_train.shape[0]:,}  "
          f"Val:{X_val.shape[0]:,}  Test:{X_test.shape[0]:,}")

    return (X_train,   X_val,   X_test,
            X_train_s, X_val_s, X_test_s,
            y_train,   y_val,   y_test)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def metrics(y_true, y_pred, target: str) -> dict:
    return {
        "target": target,
        "R2"    : r2_score(y_true, y_pred),
        "MAE"   : mean_absolute_error(y_true, y_pred),
        "RMSE"  : np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ══════════════════════════════════════════════════════════
#  STEP C — RANDOM FOREST
# ══════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """
    One RF per target — tree models work best on unscaled features.
    MAX_DEPTH=None lets each tree grow until leaves are pure,
    which is important for capturing non-linear fragment patterns.
    """
    print("\n" + "="*52)
    print("  RANDOM FOREST")
    print("="*52)
    all_metrics, all_preds, rf_models = [], [], []

    with mlflow.start_run(run_name="RandomForest", nested=True):
        mlflow.log_params({
            "rf_n_estimators" : RF_N_ESTIMATORS,
            "rf_max_depth"    : str(RF_MAX_DEPTH),
            "rf_min_samples"  : RF_MIN_SAMPLES,
            "train_size"      : X_train.shape[0],
            "n_features"      : X_train.shape[1],
        })

        for i, target in enumerate(TARGET_NAMES):
            print(f"\n  [{i+1}/3] {target} ...", flush=True)
            rf = RandomForestRegressor(
                n_estimators     = RF_N_ESTIMATORS,
                max_depth        = RF_MAX_DEPTH,
                min_samples_leaf = RF_MIN_SAMPLES,
                n_jobs           = RF_N_JOBS,
                random_state     = RANDOM_STATE,
            )
            rf.fit(X_train, y_train[:, i])
            preds = rf.predict(X_test)
            all_preds.append(preds)
            rf_models.append(rf)

            m = metrics(y_test[:, i], preds, target)
            all_metrics.append(m)
            print(f"    R²={m['R2']:.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

            mlflow.log_metrics({
                f"rf_{target}_R2"  : round(m["R2"],  4),
                f"rf_{target}_MAE" : round(m["MAE"], 4),
                f"rf_{target}_RMSE": round(m["RMSE"],4),
            })

            joblib.dump(rf, MODEL_DIR / f"rf_{target}.pkl")
            imp = pd.Series(rf.feature_importances_, name="importance").nlargest(50)
            imp.to_csv(RESULT_DIR / f"rf_importance_{target}.csv")

    return {"metrics": all_metrics, "preds": all_preds, "models": rf_models}


# ══════════════════════════════════════════════════════════
#  STEP D — NEURAL NETWORK
# ══════════════════════════════════════════════════════════

def train_neural_network(X_train_s, y_train,
                         X_val_s,   y_val,
                         X_test_s,  y_test) -> dict:
    """
    Multi-output feedforward NN. Predicts logP, QED, MW jointly.

    Key fixes vs previous version
    ------------------------------
    * torch.from_numpy()  instead of torch.tensor()
      — avoids the macOS segfault caused by copying large arrays
    * batch_size=256      instead of 512 — less memory pressure
    * device = "cpu"      MPS disabled (known crash on Apple M-series
                          with this data shape + BatchNorm combo)
    * y StandardScaler    normalises targets so MW (100-1000) does not
                          dominate the joint MSE loss over logP (-5..10)
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("\n  [skip] PyTorch not installed — pip install torch")
        return None

    torch.set_num_threads(1)          # prevents OMP fork-segfault on macOS
    torch.set_num_interop_threads(1)

    print("\n" + "="*52)
    print("  NEURAL NETWORK  (PyTorch)")
    print("="*52)

    device = "cpu"   # MPS segfaults with BatchNorm on large inputs
    print(f"  Device : {device}")
    print(f"  Input  : {X_train_s.shape[1]} features")
    print(f"  Output : {len(TARGET_NAMES)} targets")

    mlflow_run = mlflow.start_run(run_name="NeuralNetwork", nested=True)
    mlflow.log_params({
        "nn_hidden"     : str(NN_HIDDEN),
        "nn_epochs"     : NN_EPOCHS,
        "nn_batch_size" : NN_BATCH_SIZE,
        "nn_lr"         : NN_LR,
        "nn_dropout"    : NN_DROPOUT,
        "nn_patience"   : NN_PATIENCE,
        "train_size"    : X_train_s.shape[0],
        "n_features"    : X_train_s.shape[1],
    })

    # ── Normalise targets so all three contribute equally to MSE ──
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    joblib.dump(y_scaler, MODEL_DIR / "y_scaler.pkl")

    # ── Convert — use from_numpy to avoid copy-related crash ──
    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    X_tr = to_t(X_train_s);  y_tr = to_t(y_train_s)
    X_v  = to_t(X_val_s);    y_v  = to_t(y_val_s)
    X_te = to_t(X_test_s)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=NN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 0 = no subprocess — avoids macOS fork issues
    )

    # ── Architecture ──────────────────────────────────────
    class PropNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims  = [X_train_s.shape[1]] + NN_HIDDEN
            layers = []
            for a, b in zip(dims, dims[1:]):
                layers += [nn.Linear(a, b),
                           nn.BatchNorm1d(b),
                           nn.ReLU(),
                           nn.Dropout(NN_DROPOUT)]
            layers.append(nn.Linear(dims[-1], len(TARGET_NAMES)))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model     = PropNet()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=6, factor=0.5
    )
    loss_fn   = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────
    train_losses, val_losses = [], []
    best_val, best_state     = float("inf"), None
    patience_ctr             = 0

    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>9}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*9}")

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()
        batch_loss = []
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            batch_loss.append(loss.item())
        t_loss = float(np.mean(batch_loss))

        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(X_v), y_v).item()

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val - 1e-6:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimiser.param_groups[0]["lr"]
            print(f"  {epoch:>5}  {t_loss:>10.5f}  {v_loss:>10.5f}  {lr:>9.2e}")

        mlflow.log_metrics({"nn_train_mse": t_loss, "nn_val_mse": v_loss}, step=epoch)

        if patience_ctr >= NN_PATIENCE:
            print(f"  Early stop at epoch {epoch}  (best val={best_val:.5f})")
            break

    # ── Evaluate ──────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "nn_best.pt")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_te).numpy()

    # Inverse-transform back to original units for interpretable metrics
    preds_np = y_scaler.inverse_transform(preds_scaled)

    all_metrics, all_preds = [], []
    print()
    for i, target in enumerate(TARGET_NAMES):
        m = metrics(y_test[:, i], preds_np[:, i], target)
        all_metrics.append(m)
        all_preds.append(preds_np[:, i])
        print(f"  {target:<6}  R²={m['R2']:.4f}  "
              f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

        mlflow.log_metrics({
            f"nn_{target}_R2"  : round(m["R2"],  4),
            f"nn_{target}_MAE" : round(m["MAE"], 4),
            f"nn_{target}_RMSE": round(m["RMSE"],4),
        })

    mlflow_run.__exit__(None, None, None)

    return {"metrics"      : all_metrics,
            "preds"        : all_preds,
            "train_losses" : train_losses,
            "val_losses"   : val_losses,
            "model"        : model,
            "y_scaler"     : y_scaler}


# ══════════════════════════════════════════════════════════
#  STEP E — SHAP ANALYSIS
# ══════════════════════════════════════════════════════════

def make_feature_names(n_bins: int = 1000) -> list:
    """
    Human-readable names for all 1007 features.
    First 1000: m/z bin centres (e.g. 'mz_50.9')
    Last 7    : summary features + precursor_mz
    """
    mz_centres = np.linspace(50, 1000, n_bins)
    names = [f"mz_{v:.1f}" for v in mz_centres]
    names += ["TIC", "n_peaks", "mean_mz", "max_bin",
              "frac_low", "frac_high", "precursor_mz"]
    return names


def compute_shap_rf(rf_models: list, X_test: np.ndarray,
                    sample_n: int = 100) -> None:
    """
    SHAP TreeExplainer for each RF model.
    Saves a beeswarm summary plot per target to RESULT_DIR.
    Uses a small random sample + approximate mode for speed.
    (300 trees x 1007 features is slow without approximation)
    """
    try:
        import shap
    except ImportError:
        print("[shap] shap not installed — pip install shap")
        return

    print("\n" + "="*52)
    print("  SHAP — Random Forest")
    print("="*52)

    feat_names = make_feature_names()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), min(sample_n, len(X_test)), replace=False)
    X_sample = X_test[idx]

    for rf, target in zip(rf_models, TARGET_NAMES):
        print(f"  Computing SHAP for RF {target} (n={len(X_sample)}) ...", flush=True)
        explainer   = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feat_names,
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Feature Importance — RF {target}", fontsize=12)
        plt.tight_layout()
        out = RESULT_DIR / f"shap_rf_{target}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [shap] saved → {out}")


def compute_shap_nn(nn_model, X_train_s: np.ndarray,
                    X_test_s: np.ndarray, sample_n: int = 200) -> None:
    """
    SHAP GradientExplainer for the Neural Network.
    Uses a small background dataset from training data.
    Saves a beeswarm summary plot per target to RESULT_DIR.
    """
    try:
        import shap
        import torch
    except ImportError:
        print("[shap] shap or torch not installed")
        return

    print("\n" + "="*52)
    print("  SHAP — Neural Network  (GradientExplainer)")
    print("="*52)

    feat_names = make_feature_names()
    rng = np.random.default_rng(42)

    # Small background: 100 training points
    bg_idx = rng.choice(len(X_train_s), min(100, len(X_train_s)), replace=False)
    background = torch.from_numpy(
        np.ascontiguousarray(X_train_s[bg_idx])
    ).float()

    # Test sample
    te_idx  = rng.choice(len(X_test_s), min(sample_n, len(X_test_s)), replace=False)
    X_sample = torch.from_numpy(
        np.ascontiguousarray(X_test_s[te_idx])
    ).float()

    nn_model.eval()
    explainer   = shap.GradientExplainer(nn_model, background)
    shap_vals_raw = explainer.shap_values(X_sample)

    # GradientExplainer return shape varies by SHAP version — normalise to
    # a list of (n_samples, n_features) arrays, one per output target.
    if isinstance(shap_vals_raw, list):
        shap_list = shap_vals_raw                          # already a list
    elif isinstance(shap_vals_raw, np.ndarray) and shap_vals_raw.ndim == 3:
        if shap_vals_raw.shape[0] == len(TARGET_NAMES):
            shap_list = [shap_vals_raw[i] for i in range(len(TARGET_NAMES))]
        else:
            shap_list = [shap_vals_raw[:, :, i] for i in range(len(TARGET_NAMES))]
    else:
        shap_list = [shap_vals_raw] * len(TARGET_NAMES)   # single-output fallback

    X_np = X_sample.detach().numpy()
    for i, target in enumerate(TARGET_NAMES):
        print(f"  Plotting SHAP for NN {target} ...", flush=True)
        sv = shap_list[i]
        if sv.shape != X_np.shape:
            print(f"  [shap] shape mismatch {sv.shape} vs {X_np.shape} — skipping {target}")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            sv, X_np,
            feature_names=feat_names,
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP Feature Importance — NN {target}", fontsize=12)
        plt.tight_layout()
        out = RESULT_DIR / f"shap_nn_{target}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [shap] saved → {out}")


# ══════════════════════════════════════════════════════════
#  STEP F — PLOTS
# ══════════════════════════════════════════════════════════

def plot_predictions(y_test, rf_res, nn_res=None):
    n_rows = 1 + (nn_res is not None)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(14, 4.5 * n_rows),
                             squeeze=False)
    rows   = [("Random Forest", rf_res["preds"],  "#4C72B0")]
    if nn_res:
        rows += [("Neural Network", nn_res["preds"], "#DD8452")]

    for row_i, (name, preds, color) in enumerate(rows):
        for col_i, target in enumerate(TARGET_NAMES):
            ax = axes[row_i][col_i]
            yt = y_test[:, col_i]
            yp = preds[col_i]
            r2 = r2_score(yt, yp)
            idx = np.random.choice(len(yt), min(3000, len(yt)), replace=False)
            ax.scatter(yt[idx], yp[idx], alpha=0.25, s=6,
                       color=color, linewidths=0)
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel(f"Actual {target}")
            ax.set_ylabel(f"Predicted {target}")
            ax.set_title(f"{name} — {target}  R²={r2:.3f}")

    plt.tight_layout()
    out = RESULT_DIR / "predicted_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


def plot_nn_loss(nn_res):
    if nn_res is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    e = range(1, len(nn_res["train_losses"]) + 1)
    ax.plot(e, nn_res["train_losses"], label="Train MSE")
    ax.plot(e, nn_res["val_losses"],   label="Val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Neural Network — training curve")
    ax.legend()
    out = RESULT_DIR / "nn_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


def plot_metrics_bar(rf_res, nn_res=None):
    rows = [{"Model": "Random Forest", **m} for m in rf_res["metrics"]]
    if nn_res:
        rows += [{"Model": "Neural Network", **m} for m in nn_res["metrics"]]
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "metrics_summary.csv", index=False)

    palette = {"Random Forest": "#4C72B0", "Neural Network": "#DD8452"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
        sns.barplot(data=df, x="target", y=metric,
                    hue="Model", palette=palette, ax=ax)
        ax.set_title(metric); ax.set_xlabel("")
        if metric == "R2":
            ax.set_ylim(0, 1)
            ax.axhline(0.7, color="red", lw=0.8, ls="--",
                       label="R²=0.7 target")
        ax.legend(fontsize=7)
    plt.suptitle("RF vs NN comparison", fontsize=12)
    plt.tight_layout()
    out = RESULT_DIR / "metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


def plot_rf_importance(target="logP", top_n=30):
    path = RESULT_DIR / f"rf_importance_{target}.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path, index_col=0).head(top_n)
    imp.index = [f"bin {i}" for i in imp.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp.index[::-1], imp["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"Top {top_n} RF importances — {target}")
    plt.tight_layout()
    out = RESULT_DIR / f"rf_importance_{target}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Frag2Drug — Step 2: Train & Evaluate")
    parser.add_argument("--shap", action="store_true",
                        help="Run SHAP analysis after training (adds ~5 min)")
    args = parser.parse_args()

    mlflow.set_experiment("Frag2Drug")

    with mlflow.start_run(run_name="Frag2Drug_training"):

        # A — refilter + feature engineering (cached after first run)
        X, y, meta = refilter_and_featurise(force=False)

        # B — split + scale
        (X_train,   X_val,   X_test,
         X_train_s, X_val_s, X_test_s,
         y_train,   y_val,   y_test) = split_and_scale(X, y)

        mlflow.log_params({
            "n_train"      : X_train.shape[0],
            "n_val"        : X_val.shape[0],
            "n_test"       : X_test.shape[0],
            "n_features"   : X_train.shape[1],
            "random_state" : RANDOM_STATE,
        })

        # C — Random Forest (raw unscaled features)
        rf_results = train_random_forest(X_train, y_train, X_test, y_test)

        # D — Neural Network (scaled features)
        nn_results = train_neural_network(
            X_train_s, y_train,
            X_val_s,   y_val,
            X_test_s,  y_test,
        )

        # E — Plots
        plot_predictions(y_test, rf_results, nn_results)
        plot_nn_loss(nn_results)
        plot_metrics_bar(rf_results, nn_results)
        plot_rf_importance(target="logP",  top_n=30)
        plot_rf_importance(target="QED",   top_n=30)
        plot_rf_importance(target="MW",    top_n=30)

        # F — SHAP analysis (optional — run with --shap flag)
        # RF SHAP is available per-query in the Streamlit app (app.py).
        # NN SHAP uses GradientExplainer and runs in ~5 min on CPU.
        if args.shap and nn_results is not None:
            compute_shap_nn(
                nn_model  = nn_results["model"],
                X_train_s = X_train_s,
                X_test_s  = X_test_s,
                sample_n  = 200,
            )

        # Log all plots (including SHAP) as MLflow artifacts
        for png in RESULT_DIR.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="plots")

        # G — Final summary
        print("\n" + "="*52)
        print("  FINAL RESULTS SUMMARY")
        print("="*52)
        for label, res in [("Random Forest", rf_results),
                           ("Neural Network", nn_results)]:
            if res is None:
                continue
            print(f"\n  {label}:")
            for m in res["metrics"]:
                print(f"    {m['target']:<6}  R²={m['R2']:.4f}  "
                      f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

    print(f"\n[Step 2 complete]")
    print(f"  Models  → {MODEL_DIR}/")
    print(f"  Results → {RESULT_DIR}/")
    print(f"  MLflow  → run: mlflow ui")
