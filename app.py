"""
Frag2Drug — Streamlit App
Author : Dr. Amudha Kumari Duraisamy
Given an MS/MS spectrum, predict molecular properties and retrieve
drug-like candidates from the MassBank database.
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import io
from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR  = Path("results/data_processing")
MODEL_DIR = Path("results/model_training/models")

# ── Constants (must match step2) ──────────────────────────────
MZ_MIN, MZ_MAX = 50, 1000
N_BINS         = 1000
NN_HIDDEN      = [512, 256, 128]
TARGET_NAMES   = ["logP", "QED", "MW"]

# Drug-likeness thresholds
QED_MIN          = 0.5
LOGP_MIN, LOGP_MAX = 0.0, 5.0
MW_MIN,   MW_MAX   = 150.0, 500.0
TOP_K            = 10
MAX_SIM          = 0.999


# ══════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING  (cached so they load once)
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_database():
    X    = np.load(DATA_DIR / "X_clean.npy")
    y    = np.load(DATA_DIR / "y_clean.npy")
    meta = pd.read_csv(DATA_DIR / "meta_clean.csv")

    X_norm = normalize(X, norm="l2")

    # Deduplicate
    rounded    = np.round(X_norm, 4)
    hashes     = np.array([hash(r.tobytes()) for r in rounded])
    _, uid     = np.unique(hashes, return_index=True)
    uid        = np.sort(uid)
    X, X_norm, y = X[uid], X_norm[uid], y[uid]
    meta       = meta.iloc[uid].reset_index(drop=True)

    return X, X_norm, y, meta


@st.cache_resource
def load_models():
    rf = {t: joblib.load(MODEL_DIR / f"rf_{t}.pkl") for t in TARGET_NAMES}
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    y_scaler = joblib.load(MODEL_DIR / "y_scaler.pkl")

    try:
        import torch
        import torch.nn as nn

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        class PropNet(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                dims   = [n_features] + NN_HIDDEN
                layers = []
                for a, b in zip(dims, dims[1:]):
                    layers += [nn.Linear(a, b), nn.BatchNorm1d(b),
                                nn.ReLU(), nn.Dropout(0.3)]
                layers.append(nn.Linear(dims[-1], len(TARGET_NAMES)))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)

        n_feat = np.load(DATA_DIR / "X_clean.npy").shape[1]
        net    = PropNet(n_feat)
        net.load_state_dict(torch.load(MODEL_DIR / "nn_best.pt",
                                        map_location="cpu"))
        net.eval()
    except Exception:
        net = None

    return rf, scaler, y_scaler, net


# ══════════════════════════════════════════════════════════════
#  MSP PARSER
# ══════════════════════════════════════════════════════════════

def parse_msp(text: str) -> list[dict]:
    """Parse an MSP file string into a list of spectrum dicts."""
    spectra, current = [], {}
    peaks_mode = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                spectra.append(current)
                current = {}
                peaks_mode = False
            continue

        if peaks_mode:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                    current.setdefault("peaks", []).append((mz, intensity))
                except ValueError:
                    pass
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "num peaks":
                peaks_mode = True
            else:
                current[key] = val
        elif line.lower().startswith("num peaks"):
            peaks_mode = True

    if current:
        spectra.append(current)
    return spectra


def spectrum_to_vector(peaks: list[tuple], n_bins=N_BINS,
                        mz_min=MZ_MIN, mz_max=MZ_MAX) -> np.ndarray:
    """Bin peaks into a fixed-length intensity vector."""
    vec  = np.zeros(n_bins, dtype=np.float32)
    step = (mz_max - mz_min) / n_bins
    for mz, intensity in peaks:
        if mz_min <= mz < mz_max:
            idx = int((mz - mz_min) / step)
            vec[idx] += intensity
    if vec.max() > 0:
        vec /= vec.max()   # normalise to [0, 1]
    return vec


def add_summary_features(vec: np.ndarray, precursor_mz: float = 0.0) -> np.ndarray:
    """Append 6 summary features + precursor_mz (must match step2 feature order)."""
    mz_centres = np.linspace(MZ_MIN, MZ_MAX, N_BINS)
    tic        = vec.sum()
    n_peaks    = float((vec > 0).sum())
    mean_mz    = float((vec * mz_centres).sum()) if tic > 0 else 0.0
    max_bin    = float(vec.argmax())
    frac_low   = float(vec[mz_centres < 200].sum())
    frac_high  = float(vec[mz_centres > 600].sum())
    return np.concatenate([vec, [tic, n_peaks, mean_mz,
                                  max_bin, frac_low, frac_high,
                                  precursor_mz]]).astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════

def predict_properties(feat_vec: np.ndarray, rf, scaler, y_scaler, net):
    X = feat_vec.reshape(1, -1)
    X_s = scaler.transform(X)

    rf_pred = {t: float(rf[t].predict(X)[0]) for t in TARGET_NAMES}

    nn_pred = None
    if net is not None:
        try:
            import torch
            with torch.no_grad():
                t_in   = torch.from_numpy(np.ascontiguousarray(X_s)).float()
                t_out  = net(t_in).numpy()
                t_out  = y_scaler.inverse_transform(t_out)[0]
            nn_pred = {t: float(t_out[i]) for i, t in enumerate(TARGET_NAMES)}
        except Exception:
            nn_pred = None

    return rf_pred, nn_pred


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════

def retrieve(query_vec: np.ndarray, X_norm, meta, top_k=TOP_K):
    q    = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = X_norm.dot(q)
    ranked = np.argsort(sims)[::-1]

    results = []
    for idx in ranked:
        if sims[idx] >= MAX_SIM:
            continue
        row = meta.iloc[idx]
        if not (row["qed"] >= QED_MIN and
                LOGP_MIN <= row["logP"] <= LOGP_MAX and
                MW_MIN   <= row["mw"]   <= MW_MAX):
            continue
        results.append({
            "Rank"       : len(results) + 1,
            "Name"       : row["name"],
            "Similarity" : round(float(sims[idx]), 4),
            "logP"       : round(float(row["logP"]), 2),
            "QED"        : round(float(row["qed"]),  3),
            "MW"         : round(float(row["mw"]),   1),
            "SMILES"     : row["smiles"],
        })
        if len(results) >= top_k:
            break

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════

def get_feature_names(n_features=1007):
    """Human-readable names for the 1007-dim feature vector."""
    mz_centres = np.linspace(MZ_MIN, MZ_MAX, N_BINS)
    names = [f"m/z {c:.0f}" for c in mz_centres]
    names += ["TIC", "n_peaks", "mean_mz", "max_bin",
              "frac_low_<200", "frac_high_>600", "precursor_mz"]
    return names[:n_features]


@st.cache_resource
def get_shap_explainers(_rf):
    import shap
    return {t: shap.TreeExplainer(_rf[t]) for t in TARGET_NAMES}


def plot_shap(feat_vec, explainers, top_n=12):
    """Bar chart of top SHAP features for each target property."""
    feat_names = get_feature_names(len(feat_vec))
    X = feat_vec.reshape(1, -1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"logP": "#4C72B0", "QED": "#55a868", "MW": "#DD8452"}

    for ax, target in zip(axes, TARGET_NAMES):
        sv = explainers[target].shap_values(X)[0]   # shape (n_features,)
        abs_sv  = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_n]

        labels = [feat_names[i] for i in top_idx][::-1]
        values = sv[top_idx][::-1]
        bar_colors = ["#e05c5c" if v > 0 else "#5c8ae0" for v in values]

        ax.barh(labels, values, color=bar_colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"SHAP — {target}", fontsize=10)
        ax.set_xlabel("Impact on prediction")
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("Feature importance for this query  "
                 "(red = pushes value up · blue = pushes value down)",
                 fontsize=9)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════

def plot_spectrum(vec, title="Query Spectrum"):
    mz  = np.linspace(MZ_MIN, MZ_MAX, N_BINS)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.vlines(mz, 0, vec, colors="#4C72B0", linewidth=0.6)
    ax.set_xlabel("m/z"); ax.set_ylabel("Relative Intensity")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(MZ_MIN, MZ_MAX)
    plt.tight_layout()
    return fig


def plot_candidates(df):
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Similarity bar
    axes[0].barh(df["Name"].str[:25][::-1],
                 df["Similarity"][::-1], color="#4C72B0")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_title("Spectral Similarity")
    axes[0].set_xlim(0, 1)

    # logP vs QED bubble (size = MW)
    sc = axes[1].scatter(df["logP"], df["QED"],
                         s=df["MW"] / 3,
                         c=df["Similarity"], cmap="YlOrRd",
                         vmin=0, vmax=1,
                         edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=axes[1], label="Similarity")
    axes[1].axvline(LOGP_MIN, color="grey", lw=0.8, ls="--")
    axes[1].axvline(LOGP_MAX, color="grey", lw=0.8, ls="--")
    axes[1].axhline(QED_MIN,  color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("logP"); axes[1].set_ylabel("QED")
    axes[1].set_title("Property Space  (bubble ∝ MW)")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Frag2Drug", page_icon="💊", layout="wide")

st.title("💊 Frag2Drug")
st.markdown(
    "**MS/MS → Drug Candidate Retrieval & Property Prediction**  \n"
    "Upload an `.msp` spectrum file or pick a demo compound to find "
    "drug-like candidates from the MassBank database."
)


# ── Session state defaults ────────────────────────────────────
for key, default in [
    ("results_ready", False),
    ("spectrum_idx",  0),
    ("spectra",       []),
    ("input_mode",    "Upload .msp file"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load data & models (cached — happens once regardless of active tab)
with st.spinner("Loading database and models..."):
    X, X_norm, y, meta = load_database()
    rf, scaler, y_scaler, net = load_models()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Settings")
top_k  = st.sidebar.slider("Candidates to retrieve", 5, 20, TOP_K)
qed_th = st.sidebar.slider("Min QED", 0.0, 1.0, QED_MIN, 0.05)
lp_min = st.sidebar.slider("Min logP", -5.0, 5.0, LOGP_MIN, 0.5)
lp_max = st.sidebar.slider("Max logP", 0.0, 10.0, LOGP_MAX, 0.5)
mw_max = st.sidebar.slider("Max MW (Da)", 200, 800, int(MW_MAX), 50)

# ── Input ─────────────────────────────────────────────────────
st.subheader("Input Spectrum")
input_mode = st.radio("Source", ["Upload .msp file", "Pick demo compound"],
                       horizontal=True)

query_vec  = None
query_name = ""
query_raw  = None

if input_mode == "Upload .msp file":
    uploaded = st.file_uploader("Upload .msp file", type=["msp", "txt"])

    if uploaded:
        text    = uploaded.read().decode("utf-8", errors="ignore")
        spectra = parse_msp(text)

        if not spectra:
            st.error("Could not parse any spectra from this file.")
        else:
            # store parsed spectra in session state when file changes
            if spectra != st.session_state.spectra:
                st.session_state.spectra       = spectra
                st.session_state.spectrum_idx  = 0
                st.session_state.results_ready = False

            n       = len(st.session_state.spectra)
            idx     = st.session_state.spectrum_idx
            spec    = st.session_state.spectra[idx]
            peaks   = spec.get("peaks", [])
            name    = spec.get("name", f"Spectrum {idx+1}")

            st.caption(f"Spectrum {idx+1} of {n} — **{name}**")

            if peaks:
                try:
                    pmz = float(spec.get("precursor_mz", 0.0))
                except (ValueError, TypeError):
                    pmz = 0.0
                query_raw  = spectrum_to_vector(peaks)
                query_vec  = add_summary_features(query_raw, precursor_mz=pmz)
                query_name = name

                # ── Action buttons ────────────────────────────
                b1, b2, b3 = st.columns([1, 1, 5])
                with b1:
                    if st.button("▶ Run", type="primary", use_container_width=True):
                        st.session_state.results_ready = True
                with b2:
                    if st.button("✕ Clear", use_container_width=True):
                        st.session_state.results_ready = False
                if n > 1:
                    with b3:
                        bcols = st.columns([1, 1, 10])
                        with bcols[0]:
                            if st.button("◀ Prev", use_container_width=True,
                                         disabled=(idx == 0)):
                                st.session_state.spectrum_idx  = max(0, idx - 1)
                                st.session_state.results_ready = False
                                st.rerun()
                        with bcols[1]:
                            if st.button("Next ▶", use_container_width=True,
                                         disabled=(idx == n - 1)):
                                st.session_state.spectrum_idx  = min(n - 1, idx + 1)
                                st.session_state.results_ready = False
                                st.rerun()
            else:
                st.warning("Selected spectrum has no peak data.")

else:
    # Demo mode — pick from database
    dl_mask   = (
        (meta["qed"]  >= QED_MIN) &
        (meta["logP"] >= LOGP_MIN) & (meta["logP"] <= LOGP_MAX) &
        (meta["mw"]   >= MW_MIN)   & (meta["mw"]   <= MW_MAX)
    )
    demo_meta = meta[dl_mask].reset_index(drop=True)
    demo_X    = X[dl_mask.values]

    choice = st.selectbox("Select compound", demo_meta["name"].tolist())
    cidx   = demo_meta.index[demo_meta["name"] == choice][0]
    query_raw  = demo_X[cidx, :N_BINS]
    query_vec  = demo_X[cidx]
    query_name = choice
    st.info(
        f"logP = {demo_meta.loc[cidx,'logP']:.2f}  |  "
        f"QED = {demo_meta.loc[cidx,'qed']:.3f}  |  "
        f"MW = {demo_meta.loc[cidx,'mw']:.1f} Da"
    )

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("▶ Run", type="primary", use_container_width=True):
            st.session_state.results_ready = True
    with b2:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.results_ready = False

# ── Results ───────────────────────────────────────────────────
if st.session_state.results_ready and query_vec is not None:

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Query Spectrum")
        fig = plot_spectrum(query_raw, title=query_name)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("Predicted Properties")
        rf_pred, nn_pred = predict_properties(
            query_vec, rf, scaler, y_scaler, net
        )

        prop_data = {"Property": TARGET_NAMES,
                     "Random Forest": [f"{rf_pred[t]:.3f}" for t in TARGET_NAMES]}
        if nn_pred:
            prop_data["Neural Network"] = [f"{nn_pred[t]:.3f}" for t in TARGET_NAMES]

        # Best model per property: RF wins on MW (R²=0.951 vs 0.905),
        # NN wins on logP and QED.
        best = {
            "logP": nn_pred["logP"] if nn_pred else rf_pred["logP"],
            "QED" : nn_pred["QED"]  if nn_pred else rf_pred["QED"],
            "MW"  : rf_pred["MW"],   # RF is more accurate for MW
        }
        prop_data["Best estimate"] = [f"{best[t]:.3f}" for t in TARGET_NAMES]

        df_props = pd.DataFrame(prop_data).set_index("Property")
        st.dataframe(df_props, use_container_width=True)
        st.caption("Best estimate: NN for logP & QED · RF for MW  "
                   "(highest R² per property)")

        druglike = (
            best["QED"]  >= QED_MIN and
            LOGP_MIN <= best["logP"] <= LOGP_MAX and
            MW_MIN   <= best["MW"]   <= MW_MAX
        )
        if druglike:
            st.success("Best estimate: this query is **drug-like** ✓")
        else:
            st.warning("Best estimate: this query is **not drug-like**")

    # ── SHAP explanation ──────────────────────────────────────
    with st.expander("🔍 Why these predictions? (SHAP feature importance)",
                     expanded=False):
        with st.spinner("Computing SHAP values..."):
            try:
                explainers = get_shap_explainers(rf)
                fig_shap   = plot_shap(query_vec, explainers)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close()
                st.caption(
                    "Shows which spectral features drove each property prediction. "
                    "**Red bars** push the predicted value higher; "
                    "**blue bars** push it lower. "
                    "Features are from the Random Forest model (R²=0.95 for MW, 0.65 for logP, 0.78 for QED)."
                )
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

    # ── Retrieval ─────────────────────────────────────────────
    st.subheader(f"Top {top_k} Drug-like Candidates")

    q      = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims   = X_norm.dot(q)
    ranked = np.argsort(sims)[::-1]

    rows = []
    for idx in ranked:
        if sims[idx] >= MAX_SIM:
            continue
        row = meta.iloc[idx]
        if not (row["qed"] >= qed_th and
                lp_min <= row["logP"] <= lp_max and
                MW_MIN <= row["mw"]   <= mw_max):
            continue
        rows.append({
            "Rank"       : len(rows) + 1,
            "Name"       : row["name"],
            "Similarity" : round(float(sims[idx]), 4),
            "logP"       : round(float(row["logP"]), 2),
            "QED"        : round(float(row["qed"]),  3),
            "MW (Da)"    : round(float(row["mw"]),   1),
            "SMILES"     : row["smiles"],
        })
        if len(rows) >= top_k:
            break

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        st.warning("No drug-like candidates found. Try relaxing the filters in the sidebar.")
    else:
        st.dataframe(candidates.set_index("Rank"), use_container_width=True)

        fig2 = plot_candidates(candidates.rename(columns={"MW (Da)": "MW"}))
        if fig2:
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        csv = candidates.to_csv(index=False).encode()
        st.download_button("⬇ Download candidates CSV", csv,
                           file_name=f"candidates_{query_name[:30]}.csv",
                           mime="text/csv")

st.markdown("---")
st.caption(
    "Frag2Drug · Dr. Amudha Kumari Duraisamy · MassBank-NIST database · "
    "Random Forest + Neural Network property prediction · "
    "Cosine similarity retrieval"
)

# ══════════════════════════════════════════════════════════════
#  ABOUT SECTION
# ══════════════════════════════════════════════════════════════

with st.expander("📖 About & Documentation", expanded=False):
    st.markdown("""
## About Frag2Drug

Frag2Drug is an end-to-end pipeline that takes a tandem mass spectrometry
(MS/MS) spectrum as input and returns ranked **drug-like candidate molecules**
from the MassBank-NIST database, together with machine-learning predictions
for three key molecular properties.

---

### Why this matters

Interpreting MS/MS spectra — identifying *what molecule produced this
fragmentation pattern* — is a major bottleneck in drug discovery, environmental
monitoring, and metabolomics. Traditional tools (GNPS, MSFinder) focus purely
on compound identification. Frag2Drug goes further: it simultaneously
**retrieves spectrally similar known drugs** and **predicts physicochemical
properties** of the unknown query, enabling rapid drug-likeness assessment even
for compounds not present in any database.

---

### Pipeline

| Step | What happens |
|------|-------------|
| **Feature engineering** | 1000 binned m/z intensities + 6 spectral summary features + precursor m/z → 1007-dim vector |
| **Property prediction** | Random Forest and Neural Network jointly predict logP, QED, MW |
| **Retrieval** | Cosine similarity search over 61,931 deduplicated MassBank spectra |
| **Drug-likeness filter** | QED ≥ 0.5 · logP 0–5 · MW 150–500 Da (Lipinski-inspired) |

---

### Model performance

| Property | Random Forest R² | NN R² |
|----------|-----------------|-------|
| logP (lipophilicity) | 0.654 | 0.712 |
| QED (drug-likeness 0–1) | 0.779 | 0.788 |
| MW — molecular weight | **0.951** | 0.905 |

The Random Forest achieves R²=0.95 for MW by leveraging the precursor m/z
as a direct splitting feature. The Neural Network uses target normalisation
so logP, QED, and MW contribute equally to the joint loss.

---

### Dataset

**MassBank-NIST 2024.06** — 105,757 raw spectra, 62,727 retained after
quality filtering. Labels (logP, QED, MW) computed from SMILES using RDKit.
56% of compounds pass the drug-likeness filter.

---

### How to use the app

1. Upload an `.msp` file or select a demo compound above
2. Click **▶ Run** to predict properties and retrieve candidates
3. Use **Next ▶ / ◀ Prev** to step through multi-spectrum files
4. Adjust filters in the sidebar to widen or narrow results
5. Download the candidate table as CSV

A sample file with Aspirin, Ibuprofen, Caffeine, and Paracetamol is
provided at `data/example_query.msp`.

---

### Tech stack

`matchms` · `rdkit` · `scikit-learn` · `PyTorch` · `Streamlit` · `MassBank-NIST`
""")

