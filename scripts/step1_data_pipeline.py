"""
=============================================================
 Frag2Drug  |  Step 1 — Data Acquisition & Preprocessing
=============================================================
 Author  : Dr. Amudha Kumari Duraisamy
 Dataset : MassBank NIST  (~113 MB MSP format)
           High-quality annotated MS/MS spectra with SMILES
 Labels  : logP, QED, MW  (computed via RDKit from SMILES)
 Features: binned m/z intensity vector (1000-dim)
 Output  : results/data_processing/X.npy    – feature matrix  (N × 1000)
           results/data_processing/y.npy    – label matrix    (N × 3)
           results/data_processing/meta.csv – compound metadata
=============================================================

Requirements:
    conda install -c conda-forge rdkit
    pip install matchms requests tqdm pandas matplotlib numpy
"""

import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import logging
logging.getLogger("matchms").setLevel(logging.ERROR)
from matchms.importing import load_from_msp
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

# ── Config ─────────────────────────────────────────────────
IN_DIR = Path("data")
OUT_DIR      = Path("results/data_processing")
OUT_DIR.mkdir(exist_ok=True)

MSP_URL      = "https://github.com/MassBank/MassBank-data/releases/download/2024.06/MassBank_NIST.msp"
MSP_PATH     = IN_DIR / "MassBank_NIST.msp"

MZ_MIN       = 50       # ignore fragments below 50 Da
MZ_MAX       = 1000     # cap at 1000 Da
N_BINS       = 1000     # feature vector dimensionality
MIN_PEAKS    = 3        # drop spectra with fewer peaks
MAX_SPECTRA  = None     # set e.g. 20_000 for a quick test run

BIN_EDGES    = np.linspace(MZ_MIN, MZ_MAX, N_BINS + 1)


# ══════════════════════════════════════════════════════════
#  1.  DOWNLOAD  (skips if file already exists)
# ══════════════════════════════════════════════════════════

def download_massbank(url: str, dest: Path) -> None:
    """
    Download MassBank NIST MSP with a progress bar.
    Safe to re-run — skips download if file already present.
    """
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"[skip] {dest.name} already exists ({size_mb:.0f} MB)")
        return

    print(f"[download] {url}")
    print("  Expected ~113 MB ...")

    class _Progress:
        def __init__(self):
            self.bar = None

        def __call__(self, block_num, block_size, total_size):
            if self.bar is None:
                self.bar = tqdm(
                    total=total_size, unit="B",
                    unit_scale=True, desc=f"  {dest.name}"
                )
            self.bar.update(block_size)

        def close(self):
            if self.bar:
                self.bar.close()

    progress = _Progress()
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    progress.close()
    print(f"[done] {dest}  ({dest.stat().st_size / 1e6:.0f} MB)")


# ══════════════════════════════════════════════════════════
#  2.  LOAD SPECTRA via matchms
# ══════════════════════════════════════════════════════════

def load_spectra(msp_path: Path, max_records: Optional[int] = None):
    """
    Load MSP file using matchms. Returns a list of Spectrum objects.
    matchms handles MSP parsing, metadata normalisation, and peak
    loading out of the box.
    """
    print(f"\n[load] {msp_path.name}  (this may take ~30 s for large files) ...")
    spectra = []
    for spec in load_from_msp(str(msp_path)):
        spectra.append(spec)
        if max_records and len(spectra) >= max_records:
            break
    print(f"  Loaded {len(spectra):,} raw spectra")
    return spectra


# ══════════════════════════════════════════════════════════
#  3.  COMPUTE MOLECULAR PROPERTIES  (RDKit)
# ══════════════════════════════════════════════════════════

def compute_properties(smiles: str) -> Optional[dict]:
    """
    Compute drug-relevant descriptors from a SMILES string.

    Returns
    -------
    dict with keys: logP, qed, mw, hbd, hba, tpsa, rotb
    None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "logP": Descriptors.MolLogP(mol),
        "qed" : QED.qed(mol),
        "mw"  : Descriptors.MolWt(mol),
        "hbd" : rdMolDescriptors.CalcNumHBD(mol),
        "hba" : rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


# ══════════════════════════════════════════════════════════
#  4.  FEATURISE SPECTRUM  ->  binned intensity vector
# ══════════════════════════════════════════════════════════

def spectrum_to_vector(mz_array, intensity_array) -> np.ndarray:
    """
    Convert raw peaks into a fixed-length 1000-dim feature vector.

    Steps
    -----
    1. Keep only peaks in [MZ_MIN, MZ_MAX].
    2. Sum intensities landing in each equal-width bin.
    3. L2-normalise so spectra with different TIC are comparable.
    """
    vec = np.zeros(N_BINS, dtype=np.float32)

    for mz, ity in zip(mz_array, intensity_array):
        if MZ_MIN <= mz <= MZ_MAX:
            idx = int(np.searchsorted(BIN_EDGES, mz, side="right") - 1)
            idx = min(idx, N_BINS - 1)
            vec[idx] += ity

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ══════════════════════════════════════════════════════════
#  5.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def build_dataset(spectra: list):
    """
    For each spectrum:
      - extract SMILES from metadata
      - compute molecular properties via RDKit
      - featurise the peak list
      - collect valid records

    Saves X.npy, y.npy, meta.csv to OUT_DIR.
    """
    X_list, y_list, meta_rows = [], [], []

    skipped_no_smiles  = 0
    skipped_bad_smiles = 0
    skipped_few_peaks  = 0

    print("\n[pipeline] Processing spectra ...")

    for spec in tqdm(spectra, unit=" spectra"):

        # ── SMILES (MassBank stores it as 'smiles' key) ───
        smiles = (
            spec.metadata.get("smiles")
            or spec.metadata.get("inchi")
            or ""
        )
        smiles = str(smiles).strip()

        if not smiles or smiles.lower() in ("n/a", "nan", "none", ""):
            skipped_no_smiles += 1
            continue

        # ── Molecular properties ───────────────────────────
        props = compute_properties(smiles)
        if props is None:
            skipped_bad_smiles += 1
            continue

        # ── Peak quality filter ────────────────────────────
        if spec.peaks is None or len(spec.peaks.mz) < MIN_PEAKS:
            skipped_few_peaks += 1
            continue

        # ── Featurise ──────────────────────────────────────
        vec = spectrum_to_vector(spec.peaks.mz, spec.peaks.intensities)
        if vec.sum() == 0:
            skipped_few_peaks += 1
            continue

        # ── Collect ────────────────────────────────────────
        X_list.append(vec)
        y_list.append([props["logP"], props["qed"], props["mw"]])
        meta_rows.append({
            "name"        : spec.metadata.get("compound_name", ""),
            "smiles"      : smiles,
            "precursor_mz": spec.metadata.get("precursor_mz", np.nan),
            "n_peaks"     : len(spec.peaks.mz),
            **props,
        })

    # ── Assemble ───────────────────────────────────────────
    X    = np.array(X_list,  dtype=np.float32)
    y    = np.array(y_list,  dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    # ── Report ─────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  Total loaded              : {len(spectra):>8,}")
    print(f"  Kept                      : {len(X):>8,}")
    print(f"  Skipped (no SMILES)       : {skipped_no_smiles:>8,}")
    print(f"  Skipped (invalid SMILES)  : {skipped_bad_smiles:>8,}")
    print(f"  Skipped (too few peaks)   : {skipped_few_peaks:>8,}")
    print(f"{'─'*52}")
    print(f"\n  Feature matrix X : {X.shape}")
    print(f"  Label matrix   y : {y.shape}")

    print(f"\n  Label statistics:")
    for i, name in enumerate(["logP", "QED", "MW"]):
        col = y[:, i]
        print(f"    {name:<6}  mean={col.mean():7.3f}  "
              f"std={col.std():6.3f}  "
              f"min={col.min():7.3f}  "
              f"max={col.max():7.3f}")

    # ── Save ───────────────────────────────────────────────
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    meta.to_csv(OUT_DIR / "meta.csv", index=False)

    print(f"\n[saved]")
    print(f"  {OUT_DIR / 'X.npy'}    ({X.nbytes / 1e6:.1f} MB)")
    print(f"  {OUT_DIR / 'y.npy'}    ({y.nbytes / 1e6:.1f} MB)")
    print(f"  {OUT_DIR / 'meta.csv'}")

    return X, y, meta


# ══════════════════════════════════════════════════════════
#  6.  EDA — label distributions + example spectrum
# ══════════════════════════════════════════════════════════

def quick_eda(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> None:
    """
    Plot label distributions and a sample binned spectrum.
    Saves PNG files to data/.
    """
    # ── Label distributions ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    label_info = [
        ("logP", "Lipophilicity (logP)",   "#4C72B0"),
        ("qed",  "Drug-likeness (QED)",    "#55A868"),
        ("mw",   "Molecular Weight (Da)",  "#C44E52"),
    ]
    for ax, (col, title, color) in zip(axes, label_info):
        ax.hist(meta[col], bins=50, color=color,
                edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        mean_val = meta[col].mean()
        ax.axvline(mean_val, color="black", linewidth=1,
                   linestyle="--", label=f"mean={mean_val:.2f}")
        ax.legend(fontsize=8)

    plt.suptitle("Label distributions — MassBank NIST", fontsize=12, y=1.02)
    plt.tight_layout()
    out1 = OUT_DIR / "label_distributions.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"[plot] {out1}")
    plt.show()

    # ── Sample binned spectrum ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    mz_centres = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    ax.bar(mz_centres, X[0], width=0.9, color="#4C72B0", linewidth=0)
    ax.set_xlabel("m/z bin centre")
    ax.set_ylabel("Normalised intensity")
    compound = meta.iloc[0]["name"] or meta.iloc[0]["smiles"][:40]
    ax.set_title(f"Sample spectrum: {compound}", fontsize=10)
    out2 = OUT_DIR / "sample_spectrum.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[plot] {out2}")
    plt.show()


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Step 1a — download (skipped automatically if file exists)
    download_massbank(MSP_URL, MSP_PATH)

    # Step 1b — load spectra
    spectra = load_spectra(MSP_PATH, max_records=MAX_SPECTRA)

    # Step 1c — process + save
    X, y, meta = build_dataset(spectra)

    # Step 1d — EDA plots
    quick_eda(X, y, meta)

    print("\n[Step 1 complete] Ready for Step 2 — model training.")
