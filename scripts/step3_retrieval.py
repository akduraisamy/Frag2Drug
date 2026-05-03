"""
=============================================================
 Frag2Drug  |  Step 3 — Spectrum-Based Drug Candidate Retrieval
=============================================================
 Author  : Dr. Amudha Kumari Duraisamy
 Given a query MS/MS spectrum (binned vector), find the most
 similar compounds in the database using cosine similarity,
 then filter to drug-like candidates.

 Input  : results/data_processing/X_clean.npy   feature matrix
          results/data_processing/y_clean.npy   properties
          results/data_processing/meta_clean.csv metadata + SMILES

 Output : results/retrieval/candidates_<query>.csv  per query
          results/retrieval/retrieval_summary.png    plot

 Drug-likeness filters (Lipinski-inspired)
 ------------------------------------------
   QED  >= 0.5          (drug-likeness score)
   logP in [0, 5]       (oral absorption range)
   MW   in [150, 500]   (small-molecule range)
=============================================================
"""

import os
import warnings
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = Path("results/data_processing")
OUT_DIR    = Path("results/retrieval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Drug-likeness filter thresholds ─────────────────────────
QED_MIN  = 0.5
LOGP_MIN, LOGP_MAX = 0.0, 5.0
MW_MIN,   MW_MAX   = 150.0, 500.0

TOP_K = 10   # candidates to return per query


# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════

def load_database():
    print("[load] Reading database ...")
    X    = np.load(DATA_DIR / "X_clean.npy")
    y    = np.load(DATA_DIR / "y_clean.npy")
    meta = pd.read_csv(DATA_DIR / "meta_clean.csv")

    print(f"  Spectra : {X.shape[0]:,}  |  Features : {X.shape[1]}")

    # L2-normalise rows for fast cosine similarity via dot product
    X_norm = normalize(X, norm="l2")

    # Deduplicate — remove spectra with identical normalised vectors
    X, X_norm, y, meta = deduplicate(X, X_norm, y, meta)

    return X, X_norm, y, meta


def deduplicate(X, X_norm, y, meta):
    """
    Remove near-duplicate spectra (cosine similarity = 1.0) by hashing
    each normalised row rounded to 4 decimal places. Keeps one representative
    per group. This is the main cause of all-1.0 similarity in retrieval.
    """
    # Round to 4dp then hash each row — fast O(N) deduplication
    rounded   = np.round(X_norm, 4)
    row_hashes = np.array([hash(row.tobytes()) for row in rounded])
    _, unique_idx = np.unique(row_hashes, return_index=True)
    unique_idx = np.sort(unique_idx)

    n_before = len(X)
    X      = X[unique_idx]
    X_norm = X_norm[unique_idx]
    y      = y[unique_idx]
    meta   = meta.iloc[unique_idx].reset_index(drop=True)

    print(f"  Deduplicated : {n_before:,} → {len(X):,}  "
          f"(removed {n_before - len(X):,} duplicate spectra)")
    return X, X_norm, y, meta


# ══════════════════════════════════════════════════════════
#  DRUG-LIKENESS FILTER
# ══════════════════════════════════════════════════════════

def is_druglike(row) -> bool:
    return (
        row["qed"]  >= QED_MIN   and
        LOGP_MIN <= row["logP"] <= LOGP_MAX and
        MW_MIN   <= row["mw"]   <= MW_MAX
    )


# ══════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════

def retrieve_candidates(query_vec: np.ndarray,
                        X_norm: np.ndarray,
                        meta: pd.DataFrame,
                        y: np.ndarray,
                        top_k: int = TOP_K,
                        exclude_idx: int = None,
                        max_sim: float = 0.999) -> pd.DataFrame:
    """
    Cosine similarity search over the full database.
    Returns top_k drug-like hits sorted by similarity.
    max_sim: skip near-perfect matches (residual duplicates after dedup).
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = X_norm.dot(q)

    if exclude_idx is not None:
        sims[exclude_idx] = -1.0   # exclude the query itself

    # rank all by similarity
    ranked = np.argsort(sims)[::-1]

    results = []
    for idx in ranked:
        if sims[idx] >= max_sim:    # skip residual near-duplicates
            continue
        row = meta.iloc[idx]
        if not is_druglike(row):
            continue
        results.append({
            "rank"       : len(results) + 1,
            "similarity" : float(sims[idx]),
            "name"       : row["name"],
            "smiles"     : row["smiles"],
            "logP"       : round(float(row["logP"]), 3),
            "QED"        : round(float(row["qed"]),  3),
            "MW"         : round(float(row["mw"]),   1),
            "n_peaks"    : int(row["n_peaks"]),
            "db_idx"     : int(idx),
        })
        if len(results) >= top_k:
            break

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════
#  PLOT — single query result
# ══════════════════════════════════════════════════════════

def plot_retrieval(query_vec, query_name, candidates_df, out_path):
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Query spectrum ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    mz  = np.linspace(50, 1000, len(query_vec))
    ax0.vlines(mz, 0, query_vec, colors="#4C72B0", linewidth=0.6)
    ax0.set_xlabel("m/z"); ax0.set_ylabel("Intensity")
    ax0.set_title(f"Query spectrum\n{query_name}", fontsize=9)

    # ── Similarity bar chart ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    labels = [f"{r['rank']}. {r['name'][:20]}" for _, r in candidates_df.iterrows()]
    sims   = candidates_df["similarity"].values
    bars   = ax1.barh(labels[::-1], sims[::-1], color="#DD8452")
    ax1.set_xlabel("Cosine similarity")
    ax1.set_title("Top candidates by similarity", fontsize=9)
    ax1.set_xlim(0, 1)
    for bar, v in zip(bars, sims[::-1]):
        ax1.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=7)

    # ── Property scatter: logP vs QED, sized by MW ───────
    ax2 = fig.add_subplot(gs[1, 0])
    sc = ax2.scatter(
        candidates_df["logP"], candidates_df["QED"],
        s=candidates_df["MW"] / 3,
        c=candidates_df["similarity"],
        cmap="YlOrRd", vmin=0, vmax=1, edgecolors="k", linewidths=0.4
    )
    plt.colorbar(sc, ax=ax2, label="Similarity")
    ax2.axvline(LOGP_MIN, color="grey", lw=0.8, ls="--")
    ax2.axvline(LOGP_MAX, color="grey", lw=0.8, ls="--")
    ax2.axhline(QED_MIN,  color="grey", lw=0.8, ls="--")
    ax2.set_xlabel("logP"); ax2.set_ylabel("QED")
    ax2.set_title("Candidate properties\n(bubble size ∝ MW)", fontsize=9)

    # ── MW histogram ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(candidates_df["rank"], candidates_df["MW"], color="#4C72B0", width=0.6)
    ax3.set_xlabel("Rank"); ax3.set_ylabel("MW (Da)")
    ax3.set_title("Molecular weight of candidates", fontsize=9)
    ax3.set_xticks(candidates_df["rank"])

    fig.suptitle(f"Frag2Drug Retrieval — {query_name}", fontsize=11, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


# ══════════════════════════════════════════════════════════
#  BENCHMARK — run on random held-out spectra
# ══════════════════════════════════════════════════════════

def run_benchmark(X, X_norm, y, meta, n_queries=5, seed=42):
    """
    Pick n_queries random spectra from the database as queries,
    retrieve candidates (excluding the query itself), and report.
    """
    rng = np.random.default_rng(seed)
    # prefer drug-like queries for interesting results
    druglike_mask = (
        (meta["qed"]  >= QED_MIN) &
        (meta["logP"] >= LOGP_MIN) & (meta["logP"] <= LOGP_MAX) &
        (meta["mw"]   >= MW_MIN)   & (meta["mw"]   <= MW_MAX)
    )
    pool = np.where(druglike_mask)[0]
    query_indices = rng.choice(pool, size=min(n_queries, len(pool)), replace=False)

    print(f"\n[benchmark] {len(query_indices)} queries  "
          f"(drug-like filter: {druglike_mask.sum():,} / {len(meta):,} pass)")
    print(f"  Thresholds — QED≥{QED_MIN}  logP∈[{LOGP_MIN},{LOGP_MAX}]  "
          f"MW∈[{MW_MIN:.0f},{MW_MAX:.0f}]")

    all_results = []

    for qi, idx in enumerate(query_indices):
        query_vec  = X[idx]
        query_name = meta.iloc[idx]["name"]
        print(f"\n  Query {qi+1}: {query_name}  "
              f"(logP={meta.iloc[idx]['logP']:.2f}  "
              f"QED={meta.iloc[idx]['qed']:.3f}  "
              f"MW={meta.iloc[idx]['mw']:.1f})")

        candidates = retrieve_candidates(
            query_vec, X_norm, meta, y, exclude_idx=idx, max_sim=0.999
        )

        if candidates.empty:
            print("    No drug-like candidates found.")
            continue

        print(f"  {'Rank':<5} {'Name':<30} {'Sim':>6} {'logP':>6} "
              f"{'QED':>6} {'MW':>7}")
        print("  " + "-"*65)
        for _, r in candidates.iterrows():
            print(f"  {int(r['rank']):<5} {r['name'][:30]:<30} "
                  f"{r['similarity']:>6.3f} {r['logP']:>6.2f} "
                  f"{r['QED']:>6.3f} {r['MW']:>7.1f}")

        # save per-query CSV
        safe_name = "".join(c if c.isalnum() else "_" for c in query_name)[:40]
        csv_path  = OUT_DIR / f"candidates_{safe_name}.csv"
        candidates.to_csv(csv_path, index=False)

        # plot
        plot_path = OUT_DIR / f"retrieval_{safe_name}.png"
        plot_retrieval(query_vec, query_name, candidates, plot_path)

        candidates["query_name"] = query_name
        candidates["query_idx"]  = idx
        all_results.append(candidates)

    # ── Summary plot across all queries ──────────────────
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_df.to_csv(OUT_DIR / "all_candidates.csv", index=False)
        plot_summary(summary_df)
        print(f"\n[done] All results → {OUT_DIR}/")


def plot_summary(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(df["similarity"], bins=20, color="#4C72B0", edgecolor="k", lw=0.4)
    axes[0].set_xlabel("Cosine Similarity"); axes[0].set_ylabel("Count")
    axes[0].set_title("Similarity distribution")

    axes[1].scatter(df["logP"], df["QED"], s=df["MW"]/4,
                    alpha=0.5, color="#DD8452", edgecolors="k", linewidths=0.3)
    axes[1].axvline(LOGP_MIN, color="grey", lw=0.8, ls="--")
    axes[1].axvline(LOGP_MAX, color="grey", lw=0.8, ls="--")
    axes[1].axhline(QED_MIN,  color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("logP"); axes[1].set_ylabel("QED")
    axes[1].set_title("Property space of candidates\n(bubble ∝ MW)")

    axes[2].hist(df["MW"], bins=20, color="#55a868", edgecolor="k", lw=0.4)
    axes[2].set_xlabel("MW (Da)"); axes[2].set_ylabel("Count")
    axes[2].set_title("MW distribution")

    plt.suptitle("Frag2Drug — Retrieval Summary", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "retrieval_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    X, X_norm, y, meta = load_database()
    run_benchmark(X, X_norm, y, meta, n_queries=5, seed=42)
