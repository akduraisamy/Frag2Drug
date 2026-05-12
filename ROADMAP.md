# Frag2Drug — Roadmap

This document outlines planned improvements and extensions to the Frag2Drug pipeline.
Contributions and suggestions are welcome via GitHub Issues.

---

## Implemented

- [x] **Random Forest + Neural Network property prediction** — logP, QED, MW predicted from MS/MS spectrum
- [x] **Cosine similarity retrieval** — real-time search over 61,931 deduplicated MassBank-NIST spectra
- [x] **Drug-likeness gating** — Lipinski-inspired filters (QED, logP, MW) applied at retrieval time
- [x] **Precursor m/z as a learned feature** — anchors MW prediction, achieving RF R²=0.95
- [x] **Target normalisation for NN** — prevents MW scale from dominating the joint loss
- [x] **SHAP feature importance (RF)** — per-query explanation of which m/z bins drove each prediction; interactive beeswarm plots in the Streamlit app
- [x] **SHAP feature importance (NN)** — `shap.GradientExplainer` on the Neural Network; run with `python scripts/step2_train_evaluate.py --shap`; beeswarm plots saved to `results/model_training/results/`
- [x] **MLflow experiment tracking** — parameters, metrics (R²/MAE/RMSE), per-epoch NN loss, and SHAP plots logged per run; view with `mlflow ui`
- [x] **Streamlit web app** — interactive spectrum upload, Run / Clear / Next navigation, CSV download

---

## Near-term

### Models

The current pipeline uses Random Forest and a fully-connected Neural Network. The four models below are the best-suited additions for this specific problem: sparse, high-dimensional spectral vectors (1007 features) with three continuous regression targets (logP, QED, MW).

- [ ] **LightGBM** *(priority 1 — best expected gain)*
  Gradient boosting with histogram-based splits handles sparse high-dimensional tabular data better
  than Random Forest and is typically 10–100× faster to train. Histogram binning ignores near-zero
  m/z bins automatically, so no extra preprocessing is needed. Expected: logP R² > 0.75, QED R² > 0.82.
  Implementation: `pip install lightgbm`, one `LGBMRegressor` per target (logP, QED, MW), drops into
  the existing step 2 training loop alongside RF with minimal changes.

- [ ] **XGBoost** *(priority 2 — strong baseline, widely recognised)*
  Very similar to LightGBM in accuracy for this data size, but more widely cited in cheminformatics
  literature — useful for benchmarking and for communicating results to a drug-discovery audience.
  Implementation: `pip install xgboost`, one `XGBRegressor` per target.

- [ ] **1D Convolutional Neural Network** *(priority 3 — most scientifically novel)*
  A binned MS/MS spectrum is a 1D signal: adjacent m/z bins correspond to fragment mass differences
  (neutral losses), which are chemically meaningful (e.g. loss of 18 = water, 44 = CO₂).
  A 1D CNN with sliding-window filters can learn these local fragment patterns directly from the raw
  vector, without the manual feature engineering (TIC, frac_low, etc.) that the current NN requires.
  Architecture: three Conv1d layers (kernel size 5–9) → GlobalAvgPool → dense head → three outputs.
  This is the model most likely to improve logP prediction and to generalise to novel compound classes.

- [ ] **Partial Least Squares (PLS)** *(priority 4 — chemometrics credibility baseline)*
  PLS is the standard regression method in analytical chemistry and spectroscopy. Including it
  establishes credibility with a mass-spec / chemometrics audience and provides a fast linear baseline
  to show that RF and NN add non-linear value beyond what a classical method can achieve.
  Implementation: `sklearn.cross_decomposition.PLSRegression`, ~3 lines per target.

**Models deliberately excluded and why:**

| Model | Reason not suitable |
|-------|---------------------|
| KNN | Curse of dimensionality — distance metrics break down in 1007-dim sparse space |
| SVR (RBF kernel) | Kernel matrix is O(n²) for n=62K; too slow to train |
| Gaussian Process | Same scaling issue as SVR; not practical above ~5K samples |
| Transformer | Overkill for 1007-dim vectors; requires far more data to outperform tree models |
| Ridge / Lasso | Assume linear relationships; logP and QED have strong non-linear fragment dependencies |

### Visualisation

- [ ] **2D molecular structure rendering**
  Display RDKit-generated structure images for retrieved candidates directly in the app.
  Each row in the candidate table would show the 2D structure next to the SMILES string.
  Implementation: `from rdkit.Chem.Draw import MolToImage` — a few lines per candidate,
  rendered as inline images in Streamlit using `st.image`.

- [ ] **Interactive spectrum overlay**
  Plot the query spectrum and the top retrieved candidate spectrum on the same axes
  to visually show why they are spectrally similar.

---

## Medium-term

### Data & database

- [ ] **Multi-database support**
  Allow users to swap MassBank-NIST for other MS/MS databases via a simple `config.yaml` file:
  ```yaml
  database:
    path: data/my_database.msp
    format: msp        # msp or mgf
    smiles_field: smiles
  ```
  Planned databases: GNPS (MGF format), MoNA, HMDB, LipidMaps.
  Main challenge: each database uses slightly different field names in their MSP/MGF files.

- [ ] **Higher-resolution binning**
  Current binning uses 1000 bins over m/z 50–1000 = 0.95 Da per bin.
  At 0.1 Da resolution (9,500 bins) many co-eluting fragment pairs would be resolved,
  improving retrieval discrimination between structurally similar compounds and
  likely fixing the 0.999 similarity plateau seen in the current retrieval results.

- [ ] **Incremental database updates**
  Add new spectra to the search index without reprocessing the full dataset.
  Useful as MassBank releases quarterly updates.

### Pipeline

- [ ] **Batch processing**
  Accept a multi-spectrum MSP file and return results for all spectra in one run,
  with a combined downloadable CSV and summary plots.

- [ ] **Retention time as an additional feature**
  For LC-MS/MS datasets that include retention time, adding it as a feature
  could improve logP prediction (logP correlates with reversed-phase retention).

---

## Longer-term

- [ ] **Generative model** — move from retrieval to generation: given a spectrum,
  generate novel SMILES strings with predicted drug-like properties using a VAE or
  graph neural network. This is the true "Frag2Drug" vision.

- [ ] **Streamlit Community Cloud deployment** — make the app publicly accessible
  without requiring local installation. Blocked currently by the 62K-spectrum
  database size; would need a lightweight indexed version of the database.

- [ ] **Benchmarking against GNPS / SIRIUS** — formal comparison of retrieval
  and property prediction accuracy against established tools on a held-out dataset.

- [ ] **Confidence intervals on predictions** — use RF prediction variance or
  MC-Dropout in the NN to report uncertainty alongside the point estimate.

---

*Last updated: May 2026 — Dr. Amudha Kumari Duraisamy*
