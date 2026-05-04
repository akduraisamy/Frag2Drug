# Frag2Drug

**MS/MS fragmentation spectrum → Drug candidate retrieval and molecular property prediction**

Frag2Drug is an end-to-end pipeline that takes a tandem mass spectrometry (MS/MS) spectrum as input and returns ranked drug-like candidate molecules from the MassBank-NIST database, along with machine-learning predictions for three key molecular properties: lipophilicity (logP), drug-likeness (QED), and molecular weight (MW).

---

## Motivation and novelty

### The problem

Mass spectrometry is the workhorse analytical technique of drug discovery, metabolomics, and environmental science. Every day, laboratories generate thousands of MS/MS spectra from unknown compounds — metabolites, natural products, environmental contaminants, drug candidates. The central question is always the same: **what molecule produced this fragmentation pattern?**

Existing tools answer this poorly:

| Tool | What it does | What it misses |
|------|-------------|----------------|
| GNPS / MASST | Spectral matching against known compounds | Fails entirely for unknowns not in the database |
| MSFinder / SIRIUS | Structural annotation from fragments | No drug-likeness assessment, no ranked candidates |
| Manual interpretation | Expert-driven fragment analysis | Slow, not scalable, requires deep expertise |

The fundamental gap: **none of these tools tell you whether an unknown compound is worth pursuing as a drug lead.** A researcher still has to identify the molecule first, then separately assess its drug-likeness — a two-step process that breaks down completely for truly novel compounds.

### What Frag2Drug does differently

Frag2Drug treats the spectrum itself as the primary evidence and answers both questions simultaneously:

1. **Which known drugs look spectrally similar to this unknown?** — cosine similarity retrieval over 61,931 MassBank spectra, filtered to drug-like hits only
2. **What are the predicted physicochemical properties of this unknown?** — machine learning models predict logP, QED, and MW directly from the spectrum, even if the compound has never been seen before

This means a medicinal chemist or drug hunter can upload a spectrum of an unknown natural product or synthetic compound and immediately get a ranked shortlist of structurally analogous approved drugs alongside property predictions — without needing to identify the molecule first.

### Technical novelty

Most public MS/MS machine learning projects on GitHub (e.g. ms2deepscore, CANOPUS, DeepMASS) focus on a single task — either spectral similarity or structural annotation. Frag2Drug is distinct in three ways:

- **Spectrum-first property prediction** — logP, QED, and MW are predicted directly from the fragmentation pattern and precursor ion, without requiring the molecular structure to be known first. This is useful precisely when the compound is unknown.
- **Retrieval and prediction in one workflow** — the same feature vector drives both the cosine similarity search and the ML models. A researcher gets spectrally similar known drugs *and* property estimates for the unknown in a single step.
- **Drug-likeness as a hard filter on retrieval, not a post-hoc ranking** — candidates that do not meet physicochemical thresholds (QED, logP, MW) are excluded from results entirely, not just ranked lower. This keeps the output focused on actionable leads from the first result onwards.
- **Precursor m/z as an explicit ML feature** — including the parent ion mass alongside fragment peaks gives the model a direct anchor for molecular weight estimation, achieving RF R² = 0.95 and MAE = 11 Da on the test set.

---

## Pipeline

```
Raw .msp file  (MassBank-NIST, 105,757 spectra)
      │
      ▼  Step 1 — Data pipeline
  Quality filters:
    • Remove sparse spectra  < 10 non-zero m/z bins  (-39,459)
    • Remove logP outliers   outside [-5, 10]         ( -2,829)
    • Remove MW outliers     outside [100, 1000] Da   (   -742)
  62,727 spectra retained
  Labels computed from SMILES via RDKit: logP, QED, MW
      │
      ▼  Step 2 — Feature engineering & training
  Feature vector (1007-dim):
    [1000 binned m/z intensities]  +
    [TIC, n_peaks, mean_mz, max_bin, frac_low, frac_high]  +
    [precursor_mz]
  70 / 15 / 15 train / val / test split
  → Random Forest  (300 trees, one model per target)
  → Neural Network (512 → 256 → 128, BatchNorm + Dropout, joint prediction)
      │
      ▼  Step 3 — Retrieval
  L2-normalised cosine similarity over 61,931 deduplicated spectra
  Drug-likeness filter → Top-K ranked candidates with SMILES
      │
      ▼  Streamlit app
  Interactive spectrum upload → predict + retrieve in one click
```

---

## Results

### Property prediction

| Property | Random Forest R² | RF MAE | Neural Network R² | NN MAE |
|----------|-----------------|--------|-------------------|--------|
| logP     | 0.654           | 0.835  | 0.712             | 0.698  |
| QED      | 0.779           | 0.070  | 0.788             | 0.065  |
| MW (Da)  | **0.951**       | **11.0** | 0.905           | 20.8   |

The Neural Network outperforms the Random Forest on logP and QED. The Random Forest achieves a higher MW R² (0.951 vs 0.905), likely because tree-based models can split directly on the precursor m/z value as a threshold rule, making it a very strong predictor for molecular weight.

### Retrieval

- Database: 61,931 spectra after deduplication (796 exact duplicates removed)
- Drug-like pool: 34,974 / 61,931 compounds pass the Lipinski-inspired filter (56%)
- Retrieval is real-time — cosine dot product over L2-normalised vectors, ~0.1s per query

---

## Project structure

```
Frag2Drug/
├── app.py                          # Streamlit web app
├── requirements.txt
├── data/
│   └── example_query.msp           # Aspirin, Ibuprofen, Caffeine, Paracetamol
├── scripts/
│   ├── step1_data_pipeline.py      # MSP → X.npy, y.npy, meta.csv
│   ├── step2_train_evaluate.py     # Feature engineering + RF/NN training
│   └── step3_retrieval.py          # Retrieval benchmark (5 random queries)
└── results/
    ├── data_processing/            # Filtered arrays, label distribution plots
    └── model_training/
        ├── models/                 # Saved RF (.pkl), NN (.pt), scalers
        └── results/                # Predicted vs actual plots, metrics CSV
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/<your-username>/Frag2Drug.git
cd Frag2Drug
conda create -n frag2drug python=3.10 -y
conda activate frag2drug
```

### 2. Install dependencies

```bash
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
mkdir -p data
curl -L -o data/MassBank_NIST.msp \
  https://github.com/MassBank/MassBank-data/releases/download/2024.06/MassBank_NIST.msp
```

---

## Running the pipeline

```bash
# Step 1 — parse raw MSP, compute RDKit labels (~5 min)
python scripts/step1_data_pipeline.py

# Step 2 — filter, feature engineering, train RF + NN (~15 min)
python scripts/step2_train_evaluate.py

# Step 3 — retrieval benchmark
python scripts/step3_retrieval.py
```

Each step caches its output — re-runs skip the expensive computation automatically.

---

## Running the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload `data/example_query.msp` to get started immediately.

---

## Docker

Build and run the app in a self-contained container (no conda/pip setup required):

```bash
# Build
docker build -t frag2drug .

# Run — mount your trained models at runtime
docker run -p 8501:8501 \
  -v /path/to/results:/app/results \
  frag2drug
```

Replace `/path/to/results` with the absolute path to your local `results/` folder containing the trained models. Then open [http://localhost:8501](http://localhost:8501).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `matchms` | MSP file parsing |
| `rdkit` | SMILES → logP, QED, MW |
| `scikit-learn` | Random Forest, StandardScaler |
| `torch` | Neural Network |
| `streamlit` | Web application |
| `numpy / pandas` | Data handling |
| `matplotlib / seaborn` | Visualisation |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full list of planned features and extensions,
including XGBoost/LightGBM models, 2D structure rendering, higher-resolution binning,
multi-database support, and a longer-term generative model.

---

## Author

**Dr. Amudha Kumari Duraisamy**

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
