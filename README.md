# Lens Depth for Anomaly Detection

**A collaborative research project exploring Lens Depth, Fermat distances, dimensionality reduction, and clustering for anomaly detection.**

Developed as part of a TER research project, this repository contains Python utilities and Jupyter notebooks investigating how the geometry of a dataset influences statistical depth and anomaly scores. Experiments include machine-sound feature representations associated with DCASE and standard datasets used to study dimensionality reduction.

## Research questions

- How does Lens Depth behave in the original 768-dimensional feature space compared with lower-dimensional representations?
- How does replacing Euclidean distances with Fermat distances affect depth-based anomaly detection?
- How well does UMAP preserve local neighbourhoods under different distance choices?
- Can cluster-specific depth scores better represent data with several groups?
- How do the Fermat exponent, embedding dimension, and clustering parameters affect the results?

## Methods

**Lens Depth (LD)** measures how centrally a point lies relative to a reference sample. The implementation counts pairs of reference points whose associated lens contains the query point. The notebooks explore low depth as an indication of an unusual observation.

**Fermat distances** are computed as shortest-path distances on a graph whose edge weights are Euclidean distances raised to an exponent, `alpha`. The implementation also provides extensions for scoring points outside the reference sample.

**Dimensionality reduction** uses UMAP, with PCA also appearing in exploratory visualisations. Experiments compare Euclidean and Fermat distance representations and measure neighbourhood preservation using trustworthiness.

**Clustering** experiments include KMeans and DBSCAN, with additional spectral-clustering exploration. Several notebooks compute depth within separate clusters and combine the resulting scores.

Evaluation includes threshold-dependent classification metrics, ROC-AUC, partial ROC-AUC in selected experiments, and trustworthiness for dimensionality reduction.

## Experimental workflow

```text
Load feature representations or a standard dataset
                     |
       Standardise data where configured
                     |
 Keep the original space or reduce its dimension
                     |
 Compute Euclidean and/or Fermat distances
                     |
 Calculate Lens Depth, optionally within clusters
                     |
 Analyse anomaly scores and neighbourhood preservation
                     |
         Plot and export experimental results
```

The notebooks explore different branches of this workflow; they are not a single automated pipeline.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `ld_tools.py` | Lens Depth, Fermat distances, and scoring of external points; NumPy and PyTorch implementations. |
| `LD_768_cactus.ipynb` | Depth-based anomaly detection on the original 768-dimensional machine-sound features. |
| `cactus.ipynb` | Comparisons of original features, UMAP embeddings, Fermat distances, and multiple clusters. |
| `auc_alphas.ipynb` | Investigation of anomaly-detection AUC as the Fermat exponent changes. |
| `test_multiclustering.ipynb` | Cluster-based depth experiments, KMeans statistics, and DBSCAN parameter exploration. |
| `tests_clustering_768.ipynb` | Clustering experiments in the original feature space. |
| `tests_LD.ipynb` | Exploratory depth, clustering, and threshold-based evaluation experiments. |
| `tests_trustworthiness.ipynb` | Neighbourhood-preservation comparisons using different distances and embeddings. |
| `oak.ipynb` | Exploration of trustworthiness as a function of the Fermat exponent. |
| `expes_modified.ipynb` | Exploratory feature visualisation, source/target comparisons, and classification experiments. |
| `spectros.ipynb` | Audio and spectrogram exploration, including recordings from two microphone channels. |
| `matrices/` | Saved parameter-study outputs and a plotting script. |
| `stats.csv` | Exported metrics from selected depth and dimensionality-reduction experiments. |
| `stats_kmeans.csv`, `stats_kmeans_auc.csv` | Summary statistics from KMeans experiments. |
| `papers/` | Research material collected during the project. |

Notebook filenames beginning with `test` describe research experiments, not an automated software test suite.

## Getting started

### Python environment

Create a virtual environment in the repository directory:

```bash
python -m venv .venv
```

Activate it using the command appropriate for your system:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

The following packages are inferred from the imports in the supplied source:

```bash
python -m pip install numpy scipy matplotlib pandas scikit-learn umap-learn torch jupyterlab
```

Some notebooks additionally import `tensorflow` for MNIST or `librosa` for audio processing. Install these only when needed, or remove unused imports for experiments that do not require them.

The archive does not include a pinned environment specification. These commands are a starting point, not a verified reconstruction of the original environment.

### Notebook imports

Several notebooks import the utilities as:

```python
from LD.utils.ld_tools import *
```

In this archive, `ld_tools.py` is located at the repository root. When running a notebook from that directory, adapt the import to:

```python
from ld_tools import *
```

Some experiments also reference `utils`, `plot_anm_score`, or `kmeansFermat`. Those project-specific modules are **not included in the supplied archive**. Obtain them from the original project environment, or adapt the affected cells before running those experiments.

### Data

The machine-sound notebooks expect precomputed `.pt` files containing a `hidden_states` entry. Their original paths are local to the authors' machines and must be updated, including the `BASEDIR` variable.

The expected directory organisation includes:

```text
data/
  <machine>/
    train/
      *.pt
    test/
      *.pt
```

Feature files and raw recordings are not included in this archive. File naming and metadata must remain compatible with the notebook's source/target and normal/anomalous label-loading logic. The repository does not provide a complete raw-audio feature-extraction pipeline.

For a smaller starting experiment, use a standard dataset in `oak.ipynb` or `tests_trustworthiness.ipynb`. The notebooks contain configurations for Iris, Digits, Swiss Roll, and MNIST. Select the relevant cells and begin with a small sample before running the larger distance calculations.

Launch the notebook interface with:

```bash
jupyter lab
```

## Results and interpretation

The CSV files and notebook outputs preserve exploratory results from the project. They do not establish a single validated performance claim across all datasets and settings.

When interpreting or reproducing results:

- Check score orientation: if label `1` denotes an anomaly and lower depth indicates an anomaly, ROC-AUC should use an anomaly-oriented score such as `-depth`. The notebooks contain both orientations, and `cactus.ipynb` explicitly flags an AUROC calculation for review.
- Treat the maximum accuracy obtained by sweeping thresholds on labelled test data as exploratory. Select a threshold on separate validation data for a held-out evaluation.
- Record the machine, source/target split, preprocessing, embedding dimension, exponent, clustering parameters, and random seed.
- Compare experiments only when their data splits and evaluation conventions match.

The code constructs dense pairwise distance matrices and contains expensive pairwise computations. Large reference samples can therefore require substantial memory and runtime. Functions with `device='gpu'` in their signature do not explicitly move tensors to CUDA in the supplied implementation; that argument alone does not guarantee GPU execution.

## Collaboration and attribution

This work originates from the collaborative [TER---Lens-Depth project](https://github.com/asulf2/TER---Lens-Depth). Credit belongs to the original contributors and collaborators; this repository should not be presented as solely authored by its publisher.

The original repository may require access permission. Consult its commit history for individual contributions. If this project is published from a ZIP export, the export does not preserve that history, so keep this acknowledgement and the upstream link.

## Licensing and research material

No repository-level `LICENSE` file is present in the supplied archive. This README does not grant a new license to the code or third-party material. Agree on a license with the rights holders before distributing the project under one.

Research papers in `papers/` retain their publishers' and authors' rights. For a public release, prefer bibliographic references or official links unless redistribution of the PDFs is permitted. External datasets are governed by their own terms.
