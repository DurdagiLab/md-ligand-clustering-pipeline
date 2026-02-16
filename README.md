# MD Ligand Cluster Pipeline

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Schr%C3%B6dinger-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An automated workflow for extracting, aligning, clustering, and ranking ligand conformational states from Molecular Dynamics (MD) trajectories.**

---

This script implements an end-to-end workflow for analyzing Molecular Dynamics (MD) trajectories produced in **Schrödinger/Desmond** environments. The pipeline is designed to identify metastable ligand states and rank them based on a composite score derived from occupancy, binding site contacts, and structural stability.

It automates the following steps:
1.  **Extraction:** Extracts frames from trajectory bundles.
2.  **Alignment:** Aligns frames to a reference using protein backbone atoms (Kabsch algorithm).
3.  **Featurization:** Represents ligand conformations via flattened heavy-atom coordinates.
4.  **Clustering:** Performs **k-medoids** clustering (with optional PCA dimensionality reduction).
5.  **Ranking:** Ranks clusters using a multi-parameter composite score.
6.  **Kinetics:** Analyzes transition matrices and dwell times.

---

## Key Method Choices

### 1. Alignment Strategy
A rigid-body alignment is performed via the **Kabsch algorithm** using a protein backbone ASL selection (default: `protein and backbone and not H`). This removes global translation/rotation so that clustering focuses purely on internal ligand conformational variability.

### 2. Ligand Representation
* **Features:** Ligand heavy-atom Cartesian coordinates flattened into a 1D vector of length $3 \times N_{atoms}$.
* **Atom Mapping:**
    * *Primary:* Uses atom names if unique.
    * *Fallback:* Uses ASL/Index order (assumes consistent topology).
* **Metric:** True RMSD (Root Mean Square Deviation).

### 3. Clustering Algorithm
* **Method:** **K-medoids** (PAM-like initialization).
* **Space:** Can cluster in full coordinate space or PCA-reduced space.
* **Optimization:** Automatically selects the optimal number of clusters ($k$) using **Silhouette Analysis**.
* **Refinement:** If clustering is performed in PCA space, medoids are recomputed in full space to ensure representative structures are physically valid.

### 4. Cluster Ranking (Composite Score)
Clusters are ranked using robust percentile scaling (10th–90th) of the following weighted metrics:

| Metric | Weight | Description |
| :--- | :--- | :--- |
| **Occupancy** | 20% | Population of the cluster relative to total frames. |
| **Site Contact** | 40% | Fraction of binding site residues in contact with the ligand. |
| **Stability** | 20% | Center-of-Mass (COM) standard deviation (lower is better). |
| **Tightness** | 15% | Pairwise RMSD within the cluster (lower is better). |
| **Interactions** | 5% | Proxy score for H-bonds, Pi-Pi, etc. |

---

## Prerequisites

This script depends on the **Schrödinger Python API**. It must be run within the Schrödinger environment using the `$SCHRODINGER/run` wrapper.

* **Schrödinger Suite**
* **Python 3** (Included in Schrödinger)
* Standard libraries: `numpy`, `argparse`, `csv`, `glob`.
* Optional: `matplotlib` (for PCA plots).

---

## Usage

Save the script as `md_ligand_cluster_pipeline.py`.

### Basic Command
```bash
$SCHRODINGER/run md_ligand_cluster_pipeline.py \
  --out_cms /path/to/desmond_job_FILENAME-out.cms \
  --trj_dir /path/to/desmond_job_FILENAME_trj \
  --out_prefix analysis_result
```

### Citation
If you use this tool in your research or publication, please cite it as follows:

İsaoğlu, M., & Durdağı, S. (2026). MD Ligand Clustering Tool (Version 1.0) [Source Code]. 
[https://github.com/DurdagiLab/md-ligand-clustering-pipeline](https://github.com/DurdagiLab/md-ligand-clustering-pipeline)
