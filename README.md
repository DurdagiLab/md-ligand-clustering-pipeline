# MD Ligand Cluster Pipeline

**Automated Conformational Clustering and Kinetic Analysis for Molecular Dynamics Trajectories**

**Developed by:** Mine Isaoglu, Ph.D.  
**Affiliation:** Computational Drug Design Center (HITMER), Bahçeşehir University  
**Supervisor:** Serdar Durdagi, Ph.D.  

---

## Abstract

This repository contains a high-throughput Python pipeline designed to analyze ligand behavior in Molecular Dynamics (MD) simulations produced by the Schrödinger/Desmond suite. 

The pipeline automates the extraction of trajectory frames, performs rigid-body alignment on protein backbones, and clusters ligand conformations using an unsupervised K-Medoids algorithm. Beyond simple clustering, the tool provides a **Composite Ranking Score** to identify the most biologically relevant binding poses based on occupancy, stability, and protein-ligand interaction profiles.

## Key Features

* **Automated Extraction:** Wraps `trj2mae.py` to extract and process frames from raw Desmond trajectories (`.cms` / `_trj`).
* **Structural Alignment:** Implements the **Kabsch Algorithm** to align all frames to a reference protein backbone, isolating ligand motion.
* **Robust Feature Extraction:** Uses flattened heavy-atom coordinates (True RMSD metric) with atom-name mapping to handle topology consistency.
* **Dimensionality Reduction:** Optional **PCA (SVD)** implementation to visualize the conformational landscape.
* **K-Medoids Clustering:** Robust clustering (PAM-like) with Silhouette Score optimization to automatically determine the optimal number of clusters ($k$).
* **Kinetic Analysis:** Generates transition matrices and dwell time statistics (run lengths) for each cluster.
* **Cluster Ranking:** Ranks clusters using a weighted score of:
    * Occupancy
    * Binding Site Contact Fraction
    * Center of Mass (COM) Stability
    * Cluster Tightness (Internal RMSD)
    * Interaction Strength Proxies (H-bond, Pi-Pi, etc.)

## Prerequisites

This script depends on the **Schrödinger Python API**. It must be run within the Schrödinger environment.

* **Software:** Schrödinger Suite (2018-4 or newer recommended).
* **Environment:** Access to the `$SCHRODINGER/run` command.

## Installation

Clone this repository and ensure the script is executable:

```bash
git clone [https://github.com/YourUsername/md_ligand_cluster_pipeline.git](https://github.com/YourUsername/md_ligand_cluster_pipeline.git)
cd md_ligand_cluster_pipeline
chmod +x md_ligand_cluster_pipeline.py
