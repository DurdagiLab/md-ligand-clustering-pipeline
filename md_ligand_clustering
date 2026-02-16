#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MD Ligand Conformational Clustering Pipeline (Desmond Trajectories)
==================================================================

Author Information
------------------
Developer: Mine Isaoglu, Ph.D.
Principal Investigator: Serdar Durdagi, Ph.D.
Affiliation: Computational Drug Design Center (HITMER), Faculty of Pharmacy,
            Bahçeşehir University, Istanbul, Turkey.
Version: February 2026

Overview
--------
This script provides an end-to-end workflow to analyze ligand conformations from
MD trajectories produced in a Schrödinger/Desmond environment. It:

1) extracts trajectory frames,
2) aligns frames using protein backbone atoms (rigid-body),
3) builds ligand conformation features from heavy-atom Cartesian coordinates,
4) optionally performs PCA (NumPy SVD),
5) clusters conformations with k-medoids,
6) ranks clusters using occupancy plus simple contact/interaction proxy metrics,
7) generates a discrete-state timeseries summary (runs/dwell/transition matrix).

Key Method Choices
------------------
Alignment
- Rigid-body alignment via the Kabsch algorithm using a protein backbone ASL
  selection (default: protein backbone heavy atoms).
- Removes global translation/rotation so clustering reflects internal ligand
  variability.

Ligand feature representation
- Uses flattened heavy-atom coordinates: [x1,y1,z1, x2,y2,z2, ...].
- Atom mapping:
  a) Prefer unique ligand atom-name mapping (stable across frames).
  b) Otherwise fall back to ASL/index ordering (assumes consistent topology).

RMSD convention (“true RMSD”)
- For flattened vectors x,y (length = 3*N_atoms):
  RMSD = sqrt( sum((x-y)^2) / N_atoms )
  (i.e., not divided by 3*N_atoms).

Optional PCA
- PCA computed via NumPy SVD on mean-centered features (optionally scaled).
- Can cluster on full space or on PCA-reduced space.

Clustering
- k-medoids (PAM-like methods). Distance is Euclidean in the chosen space.
- k is selected by silhouette score on a subsample, with a small penalty for
  larger k to avoid over-fragmentation.
- Optionally recompute medoids in full space if clustering was done on PCA space
  (for physically interpretable representatives).

Binding-site definition
- Binding-site residues are defined once from a reference structure as residues
  whose minimum heavy-atom distance to the ligand is within a cutoff
  (default site_cutoff = 5.0 Å).

Cluster ranking (triage score)
- Features are robustly scaled (10th–90th percentile) to reduce outlier impact.
- Composite score is a weighted sum of:
  - occupancy (cluster population fraction),
  - binding-site contact fraction,
  - COM stability (lower is better; std of ligand–site COM distance within cluster),
  - cluster tightness (lower is better; mean pairwise distance estimate),
  - interaction proxy (HBonds/salt bridges/pi/halogen + hydrophobic contacts).
- Weights are configurable via command-line arguments.

Timeseries / kinetics summary
-----------------------------
Treats the cluster assignment per frame as a discrete-state sequence and reports:
- contiguous runs (dwell segments),
- transition count matrix and row-normalized transition probabilities,
- dwell statistics per cluster (mean/median/min/max), optionally in ns.

Outputs (prefix = --out_prefix)
-------------------------------
Core clustering:
- <prefix>_k_silhouette.csv
- <prefix>_cluster_assignments.csv
- <prefix>_cluster_ranking.csv
- <prefix>_binding_site_residues.txt
- <prefix>_cluster<i>_medoid_aligned.mae
- <prefix>_cluster<i>_medoid_original.mae

Timeseries / kinetics:
- <prefix>_cluster_runs.csv
- <prefix>_cluster_timeseries_summary.txt

Optional PCA:
- <prefix>_pca_coords.csv
- <prefix>_pca_summary.txt
- <prefix>_pca_clusters_plot.png

Example
-------
$SCHRODINGER/run python3 md_ligand_cluster_pipeline.py \
  --out_cms /path/to/desmond_FILENAME-out.cms \
  --trj_dir /path/to/desmond__FILENAME_trj \
  --out_prefix clustering_FILENAME \
  --stride_extract 10 \
  --pca_n 2 \
  --cluster_on_pca \
  --recompute_medoids_fullspace \
  --frame_dt_ns 0.1

"""

import argparse
import csv
import glob
import math
import os
import random
import subprocess
import numpy as np

# Optional plotting (safe for headless servers)
HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False
    plt = None

from schrodinger import structure
from schrodinger.structutils.analyze import evaluate_asl

# Optional interactions (pipeline still works if these imports fail)
HAS_INTERACTIONS = True
try:
    from schrodinger.structutils.interactions.hbond import get_hydrogen_bonds
    from schrodinger.structutils.interactions.saltbridge import get_salt_bridges
    from schrodinger.structutils.interactions.pi import find_pi_pi_interactions, find_pi_cation_interactions
    try:
        from schrodinger.structutils.interactions.halogen import get_halogen_bonds
    except Exception:
        get_halogen_bonds = None
except Exception:
    HAS_INTERACTIONS = False
    get_hydrogen_bonds = None
    get_salt_bridges = None
    find_pi_pi_interactions = None
    find_pi_cation_interactions = None
    get_halogen_bonds = None


# Atomic masses for simple COM computations (fallback mass = 12.0)
_ATOMIC_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "Na": 22.99, "K": 39.098, "Ca": 40.078, "Mg": 24.305, "Zn": 65.38,
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(d):
    """Create directory d if it does not exist."""
    if not os.path.isdir(d):
        os.makedirs(d)


def sorted_maestro_files(folder):
    """Return sorted list of Maestro files (*.mae, *.maegz) in folder."""
    files = []
    files += glob.glob(os.path.join(folder, "*.mae"))
    files += glob.glob(os.path.join(folder, "*.maegz"))
    files.sort()
    return files


def find_schrodinger_run():
    """
    Find Schrödinger 'run' wrapper executable by checking SCHRODINGER18 or SCHRODINGER env vars.
    """
    root = os.environ.get("SCHRODINGER18") or os.environ.get("SCHRODINGER")
    if root:
        run_exe = os.path.join(root, "run")
        if os.path.isfile(run_exe):
            return run_exe
    return None


def atom_mass(ele):
    """Return atomic mass for element symbol, with a reasonable fallback."""
    return _ATOMIC_MASS.get(ele, 12.0)


def heavy_aids(st, aids):
    """Filter atom indices (aids) to heavy atoms only (exclude hydrogens)."""
    return [i for i in aids if st.atom[i].element != "H"]


def coords_from_aids(st, aids):
    """Return coordinates array of shape (len(aids), 3) for atom indices in aids."""
    return np.array([st.atom[i].xyz for i in aids], dtype=float)


def center_of_mass(st, aids):
    """
    Compute center of mass for atom indices aids (simple mass-weighted average).
    Returns NaNs if aids is empty.
    """
    if not aids:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    xyz = coords_from_aids(st, aids)
    masses = np.array([atom_mass(st.atom[i].element) for i in aids], dtype=float)
    w = masses / masses.sum()
    return (xyz * w[:, None]).sum(axis=0)


# =============================================================================
# Alignment: Kabsch algorithm
# =============================================================================

def kabsch_align(P, Q):
    """
    Compute best-fit rotation R and translation t that maps P -> Q
    in a least-squares sense (Kabsch algorithm).

    P, Q: arrays of shape (N, 3) with corresponding points
    Returns:
      R: (3,3) rotation matrix
      t: (3,) translation vector such that P*R + t ~ Q
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    C = np.dot(P0.T, Q0)
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    R = np.dot(np.dot(V, D), Wt)
    t = Qc - np.dot(Pc, R)
    return R, t


def apply_transform_to_structure(st, R, t):
    """
    Apply affine transform to all atoms of structure st:
      xyz' = xyz * R + t
    Uses bulk getXYZ/setXYZ if available; otherwise per-atom fallback.
    """
    try:
        xyz = st.getXYZ()
        xyz2 = np.dot(xyz, R) + t
        st.setXYZ(xyz2)
        return
    except Exception:
        nat = st.atom_total
        for i in range(1, nat + 1):
            x, y, z = st.atom[i].xyz
            v = np.dot(np.array([x, y, z]), R) + t
            st.atom[i].xyz = (float(v[0]), float(v[1]), float(v[2]))


def rmsd_flat(x, y):
    """
    TRUE RMSD for flattened coordinate vectors x,y of length 3*N:
      RMSD = sqrt( sum((x-y)^2) / N_atoms )
    """
    d = x - y
    n_atoms = max(1.0, (len(x) / 3.0))
    return float(np.sqrt(np.sum(d * d) / n_atoms))


# =============================================================================
# PCA via NumPy SVD
# =============================================================================

def pca_reduce_numpy(X, n_components=2, scale=False):
    """
    PCA using SVD on mean-centered feature matrix X.

    Parameters
    ----------
    X : (N, D) array
      Feature matrix (rows = frames, cols = features).
    n_components : int
      Number of principal components to retain.
    scale : bool
      If True, divide each feature by its standard deviation after centering.

    Returns
    -------
    X_red : (N, n_components)
      PCA-projected coordinates.
    components : (n_components, D)
      Principal axes (loadings).
    evr : (n_components,)
      Explained variance ratio for retained components.
    mu : (D,)
      Feature mean used for centering.
    std : (D,) or None
      Feature std used for scaling (None if scale=False).
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    Xc = X - mu

    std = None
    if scale:
        std = Xc.std(axis=0)
        std[std < 1e-12] = 1.0
        Xc = Xc / std

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components, :]
    X_red = np.dot(Xc, components.T)

    N = float(X.shape[0])
    if N > 1:
        ev = (S * S) / (N - 1.0)
        evr_full = ev / ev.sum() if ev.sum() > 1e-30 else ev * 0.0
    else:
        evr_full = np.zeros_like(S)

    evr = evr_full[:n_components]
    return X_red, components, evr, mu, std


def write_pca_outputs(prefix, aligned_used, orig_map, X_red, evr):
    """
    Write PCA coordinates and summary for reporting.
    """
    pca_csv = "{}_pca_coords.csv".format(prefix)
    pca_txt = "{}_pca_summary.txt".format(prefix)

    with open(pca_txt, "w") as f:
        f.write("PCA explained variance ratio:\n")
        for i, v in enumerate(evr, start=1):
            f.write("  PC{}: {:.6f}\n".format(i, float(v)))
        f.write("  Sum: {:.6f}\n".format(float(np.sum(evr))))

    with open(pca_csv, "w") as f:
        w = csv.writer(f)
        header = ["i", "aligned_file", "original_file"]
        for j in range(X_red.shape[1]):
            header.append("PC{}".format(j + 1))
        w.writerow(header)
        for i in range(X_red.shape[0]):
            aln = aligned_used[i]
            base = os.path.basename(aln)
            org = orig_map.get(base, "")
            row = [i, aln, org] + ["{:.6f}".format(float(v)) for v in X_red[i, :]]
            w.writerow(row)

    return pca_csv, pca_txt


def plot_pca_clusters_png(prefix, X_red_2d, labels, medoids=None, dpi=300,
                          title="Ligand Conformational Landscape (PCA)"):
    """
    Generate a publication-style 2D PCA scatter plot colored by cluster labels.

    Notes
    -----
    - Intended as an exploratory visualization.
    - If medoids are provided, they are marked with stars.
    """
    if X_red_2d is None:
        return ""
    X_red_2d = np.asarray(X_red_2d, dtype=float)
    if X_red_2d.ndim != 2 or X_red_2d.shape[1] < 2:
        return ""

    if not HAS_MPL:
        print("WARNING: matplotlib not available; PCA PNG plot skipped.")
        return ""

    labs = np.asarray(labels, dtype=int)
    uniq = np.unique(labs)
    k = len(uniq)

    out_png = "{}_pca_clusters_plot.png".format(prefix)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    cmap_name = "tab10" if k <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)

    for c in uniq:
        idx = np.where(labs == c)[0]
        ax.scatter(
            X_red_2d[idx, 0], X_red_2d[idx, 1],
            s=60, alpha=0.8,
            edgecolors="k", linewidths=0.3,
            c=[cmap(int(c) % cmap.N)],
            label="{}".format(int(c))
        )

    if medoids is not None:
        try:
            for c, mi in enumerate(medoids):
                mi = int(mi)
                if 0 <= mi < X_red_2d.shape[0]:
                    ax.scatter(
                        [X_red_2d[mi, 0]], [X_red_2d[mi, 1]],
                        s=220, marker="*",
                        edgecolors="k", linewidths=0.8,
                        c=[cmap(int(c) % cmap.N)]
                    )
        except Exception:
            pass

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi))
    plt.close(fig)
    return out_png


# =============================================================================
# Binding-site residue utilities
# =============================================================================

def residue_key(atom):
    """Return a stable residue identifier: (chain, resnum, inscode)."""
    ch = getattr(atom, "chain", "") or ""
    rn = int(atom.resnum)
    ins = getattr(atom, "inscode", "") or ""
    return (ch, rn, ins)


def residue_aids_by_key(st, key):
    """Return atom indices for residue identified by (chain,resnum,inscode)."""
    ch, rn, ins = key
    aids = []
    for a in st.atom:
        if a is None:
            continue
        if int(a.resnum) != rn:
            continue
        if (getattr(a, "chain", "") or "") != ch:
            continue
        if (getattr(a, "inscode", "") or "") != ins:
            continue
        aids.append(a.index)
    return aids


def protein_residue_keys(st, prot_aids):
    """Return unique residue keys present in a protein atom selection."""
    keys = []
    seen = set()
    for i in prot_aids:
        a = st.atom[i]
        k = residue_key(a)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def min_heavy_distance(st, aids1, aids2):
    """
    Minimum heavy-atom distance between two atom sets.
    Returns inf if either set has no heavy atoms.
    """
    a1 = heavy_aids(st, aids1) if aids1 else []
    a2 = heavy_aids(st, aids2) if aids2 else []
    if (not a1) or (not a2):
        return float("inf")
    X = coords_from_aids(st, a1)
    Y = coords_from_aids(st, a2)
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def binding_site_from_ligand(ref_st, lig_aids_h, prot_aids, site_cutoff):
    """
    Define binding-site residues as those with minimum heavy-atom distance to ligand
    below site_cutoff (Å) in a reference structure.
    """
    prot_keys = protein_residue_keys(ref_st, prot_aids)
    X = coords_from_aids(ref_st, lig_aids_h)

    site_keys = []
    for k in prot_keys:
        r_aids = heavy_aids(ref_st, residue_aids_by_key(ref_st, k))
        if not r_aids:
            continue
        Y = coords_from_aids(ref_st, r_aids)
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
        if float(np.sqrt(d2.min())) <= site_cutoff:
            site_keys.append(k)
    return site_keys


# =============================================================================
# Interaction / contact proxies (optional)
# =============================================================================

def hydrophobic_contact_fraction(st, prot_aids, lig_aids, cutoff=4.0):
    """
    Simple hydrophobic contact proxy:
      fraction of ligand hydrophobic atoms within cutoff (Å) of protein hydrophobic atoms.
    """
    lig_h = [i for i in heavy_aids(st, lig_aids) if st.atom[i].element in ("C", "S", "Cl", "Br", "I", "F")]
    prot_h = [i for i in heavy_aids(st, prot_aids) if st.atom[i].element in ("C", "S")]
    if not lig_h or not prot_h:
        return 0.0
    X = coords_from_aids(st, lig_h)
    Y = coords_from_aids(st, prot_h)
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    mind = np.sqrt(d2.min(axis=1))
    return float((mind < cutoff).mean())


def interaction_counts_single_ct(st, prot_aids, lig_aids):
    """
    Count selected interaction types (if Schrödinger interaction modules are available).
    This is used as a *proxy* for interaction richness rather than a strict energetic metric.
    """
    if not HAS_INTERACTIONS:
        return {"HBond": 0, "SaltBridge": 0, "PiPi": 0, "PiCat": 0, "Halogen": 0}

    prot_h = heavy_aids(st, prot_aids)
    lig_h = heavy_aids(st, lig_aids)
    counts = {"HBond": 0, "SaltBridge": 0, "PiPi": 0, "PiCat": 0, "Halogen": 0}

    try:
        hb = list(get_hydrogen_bonds(st, atoms1=lig_h, atoms2=prot_h, max_dist=3.5, honor_pbc=False))
        counts["HBond"] = len(hb)
    except Exception:
        pass
    try:
        sb = list(get_salt_bridges(st, group1=lig_h, group2=prot_h, cutoff=4.0))
        counts["SaltBridge"] = len(sb)
    except Exception:
        pass
    try:
        pp = list(find_pi_pi_interactions(st, struct2=st))
        counts["PiPi"] = len(pp)
    except Exception:
        pass
    try:
        pc = list(find_pi_cation_interactions(st, struct2=st))
        counts["PiCat"] = len(pc)
    except Exception:
        pass
    if get_halogen_bonds is not None:
        try:
            hb2 = list(get_halogen_bonds(st, atoms1=lig_h, atoms2=prot_h))
            counts["Halogen"] = len(hb2)
        except Exception:
            pass

    return counts


# =============================================================================
# Robust scaling (ranking normalization)
# =============================================================================

def robust_scale(values, higher_is_better=True):
    """
    Robust percentile scaling for ranking features:
      - Map 10th percentile -> 0, 90th percentile -> 1, clip outside.
      - Optionally invert if lower values are better.

    This mitigates outlier sensitivity compared to min-max scaling.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return []
    good = np.isfinite(arr)
    if not good.any():
        return [0.0 for _ in values]

    a = arr.copy()
    a[~good] = np.nanmedian(a[good])
    lo = np.percentile(a, 10)
    hi = np.percentile(a, 90)

    if abs(hi - lo) < 1e-12:
        scaled = np.zeros_like(a)
    else:
        scaled = (a - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)

    if not higher_is_better:
        scaled = 1.0 - scaled
    return list(scaled)


# =============================================================================
# (1) Frame extraction via trj2mae.py
# =============================================================================

def run_trj2mae_extract(out_cms, trj_dir, out_folder, basename, extract_asl, stride_extract):
    """
    Extract frames from a Desmond trajectory using trj2mae.py and write separate Maestro files.

    Parameters
    ----------
    out_cms : str
      The CMS file (system definition).
    trj_dir : str
      Desmond trajectory directory.
    out_folder : str
      Output folder for extracted frames.
    basename : str
      Base filename for extracted frames.
    extract_asl : str
      ASL defining which atoms to include in extracted frames.
    stride_extract : int
      Frame stride for extraction.

    Raises
    ------
    RuntimeError if extraction fails.
    """
    ensure_dir(out_folder)

    out_cms_abs = os.path.abspath(out_cms)
    trj_dir_abs = os.path.abspath(trj_dir)
    if not os.path.isfile(out_cms_abs):
        raise RuntimeError("CMS file not found: {}".format(out_cms_abs))
    if not os.path.exists(trj_dir_abs):
        raise RuntimeError("Trajectory path not found: {}".format(trj_dir_abs))

    run_exe = find_schrodinger_run()
    if run_exe:
        prefix = [run_exe, "trj2mae.py"]
    else:
        prefix = ["trj2mae.py"]

    cmd = prefix + [out_cms_abs, trj_dir_abs, basename,
                    "-out-format", "MAE",
                    "-extract-asl", extract_asl,
                    "-separate"]

    if stride_extract and int(stride_extract) > 1:
        cmd += ["-s", "0:-1:{}".format(int(stride_extract))]
    else:
        cmd += ["-s", "0:-1:1"]

    p = subprocess.run(
        cmd,
        cwd=out_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    if p.returncode != 0:
        raise RuntimeError("trj2mae failed.\nCMD: {}\n\nOUTPUT:\n{}".format(" ".join(cmd), p.stdout))
    return p.stdout


# =============================================================================
# (2) Alignment
# =============================================================================

def align_frames(input_folder, aligned_folder, fit_asl, ref_index=0):
    """
    Align all extracted frames to a reference frame using Kabsch alignment.

    - Reference frame is chosen by ref_index in the sorted frame list.
    - Alignment atoms are selected by fit_asl (default: protein backbone heavy atoms).

    Returns
    -------
    extracted_files : list
    aligned_paths : list
    ref_path : str
    """
    ensure_dir(aligned_folder)
    files = sorted_maestro_files(input_folder)
    if not files:
        raise RuntimeError("No .mae/.maegz files found in {}".format(input_folder))

    ref_path = files[max(0, min(ref_index, len(files) - 1))]
    ref_st = next(structure.StructureReader(ref_path))
    ref_fit = heavy_aids(ref_st, evaluate_asl(ref_st, fit_asl))
    if not ref_fit:
        raise RuntimeError("Reference fit_asl is empty: {}".format(fit_asl))
    ref_fit_xyz = coords_from_aids(ref_st, ref_fit)

    aligned_paths = []
    for fp in files:
        st = next(structure.StructureReader(fp))
        fit = heavy_aids(st, evaluate_asl(st, fit_asl))
        if not fit or len(fit) != len(ref_fit):
            continue
        fit_xyz = coords_from_aids(st, fit)
        R, t = kabsch_align(fit_xyz, ref_fit_xyz)
        apply_transform_to_structure(st, R, t)

        base = os.path.basename(fp)
        if base.endswith(".maegz"):
            base = base[:-6] + ".mae"
        outp = os.path.join(aligned_folder, base)

        w = structure.StructureWriter(outp)
        w.append(st)
        w.close()
        aligned_paths.append(outp)

    if len(aligned_paths) < 10:
        raise RuntimeError("Too few aligned frames written: {}".format(len(aligned_paths)))

    return files, aligned_paths, ref_path


# =============================================================================
# Ligand feature building with robust atom mapping
# =============================================================================

def ligand_atom_names_from_ref(st, lig_asl):
    """
    Determine whether ligand heavy atom names are unique in a reference structure.
    If unique, return a sorted list of names to enforce consistent mapping.

    If not unique (or missing), return None to indicate fallback to ASL/index order.
    """
    lig = heavy_aids(st, evaluate_asl(st, lig_asl))
    if not lig:
        return None

    name_to_aid = {}
    for i in lig:
        nm = (st.atom[i].pdbname or st.atom[i].name or "").strip()
        if not nm:
            continue
        if nm not in name_to_aid:
            name_to_aid[nm] = i

    if len(name_to_aid) != len(lig):
        print("WARNING: Ligand atom names are not unique or some names are empty.")
        print("  Ligand heavy atom count : {}".format(len(lig)))
        print("  Unique name count       : {}".format(len(name_to_aid)))
        print("  ==> Name-based mapping DISABLED. Falling back to ASL/index ordering.")
        print("  ==> Assumption: ligand atom ordering/topology is consistent across frames.")
        return None

    return sorted(name_to_aid.keys())


def ligand_aids_by_names(st, lig_asl, names):
    """
    Map ligand heavy atoms by a reference name list.
    Returns None if any name is missing in this frame.
    """
    lig = heavy_aids(st, evaluate_asl(st, lig_asl))
    if not lig:
        return None
    name_to_aid = {}
    for i in lig:
        nm = (st.atom[i].pdbname or st.atom[i].name or "").strip()
        if nm and nm not in name_to_aid:
            name_to_aid[nm] = i
    aids = []
    for nm in names:
        if nm not in name_to_aid:
            return None
        aids.append(name_to_aid[nm])
    return aids


def build_feature_matrix(aligned_files, lig_asl, names_ref):
    """
    Build feature matrix X from aligned frames:
      - Each row = flattened ligand heavy-atom coordinates (3*N).
      - Uses name-based mapping if names_ref is provided; otherwise ASL order.

    Returns
    -------
    X : (N, 3*N_atoms) array
    keep_files : list of aligned files used
    """
    X = []
    keep_files = []
    kept_natoms = None

    for fp in aligned_files:
        st = next(structure.StructureReader(fp))

        if names_ref:
            aids = ligand_aids_by_names(st, lig_asl, names_ref)
            if aids is None or len(aids) == 0:
                continue
        else:
            aids = heavy_aids(st, evaluate_asl(st, lig_asl))
            if not aids:
                continue

        if kept_natoms is None:
            kept_natoms = len(aids)
        else:
            if len(aids) != kept_natoms:
                continue

        xyz = coords_from_aids(st, aids)
        X.append(xyz.reshape(-1))
        keep_files.append(fp)

    X = np.array(X, dtype=float)
    return X, keep_files


# =============================================================================
# k-medoids clustering
# =============================================================================

def pairwise_dist_matrix(X):
    """
    Compute pairwise Euclidean distances between rows of X, with normalization by dimension:
      d(i,j) = sqrt( ||xi - xj||^2 / D )
    This yields an RMS-like metric consistent with the flattened-coordinate RMSD convention.
    """
    n, D = X.shape
    G = np.dot(X, X.T)
    x2 = np.sum(X * X, axis=1)
    d2 = (x2[:, None] + x2[None, :] - 2.0 * G) / float(D)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def init_medoids_pp(Dmat, k, seed=0):
    """
    k-medoids++-like initialization using squared distance weighting.
    """
    rnd = random.Random(seed)
    n = Dmat.shape[0]
    medoids = [rnd.randrange(n)]
    for _ in range(1, k):
        dmin = np.min(Dmat[:, medoids], axis=1)
        w = dmin * dmin
        s = float(np.sum(w))
        if s <= 1e-12:
            cand = [i for i in range(n) if i not in medoids]
            medoids.append(rnd.choice(cand))
            continue
        r = rnd.random() * s
        csum = 0.0
        pick = 0
        for i in range(n):
            csum += w[i]
            if csum >= r:
                pick = i
                break
        if pick in medoids:
            cand = [i for i in range(n) if i not in medoids]
            pick = rnd.choice(cand)
        medoids.append(pick)
    return medoids


def pam_on_dmat(Dmat, k, seed=0, max_iter=60):
    """
    PAM-like k-medoids on a precomputed distance matrix (used for silhouette-based k selection).
    """
    n = Dmat.shape[0]
    medoids = init_medoids_pp(Dmat, k, seed=seed)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist_to = Dmat[:, medoids]
        labels_new = np.argmin(dist_to, axis=1)

        medoids_new = []
        for c in range(k):
            idx = np.where(labels_new == c)[0]
            if len(idx) == 0:
                far = int(np.argmax(np.min(dist_to, axis=1)))
                medoids_new.append(far)
                continue
            sub = Dmat[np.ix_(idx, idx)]
            best_local = idx[int(np.argmin(np.sum(sub, axis=1)))]
            medoids_new.append(int(best_local))

        medoids_new = list(dict.fromkeys(medoids_new))
        while len(medoids_new) < k:
            cand = [i for i in range(n) if i not in medoids_new]
            medoids_new.append(cand[(seed + len(medoids_new)) % len(cand)])

        if np.all(labels_new == labels) and medoids_new == medoids:
            labels = labels_new
            medoids = medoids_new
            break
        labels = labels_new
        medoids = medoids_new
    return labels, medoids


def silhouette_score(Dmat, labels):
    """
    Compute mean silhouette score given distance matrix and cluster labels.
    """
    n = Dmat.shape[0]
    labs = np.array(labels, dtype=int)
    uniq = np.unique(labs)
    if len(uniq) < 2:
        return -1.0
    clusters = {c: np.where(labs == c)[0] for c in uniq}

    s_all = []
    for i in range(n):
        ci = labs[i]
        in_idx = clusters[ci]
        if len(in_idx) <= 1:
            a = 0.0
        else:
            a = float(np.sum(Dmat[i, in_idx]) / float(len(in_idx) - 1))
        b = float("inf")
        for c in uniq:
            if c == ci:
                continue
            idx = clusters[c]
            b = min(b, float(np.mean(Dmat[i, idx])))
        denom = max(a, b)
        s = 0.0 if denom <= 1e-12 else (b - a) / denom
        s_all.append(s)
    return float(np.mean(s_all))


def kmedoids_full(X, k, seed=0, max_iter=30, medoid_candidates=80):
    """
    Approximate k-medoids in full feature space without computing full NxN matrix.
    Uses randomized candidate medoid updates to keep costs manageable.
    """
    rnd = np.random.RandomState(seed)
    N = X.shape[0]
    medoids = list(rnd.choice(N, size=k, replace=False))
    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        dist = np.empty((N, k), dtype=float)
        x2 = np.sum(X * X, axis=1)
        D = float(X.shape[1])
        for j, mi in enumerate(medoids):
            m = X[mi]
            m2 = float(np.dot(m, m))
            xm = np.dot(X, m)
            d2 = (x2 + m2 - 2.0 * xm) / D
            d2 = np.maximum(d2, 0.0)
            dist[:, j] = np.sqrt(d2)
        labels_new = np.argmin(dist, axis=1)

        medoids_new = []
        for c in range(k):
            idx = np.where(labels_new == c)[0]
            if len(idx) == 0:
                far = int(np.argmax(np.min(dist, axis=1)))
                medoids_new.append(far)
                continue

            cand = idx if len(idx) <= medoid_candidates else rnd.choice(idx, size=medoid_candidates, replace=False)
            Xc = X[idx]
            Dc = float(Xc.shape[1])
            x2c = np.sum(Xc * Xc, axis=1)

            best = None
            best_val = float("inf")
            for ci in cand:
                m = X[ci]
                m2 = float(np.dot(m, m))
                xm = np.dot(Xc, m)
                d2 = (x2c + m2 - 2.0 * xm) / Dc
                d2 = np.maximum(d2, 0.0)
                val = float(np.mean(np.sqrt(d2)))
                if val < best_val:
                    best_val = val
                    best = int(ci)
            medoids_new.append(best)

        medoids_new = list(dict.fromkeys(medoids_new))
        while len(medoids_new) < k:
            cand = int(rnd.randint(0, N))
            if cand not in medoids_new:
                medoids_new.append(cand)

        if np.all(labels_new == labels) and medoids_new == medoids:
            labels = labels_new
            medoids = medoids_new
            break

        labels = labels_new
        medoids = medoids_new

    return labels, medoids


def recompute_medoids_in_fullspace(X_full, labels, k, seed=7, max_candidates=200):
    """
    If clustering was done in reduced space (e.g., PCA), recompute medoids in full space
    to yield physically interpretable representative structures.
    """
    rnd = np.random.RandomState(seed)
    medoids = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            medoids.append(0)
            continue
        if len(idx) == 1:
            medoids.append(int(idx[0]))
            continue
        cand = idx if len(idx) <= max_candidates else rnd.choice(idx, size=max_candidates, replace=False)

        Xc = X_full[idx]
        Dc = float(Xc.shape[1])
        x2c = np.sum(Xc * Xc, axis=1)

        best = None
        best_val = float("inf")
        for ci in cand:
            m = X_full[int(ci)]
            m2 = float(np.dot(m, m))
            xm = np.dot(Xc, m)
            d2 = (x2c + m2 - 2.0 * xm) / Dc
            d2 = np.maximum(d2, 0.0)
            val = float(np.mean(np.sqrt(d2)))
            if val < best_val:
                best_val = val
                best = int(ci)
        medoids.append(best if best is not None else int(idx[0]))
    return medoids


# =============================================================================
# Per-frame metrics and cluster "tightness" metrics
# =============================================================================

def compute_frame_metrics(st, lig_asl, prot_asl, site_keys, contact_cutoff):
    """
    Compute per-frame metrics used later for cluster ranking.

    Metrics
    -------
    contact : float
      Fraction of binding-site residues that have any heavy-atom contact to ligand
      within contact_cutoff (Å).
    com_dist : float
      Distance between ligand COM and binding-site heavy-atom COM (Å).
      Used indirectly via COM std within cluster as a stability proxy.
    inter_strength : float
      Weighted interaction proxy score from counts of detected interactions plus
      hydrophobic contact fraction.

    Notes
    -----
    These are proxies designed for ranking/triage rather than rigorous free-energy estimates.
    """
    lig_aids = heavy_aids(st, evaluate_asl(st, lig_asl))
    prot_aids = evaluate_asl(st, prot_asl)
    if not lig_aids:
        return None

    site_ct_aids = []
    per_res = []
    for k in site_keys:
        r_aids = residue_aids_by_key(st, k)
        if r_aids:
            site_ct_aids.extend(heavy_aids(st, r_aids))
        dmin = min_heavy_distance(st, r_aids, lig_aids)
        per_res.append(1.0 if dmin < contact_cutoff else 0.0)

    contact_frac = float(np.mean(per_res)) if per_res else 0.0

    if site_ct_aids:
        com_l = center_of_mass(st, lig_aids)
        com_s = center_of_mass(st, site_ct_aids)
        com_dist = float(np.linalg.norm(com_l - com_s))
        prot_for_inter = site_ct_aids
    else:
        com_dist = float("nan")
        prot_for_inter = prot_aids

    hyd = hydrophobic_contact_fraction(st, prot_for_inter, lig_aids, cutoff=4.0)
    ic = interaction_counts_single_ct(st, prot_for_inter, lig_aids)

    avg_hb = float(ic.get("HBond", 0))
    avg_sb = float(ic.get("SaltBridge", 0))
    avg_pp = float(ic.get("PiPi", 0))
    avg_pc = float(ic.get("PiCat", 0))
    avg_hal = float(ic.get("Halogen", 0))

    # A simple composite interaction proxy:
    inter_strength = (avg_hb + 2.0 * avg_sb + 0.5 * (avg_pp + avg_pc) + 0.5 * avg_hal + 0.5 * hyd)
    return {"contact": contact_frac, "com_dist": com_dist, "inter_strength": inter_strength}


def mean_pairwise_subset_rmsd(X_cluster, max_n=200, seed=7):
    """
    Estimate cluster tightness as mean pairwise distance (subset-sampled if needed).
    """
    n = X_cluster.shape[0]
    if n < 2:
        return 0.0
    if n > max_n:
        rnd = np.random.RandomState(seed)
        idx = rnd.choice(n, size=max_n, replace=False)
        Xs = X_cluster[idx]
    else:
        Xs = X_cluster
    D = pairwise_dist_matrix(Xs)
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.mean(D[iu])) if len(iu[0]) else 0.0


# =============================================================================
# Timeseries / kinetics summary
# =============================================================================

def compute_runs(labels):
    """
    Convert a label timeseries into contiguous runs (dwell segments).

    Returns
    -------
    runs : list of dict(cluster, start, end, length)
    """
    labs = list(map(int, list(labels)))
    if not labs:
        return []
    runs = []
    cur = labs[0]
    start = 0
    for i in range(1, len(labs)):
        if labs[i] != cur:
            runs.append({"cluster": cur, "start": start, "end": i - 1, "length": (i - start)})
            cur = labs[i]
            start = i
    runs.append({"cluster": cur, "start": start, "end": len(labs) - 1, "length": (len(labs) - start)})
    return runs


def transition_matrix(labels, k):
    """Compute stepwise transition count matrix T[from,to]."""
    labs = list(map(int, list(labels)))
    T = np.zeros((k, k), dtype=int)
    for i in range(1, len(labs)):
        a = labs[i - 1]
        b = labs[i]
        if (0 <= a < k) and (0 <= b < k):
            T[a, b] += 1
    return T


def write_timeseries_outputs(prefix, labels, k, frame_dt_ns=0.0):
    """
    Write:
      - runs CSV (start/end/length, plus ns if frame_dt_ns is provided)
      - summary TXT (transition matrices + dwell stats)
    """
    runs = compute_runs(labels)
    runs_csv = "{}_cluster_runs.csv".format(prefix)
    summ_txt = "{}_cluster_timeseries_summary.txt".format(prefix)

    with open(runs_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "cluster_id", "start_i", "end_i", "length_frames",
                    "length_ns" if frame_dt_ns and frame_dt_ns > 0 else ""])
        for r_i, r in enumerate(runs, start=1):
            if frame_dt_ns and frame_dt_ns > 0:
                w.writerow([r_i, r["cluster"], r["start"], r["end"], r["length"],
                            "{:.6f}".format(r["length"] * frame_dt_ns)])
            else:
                w.writerow([r_i, r["cluster"], r["start"], r["end"], r["length"]])

    T = transition_matrix(labels, k)
    total_trans = int(T.sum())
    offdiag = int((T.sum() - np.trace(T)))

    per = {}
    for r in runs:
        c = int(r["cluster"])
        per.setdefault(c, []).append(int(r["length"]))

    def fmt_ns(frames):
        if frame_dt_ns and frame_dt_ns > 0:
            return "{:.3f} ns".format(frames * frame_dt_ns)
        return "{} frames".format(frames)

    with open(summ_txt, "w") as f:
        f.write("Cluster timeseries summary\n")
        f.write("N_frames: {}\n".format(len(labels)))
        if frame_dt_ns and frame_dt_ns > 0:
            f.write("frame_dt_ns: {:.6f}\n".format(frame_dt_ns))
        f.write("\nTransitions:\n")
        f.write("  total_step_transitions (i-1 -> i): {}\n".format(total_trans))
        f.write("  off-diagonal transitions (switch events counted per step): {}\n".format(offdiag))
        f.write("  number_of_runs (blocks): {}\n".format(len(runs)))
        f.write("\nTransition count matrix (rows=from, cols=to):\n")
        for i in range(k):
            f.write("  from {}: {}\n".format(i, " ".join(map(str, list(T[i, :])))))

        f.write("\nTransition probability matrix (row-normalized):\n")
        for i in range(k):
            row = T[i, :].astype(float)
            s = row.sum()
            if s > 0:
                row = row / s
            f.write("  from {}: {}\n".format(i, " ".join(["{:.3f}".format(x) for x in row])))

        f.write("\nDwell/run statistics (by cluster):\n")
        for c in range(k):
            lens = per.get(c, [])
            if not lens:
                f.write("  cluster {}: no runs\n".format(c))
                continue
            arr = np.array(lens, dtype=float)
            f.write("  cluster {}:\n".format(c))
            f.write("    runs: {}\n".format(len(lens)))
            f.write("    total_time: {}\n".format(fmt_ns(int(arr.sum()))))
            f.write("    mean_run: {}\n".format(fmt_ns(float(arr.mean()))))
            f.write("    median_run: {}\n".format(fmt_ns(float(np.median(arr)))))
            f.write("    min_run: {}\n".format(fmt_ns(int(arr.min()))))
            f.write("    max_run: {}\n".format(fmt_ns(int(arr.max()))))

        all_len = np.array([r["length"] for r in runs], dtype=float) if runs else np.array([0.0])
        f.write("\nGlobal dwell/run stats:\n")
        f.write("  mean_run: {}\n".format(fmt_ns(float(all_len.mean()))))
        f.write("  median_run: {}\n".format(fmt_ns(float(np.median(all_len)))))
        f.write("  max_run: {}\n".format(fmt_ns(float(all_len.max()))))

    return runs_csv, summ_txt


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="MD trajectory analysis: extract -> align -> feature -> (PCA) -> k-medoids -> rank -> kinetics"
    )

    # Inputs
    ap.add_argument("--out_cms", required=True, help="Desmond output CMS file.")
    ap.add_argument("--trj_dir", required=True, help="Desmond trajectory directory.")
    ap.add_argument("--out_prefix", default="UNK_auto", help="Prefix for all output files.")

    # Working directories
    ap.add_argument("--trajectory_frames_dir", default="trajectory_frames",
                    help="Folder for extracted frame files.")
    ap.add_argument("--aligned_frames_dir", default="aligned_trajectory_frames",
                    help="Folder for aligned frame files.")

    # Extraction settings
    ap.add_argument("--basename", default="frame", help="Base name for extracted frames.")
    ap.add_argument("--extract_asl",
                    default='((res.ptype UNK) or (res.ptype "UNK ") or (protein))',
                    help="ASL for extraction (which atoms to write to frames).")
    ap.add_argument("--stride_extract", type=int, default=1,
                    help="Stride for frame extraction. Example: 100 keeps every 100th frame.")

    ap.add_argument("--skip_extract", action="store_true", help="Skip extraction if frames already exist.")
    ap.add_argument("--skip_align", action="store_true", help="Skip alignment if aligned frames already exist.")

    # Alignment & selections
    ap.add_argument("--fit_asl", default="protein and backbone and not atom.ele H",
                    help="ASL used for alignment (Kabsch fit atoms).")
    ap.add_argument("--lig_asl", default="res.ptype UNK and not atom.ele H",
                    help="ASL to define the ligand heavy atoms.")
    ap.add_argument("--prot_asl", default="protein", help="ASL to define the protein.")

    # Binding site definition
    ap.add_argument("--site_ref_file", default="",
                    help="Reference structure file for binding-site definition (default: out_cms).")
    ap.add_argument("--site_ref_ct", type=int, default=0, help="CT index in the reference file.")
    ap.add_argument("--site_cutoff", type=float, default=5.0,
                    help="Binding-site residue cutoff (Å): residue is in site if min heavy distance <= cutoff.")
    ap.add_argument("--contact_cutoff", type=float, default=4.0,
                    help="Contact cutoff (Å) for per-frame residue contact fraction.")

    # PCA
    ap.add_argument("--pca_n", type=int, default=0, help="Number of PCA components to compute (0 disables PCA).")
    ap.add_argument("--pca_scale", action="store_true", help="Scale features by std prior to PCA.")
    ap.add_argument("--cluster_on_pca", action="store_true", help="Run clustering on PCA space instead of full space.")
    ap.add_argument("--recompute_medoids_fullspace", action="store_true",
                    help="If clustering on PCA, recompute medoids in full space for interpretability.")

    # Kinetics / timeseries reporting
    ap.add_argument("--frame_dt_ns", type=float, default=0.0,
                    help="Time step (ns) per USED frame (after stride_extract). Used only for dwell reporting.")

    # PCA plot
    ap.add_argument("--skip_pca_plot", action="store_true",
                    help="Disable PCA scatter plot output even if PCA is computed.")
    ap.add_argument("--pca_plot_dpi", type=int, default=300,
                    help="Resolution (dpi) for PCA scatter plot PNG (default=300).")

    # k selection / clustering
    ap.add_argument("--k_min", type=int, default=2, help="Minimum k for silhouette-based search.")
    ap.add_argument("--k_max", type=int, default=0, help="Maximum k (0 = auto based on N).")
    ap.add_argument("--k_sample", type=int, default=400, help="Subsample size for silhouette search.")
    ap.add_argument("--k_restarts", type=int, default=3, help="Restarts per k for silhouette search.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")

    # Ranking & tightness
    ap.add_argument("--max_tightness_frames", type=int, default=200,
                    help="Max frames sampled for pairwise tightness metric per cluster.")

    # Composite score weights
    ap.add_argument("--w_occ", type=float, default=0.20, help="Weight: occupancy.")
    ap.add_argument("--w_contact", type=float, default=0.40, help="Weight: site contact fraction.")
    ap.add_argument("--w_stab", type=float, default=0.20, help="Weight: COM stability (lower std is better).")
    ap.add_argument("--w_tight", type=float, default=0.15, help="Weight: cluster tightness (lower is better).")
    ap.add_argument("--w_inter", type=float, default=0.05, help="Weight: interaction proxy score.")

    ap.add_argument("--top_n", type=int, default=5, help="How many top clusters to print.")

    args = ap.parse_args()

    traj_dir = args.trajectory_frames_dir
    aln_dir = args.aligned_frames_dir

    # -------------------------------------------------------------------------
    # 1) Extract frames
    # -------------------------------------------------------------------------
    ensure_dir(traj_dir)
    if args.skip_extract and sorted_maestro_files(traj_dir):
        print("Skipping extraction (frames already exist in {})".format(traj_dir))
    else:
        print("Extracting frames to {} ... (stride_extract={})".format(traj_dir, args.stride_extract))
        out = run_trj2mae_extract(args.out_cms, args.trj_dir, traj_dir,
                                 args.basename, args.extract_asl, args.stride_extract)
        print(out)

    # -------------------------------------------------------------------------
    # 2) Align frames
    # -------------------------------------------------------------------------
    ensure_dir(aln_dir)
    if args.skip_align and sorted_maestro_files(aln_dir):
        print("Skipping alignment (aligned frames already exist in {})".format(aln_dir))
        aligned_files = sorted_maestro_files(aln_dir)
        extracted_files = sorted_maestro_files(traj_dir)
    else:
        print("Aligning frames -> {} ...".format(aln_dir))
        extracted_files, aligned_files, _ = align_frames(traj_dir, aln_dir, args.fit_asl, ref_index=0)
        print("Aligned frames written: {}".format(len(aligned_files)))

    # Map aligned basename -> original extracted file
    orig_map = {}
    for fp in extracted_files:
        b = os.path.basename(fp)
        orig_map[b] = fp
        if b.endswith(".maegz"):
            orig_map[b[:-6] + ".mae"] = fp

    # -------------------------------------------------------------------------
    # 3) Define binding-site residues from a reference
    # -------------------------------------------------------------------------
    site_ref = args.site_ref_file if args.site_ref_file else args.out_cms
    ref_cts = list(structure.StructureReader(site_ref))
    if not ref_cts:
        raise SystemExit("Reference file empty/unreadable: {}".format(site_ref))
    ref_idx = max(0, min(args.site_ref_ct, len(ref_cts) - 1))
    ref_st = ref_cts[ref_idx]

    lig_ref = heavy_aids(ref_st, evaluate_asl(ref_st, args.lig_asl))
    prot_ref = evaluate_asl(ref_st, args.prot_asl)
    if not lig_ref:
        raise SystemExit("Ligand not found in reference. Check --lig_asl.")
    site_keys = binding_site_from_ligand(ref_st, lig_ref, prot_ref, args.site_cutoff)
    if not site_keys:
        raise SystemExit("Binding site not found. Try increasing --site_cutoff.")

    site_txt = "{}_binding_site_residues.txt".format(args.out_prefix)
    with open(site_txt, "w") as f:
        f.write("Reference: {} CT={}\n".format(os.path.basename(site_ref), ref_idx))
        f.write("Ligand ASL: {}\n".format(args.lig_asl))
        f.write("Site cutoff: {} Å\n".format(args.site_cutoff))
        f.write("Residues (chain resnum inscode):\n")
        for k in site_keys:
            f.write("{} {} {}\n".format(k[0], k[1], k[2]))
    print("Binding-site residue count: {} -> {}".format(len(site_keys), site_txt))

    # -------------------------------------------------------------------------
    # 4) Feature matrix (full space)
    # -------------------------------------------------------------------------
    st0 = next(structure.StructureReader(aligned_files[0]))
    names_ref = ligand_atom_names_from_ref(st0, args.lig_asl)
    X_full, aligned_used = build_feature_matrix(aligned_files, args.lig_asl, names_ref)

    if X_full.shape[0] < 10:
        raise SystemExit("Too few usable aligned frames for clustering: {}".format(X_full.shape[0]))

    N = X_full.shape[0]
    print("Frames for clustering: {}  full_feature_dim={}".format(N, X_full.shape[1]))

    # -------------------------------------------------------------------------
    # 4b) PCA (optional)
    # -------------------------------------------------------------------------
    X_used = X_full
    pca_csv = ""
    pca_txt = ""
    X_red_for_plot = None

    if args.pca_n and int(args.pca_n) > 0:
        ncomp = int(args.pca_n)
        ncomp = max(1, min(ncomp, X_full.shape[1], X_full.shape[0]))
        print("Running PCA (n_components={}, scale={}) ...".format(ncomp, bool(args.pca_scale)))
        X_red, comps, evr, mu, std = pca_reduce_numpy(X_full, n_components=ncomp, scale=bool(args.pca_scale))
        X_red_for_plot = X_red
        pca_csv, pca_txt = write_pca_outputs(args.out_prefix, aligned_used, orig_map, X_red, evr)
        print("PCA outputs: {}, {}".format(pca_csv, pca_txt))

        if args.cluster_on_pca:
            X_used = X_red
            print("Clustering will run on PCA-reduced space (dim={}).".format(X_used.shape[1]))
        else:
            print("Clustering will run on FULL space; PCA is for analysis/plotting only.")
    else:
        if args.cluster_on_pca:
            print("WARNING: --cluster_on_pca given but --pca_n=0. Using FULL space.")
        args.cluster_on_pca = False

    # -------------------------------------------------------------------------
    # 5) Auto-select k via silhouette on a sample (in X_used space)
    # -------------------------------------------------------------------------
    if args.k_max <= 0:
        args.k_max = int(min(25, max(args.k_min + 2, round(math.sqrt(N)))))
    args.k_max = max(args.k_min, args.k_max)

    rnd = np.random.RandomState(args.seed)
    n_sample = min(N, args.k_sample)
    sample_idx = rnd.choice(N, size=n_sample, replace=False) if n_sample < N else np.arange(N)
    Xs = X_used[sample_idx]
    Dmat = pairwise_dist_matrix(Xs)

    sil_csv = "{}_k_silhouette.csv".format(args.out_prefix)
    best_k = None
    best_adj = -1e9
    rows = []

    print("Auto-k search: k in [{}..{}], sample={} (space={})".format(
        args.k_min, args.k_max, n_sample, "PCA" if args.cluster_on_pca else "FULL"
    ))
    for k in range(args.k_min, args.k_max + 1):
        sil_best = -1e9
        for r in range(args.k_restarts):
            labels_s, _ = pam_on_dmat(Dmat, k, seed=args.seed + 31 * r + 7 * k, max_iter=60)
            sil = silhouette_score(Dmat, labels_s)
            sil_best = max(sil_best, sil)
        penalty = 0.02 * (k / float(max(1, args.k_max)))
        adj = sil_best - penalty
        rows.append((k, sil_best, adj))
        print("  k={} silhouette={:.4f} adj={:.4f}".format(k, sil_best, adj))
        if adj > best_adj:
            best_adj = adj
            best_k = k

    with open(sil_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["k", "silhouette_best", "adjusted_score"])
        for k, sil, adj in rows:
            w.writerow([k, "{:.6f}".format(sil), "{:.6f}".format(adj)])

    print("Chosen k = {}  (see {})".format(best_k, sil_csv))

    # -------------------------------------------------------------------------
    # 6) Final clustering (on X_used)
    # -------------------------------------------------------------------------
    labels, medoids = kmedoids_full(X_used, best_k, seed=args.seed, max_iter=30, medoid_candidates=80)

    # If clustering on PCA, optionally recompute medoids in full space
    if args.cluster_on_pca and args.recompute_medoids_fullspace:
        print("Recomputing medoids in FULL space for physical interpretability ...")
        medoids = recompute_medoids_in_fullspace(X_full, labels, best_k, seed=args.seed, max_candidates=250)

    # PCA plot after clustering
    pca_png = ""
    if (X_red_for_plot is not None) and (X_red_for_plot.shape[1] >= 2) and (not args.skip_pca_plot):
        pca_png = plot_pca_clusters_png(
            args.out_prefix,
            X_red_for_plot[:, :2],
            labels,
            medoids=medoids,
            dpi=args.pca_plot_dpi
        )
        if pca_png:
            print("PCA cluster plot: {}".format(pca_png))

    # Timeseries outputs
    runs_csv, kin_txt = write_timeseries_outputs(args.out_prefix, labels, best_k, frame_dt_ns=args.frame_dt_ns)
    print("Timeseries outputs: {}, {}".format(runs_csv, kin_txt))

    # -------------------------------------------------------------------------
    # 7) Per-frame metrics for ranking (computed on aligned frames)
    # -------------------------------------------------------------------------
    print("Computing per-frame metrics (contacts/COM/interactions) ...")
    frame_metrics = []
    for fp in aligned_used:
        st = next(structure.StructureReader(fp))
        m = compute_frame_metrics(st, args.lig_asl, args.prot_asl, site_keys, args.contact_cutoff)
        frame_metrics.append(m)

    # -------------------------------------------------------------------------
    # 8) Cluster aggregates + tightness in FULL space
    # -------------------------------------------------------------------------
    clusters = []
    for c in range(best_k):
        idx = np.where(labels == c)[0]
        members = int(len(idx))
        occ = members / float(N)

        contacts = [frame_metrics[i]["contact"] for i in idx if frame_metrics[i] is not None]
        coms = [frame_metrics[i]["com_dist"] for i in idx
                if (frame_metrics[i] is not None and math.isfinite(frame_metrics[i]["com_dist"]))]
        inters = [frame_metrics[i]["inter_strength"] for i in idx if frame_metrics[i] is not None]

        contact_mean = float(np.mean(contacts)) if contacts else 0.0
        contact_min = float(np.min(contacts)) if contacts else 0.0
        com_std = float(np.std(coms)) if coms else float("nan")
        inter_strength = float(np.mean(inters)) if inters else 0.0

        med_i = int(medoids[c])
        med_aln = aligned_used[med_i]
        med_base = os.path.basename(med_aln)
        med_orig = orig_map.get(med_base, "")

        Xc_full = X_full[idx]
        x_med_full = X_full[med_i]
        tight_to_med = float(np.mean([rmsd_flat(X_full[i], x_med_full) for i in idx])) if members > 0 else float("nan")
        tight_pair = mean_pairwise_subset_rmsd(Xc_full, max_n=args.max_tightness_frames, seed=args.seed + 13 * c)

        clusters.append({
            "cluster_id": c,
            "members": members,
            "occ": occ,
            "contact_mean": contact_mean,
            "contact_min": contact_min,
            "com_std": com_std,
            "tight_to_medoid": tight_to_med,
            "tight_pairwise_subset": tight_pair,
            "inter_strength": inter_strength,
            "medoid_i": med_i,
            "medoid_aligned_file": med_aln,
            "medoid_original_file": med_orig
        })

    # -------------------------------------------------------------------------
    # 9) Robust scaling + final composite score
    # -------------------------------------------------------------------------
    occ_s = robust_scale([m["occ"] for m in clusters], higher_is_better=True)
    contact_s = robust_scale([m["contact_mean"] for m in clusters], higher_is_better=True)
    stab_s = robust_scale([m["com_std"] for m in clusters], higher_is_better=False)
    tight_s = robust_scale([m["tight_pairwise_subset"] for m in clusters], higher_is_better=False)
    inter_s = robust_scale([m["inter_strength"] for m in clusters], higher_is_better=True)

    for i, m in enumerate(clusters):
        m["occ_s"] = occ_s[i]
        m["contact_s"] = contact_s[i]
        m["stab_s"] = stab_s[i]
        m["tight_s"] = tight_s[i]
        m["inter_s"] = inter_s[i]
        m["score"] = float(args.w_occ * occ_s[i] +
                           args.w_contact * contact_s[i] +
                           args.w_stab * stab_s[i] +
                           args.w_tight * tight_s[i] +
                           args.w_inter * inter_s[i])

    clusters.sort(key=lambda x: x["score"], reverse=True)

    # -------------------------------------------------------------------------
    # 10) Write per-frame cluster assignments
    # -------------------------------------------------------------------------
    assign_csv = "{}_cluster_assignments.csv".format(args.out_prefix)
    with open(assign_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["i", "aligned_file", "original_file", "cluster_id"])
        for i in range(N):
            aln_fp = aligned_used[i]
            base = os.path.basename(aln_fp)
            orig_fp = orig_map.get(base, "")
            w.writerow([i, aln_fp, orig_fp, int(labels[i])])

    # -------------------------------------------------------------------------
    # 11) Write ranking CSV + export medoid structures
    # -------------------------------------------------------------------------
    rank_csv = "{}_cluster_ranking.csv".format(args.out_prefix)
    with open(rank_csv, "w") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "cluster_id", "members",
            "occ", "occ_scaled",
            "site_contact_mean", "contact_scaled",
            "lig_site_com_std", "stab_scaled",
            "tight_pairwise_subset_ligRMSD", "tight_scaled",
            "tight_mean_to_medoid_ligRMSD",
            "inter_strength", "inter_scaled",
            "score",
            "medoid_i",
            "medoid_aligned_file", "medoid_original_file",
            "medoid_aligned_out", "medoid_original_out"
        ])

        for r, m in enumerate(clusters, start=1):
            out_aln = "{}_cluster{}_medoid_aligned.mae".format(args.out_prefix, m["cluster_id"])
            out_org = "{}_cluster{}_medoid_original.mae".format(args.out_prefix, m["cluster_id"])

            st_aln = next(structure.StructureReader(m["medoid_aligned_file"]))
            w_aln = structure.StructureWriter(out_aln)
            w_aln.append(st_aln)
            w_aln.close()

            if m["medoid_original_file"] and os.path.isfile(m["medoid_original_file"]):
                st_org = next(structure.StructureReader(m["medoid_original_file"]))
                w_org = structure.StructureWriter(out_org)
                w_org.append(st_org)
                w_org.close()
            else:
                out_org = ""

            w.writerow([
                r, m["cluster_id"], m["members"],
                "{:.6f}".format(m["occ"]), "{:.3f}".format(m["occ_s"]),
                "{:.6f}".format(m["contact_mean"]), "{:.3f}".format(m["contact_s"]),
                (("{:.6f}".format(m["com_std"]) if math.isfinite(m["com_std"]) else "")),
                "{:.3f}".format(m["stab_s"]),
                "{:.6f}".format(m["tight_pairwise_subset"]), "{:.3f}".format(m["tight_s"]),
                "{:.6f}".format(m["tight_to_medoid"]),
                "{:.6f}".format(m["inter_strength"]), "{:.3f}".format(m["inter_s"]),
                "{:.6f}".format(m["score"]),
                m["medoid_i"],
                m["medoid_aligned_file"], m["medoid_original_file"],
                out_aln, out_org
            ])

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("\nDONE")
    print("  k silhouette : {}".format(sil_csv))
    print("  assignments  : {}".format(assign_csv))
    print("  ranking      : {}".format(rank_csv))
    print("  site residues: {}".format(site_txt))
    print("  kinetics     : {}".format(kin_txt))
    print("  runs         : {}".format(runs_csv))
    if pca_csv:
        print("  PCA coords   : {}".format(pca_csv))
    if pca_txt:
        print("  PCA summary  : {}".format(pca_txt))
    if pca_png:
        print("  PCA plot     : {}".format(pca_png))

    print("\nTOP {} clusters:".format(max(1, args.top_n)))
    for i, m in enumerate(clusters[:max(1, args.top_n)], start=1):
        print("[{}] cluster={} score={:.3f} occ={:.3f} contact={:.3f} COMstd={} tight_pair={:.3f} inter={:.3f} medoid_orig={}".format(
            i, m["cluster_id"], m["score"], m["occ"], m["contact_mean"],
            ("{:.3f}".format(m["com_std"]) if math.isfinite(m["com_std"]) else "nan"),
            m["tight_pairwise_subset"], m["inter_strength"],
            ("{}_cluster{}_medoid_original.mae".format(args.out_prefix, m["cluster_id"]))
        ))


if __name__ == "__main__":
    main()
