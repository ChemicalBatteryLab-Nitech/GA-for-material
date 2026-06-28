#!/usr/bin/env python3
# coding: utf-8
"""
GAML standard ML interface for GmAte.py.

Public API
----------
train(genes, energies) -> (model, rmse)
predict(model, candidate_genes) -> sorted_indices
"""

import os
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from ase.io import read
from ase.geometry.analysis import Analysis

# Loaded once at import time (GmAte.py runs from the project root).
from Specific import inp_POSCAR as _inp
_ELEM = _inp.ELEM
_ions = _inp.ions
_POSCAR_ORG = os.path.join("Specific", "POSCAR_org")

_RDF_RMAX = 4
_RDF_NBINS = 20
_RDF_KEYS = None  # built lazily


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rdf_keys():
    """Build RDF feature key list (mirrors tabledescript in GmAte.py)."""
    ionsl = sorted(_ions)[0]
    keys, seen = [], set()
    for a in ionsl:
        for b in ionsl:
            k = f"rdf_{a}-{b}"
            if k not in seen:
                keys.append(k)
                seen.add(k)
    for d in ionsl:
        keys.append(f"rdf_{d}")
    keys.append("rdfall")
    return keys


def _get_rdf_keys():
    global _RDF_KEYS
    if _RDF_KEYS is None:
        _RDF_KEYS = _build_rdf_keys()
    return _RDF_KEYS


def _build_poscar_str(gene_strings):
    """
    Reconstruct POSCAR content from a list of gene strings.
    Mirrors Indivisual.gene_to_POSCAR() but returns a string instead of
    writing to disk.
    """
    with open(_POSCAR_ORG, "r") as f:
        all_lines = f.readlines()
    all_lines = [line.split() for line in all_lines]
    labels = [row[:] for row in all_lines[:8]]
    lines  = [row[:] for row in all_lines[8:]]

    strings = gene_strings
    label_info = []
    for i in range(len(labels[5])):
        if "ELEM" in labels[5][i]:
            flag = labels[5][i].lstrip("ELEM")
            flag = 1 if flag == "" else int(flag)
            for xx, x in enumerate(_ELEM[flag - 1]):
                count = str(str(strings[flag - 1]).count(str(xx)))
                if x != "Vac":
                    label_info.append([x, count])
        else:
            label_info.append([labels[5][i], labels[6][i]])
    label_info_sort = sorted(label_info, key=lambda x: x[0])
    labels[5] = [item[0] for item in label_info_sort]
    labels[6] = [item[1] for item in label_info_sort]

    count_list = [0] * len(strings)
    for line in lines:
        if "ELEM" in line[3]:
            flag = line[3].lstrip("ELEM")
            flag = 1 if flag == "" else int(flag)
            gen = int(strings[flag - 1][count_list[flag - 1]])
            count_list[flag - 1] += 1
            line[3] = _ELEM[flag - 1][gen]
    lines_sort = sorted(lines, key=lambda x: x[3])

    rows = [" ".join(row) for row in labels]
    for line in lines_sort:
        if "Vac" not in line:
            rows.append(" ".join(line))
    return "\n".join(rows) + "\n"


def _gene_to_atoms(gene_strings):
    """Convert gene strings to an ASE Atoms object (in-memory, no disk I/O)."""
    poscar_str = _build_poscar_str(gene_strings)
    return read(io.StringIO(poscar_str), format="vasp")


def _compute_features(atoms):
    """Compute concatenated RDF feature vector from an ASE Atoms object."""
    rdf_keys = _get_rdf_keys()
    analysis = Analysis(atoms)
    features = []
    for key in rdf_keys:
        if "all" in key:
            nelem = None
        else:
            nelem = tuple(key.replace("rdf_", "").split("-"))
        rdf = analysis.get_rdf(_RDF_RMAX, _RDF_NBINS, elements=nelem)
        features.extend(rdf[0].tolist())
    return features


def _genes_to_matrix(genes):
    """
    Convert a list of gene-string lists to a feature matrix.
    Returns (X_valid, valid_indices) where X_valid[i] corresponds to
    genes[valid_indices[i]].
    """
    X_valid, valid_indices = [], []
    for i, gene_strings in enumerate(genes):
        try:
            atoms = _gene_to_atoms(gene_strings)
            feats = _compute_features(atoms)
            X_valid.append(feats)
            valid_indices.append(i)
        except Exception as e:
            print(f"GAML: descriptor failed for gene {i} ({gene_strings}): {e}")
    return X_valid, valid_indices


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(genes: list, energies: list) -> tuple:
    """
    Train a Random Forest model on gene strings and their energies.

    Parameters
    ----------
    genes : list of list of str
        Gene strings for evaluated individuals.  Each element is a list of
        strings with length == NUM_OF_STRINGS (one string per ELEM group).
    energies : list of float
        Corresponding energy values (lower = more stable).

    Returns
    -------
    model : trained RandomForestRegressor, or None if training fails.
    rmse : float
        Best cross-validation RMSE; inf if insufficient data.
    """
    X_raw, valid_idx = _genes_to_matrix(genes)
    if len(X_raw) < 5:
        print(f"GAML train: only {len(X_raw)} valid samples — skipping.")
        return None, float('inf')

    X = np.array(X_raw, dtype=float)
    y = np.array([energies[i] for i in valid_idx], dtype=float)

    param_grid = {
        'max_depth':        [5, 10],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators':     [10, 30],
    }
    n_splits = min(10, len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=777)

    best_val_rmse = float('inf')
    best_params   = {'max_depth': 5, 'min_samples_leaf': 1, 'n_estimators': 10}

    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_leaf in param_grid['min_samples_leaf']:
                pred_val, ans_val = [], []
                for tr_idx, va_idx in kf.split(X):
                    clf = RandomForestRegressor(
                        max_depth=max_d, min_samples_leaf=min_leaf,
                        n_estimators=n_est, n_jobs=-1, random_state=42
                    )
                    clf.fit(X[tr_idx], y[tr_idx])
                    pred_val.extend(clf.predict(X[va_idx]).tolist())
                    ans_val.extend(y[va_idx].tolist())
                val_rmse = _rmse(ans_val, pred_val)
                print(f"GAML train: n_est={n_est}, max_depth={max_d}, "
                      f"min_leaf={min_leaf}, val_rmse={val_rmse:.4f}")
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params   = {
                        'max_depth':        max_d,
                        'min_samples_leaf': min_leaf,
                        'n_estimators':     n_est,
                    }

    model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
    model.fit(X, y)
    print(f"GAML train: best val_rmse={best_val_rmse:.4f}, params={best_params}")
    return model, best_val_rmse


def predict(model, candidate_genes: list) -> list:
    """
    Predict energies for candidate genes and return indices sorted ascending.

    Parameters
    ----------
    model : trained model returned by train(), or None.
    candidate_genes : list of list of str
        Gene strings for candidate individuals (same format as genes in train()).

    Returns
    -------
    sorted_indices : list of int
        Indices into candidate_genes sorted by predicted energy (ascending).
        Candidates whose descriptor computation failed are appended at the end.
    """
    if model is None or not candidate_genes:
        return list(range(len(candidate_genes)))

    X_raw, valid_idx = _genes_to_matrix(candidate_genes)
    failed_idx = [i for i in range(len(candidate_genes)) if i not in valid_idx]

    if not X_raw:
        return list(range(len(candidate_genes)))

    X     = np.array(X_raw, dtype=float)
    preds = model.predict(X)
    order = np.argsort(preds)
    return [valid_idx[j] for j in order] + failed_idx
