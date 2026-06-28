#!/usr/bin/env python
# coding: utf-8
"""
calc_engine_mace.py  --  MACE universal NNP engine for GmAte

Setup
-----
    pip install mace-torch

Usage
-----
    cp example/Specific_mace/* Specific/
    # Edit Specific/inp_POSCAR.py (ions, ELEM)
    # Edit user settings below if needed
    python GmAte.py -ga

n_parallel (inp_ga.py)
----------------------
    n_parallel = CPU cores   (CPU-only, torch.set_num_threads(1) is set below)
    n_parallel = GPU count   (set USE_GPU = True below)
"""

import os

# ── user settings ─────────────────────────────────────────────────────────────
MODEL      = "medium"   # MACE-MP preset: "small", "medium", "large"
MODEL_PATH = None       # local .pt checkpoint path; overrides MODEL when set
DTYPE      = "float32"  # "float32" or "float64"
DISPERSION = False      # add D3 dispersion correction
FMAX       = 0.05       # force convergence criterion (eV/Å)
MAXSTEP    = 300        # max optimizer steps
RELAX_CELL = True       # True: relax cell + atoms;  False: atoms only
USE_GPU    = False      # True: use CUDA GPU
# ──────────────────────────────────────────────────────────────────────────────

_calculator = None   # loaded once per worker process


def _get_calculator():
    global _calculator
    if _calculator is not None:
        return _calculator
    import torch
    torch.set_num_threads(1)
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    from mace.calculators import mace_mp, MACECalculator
    if MODEL_PATH is not None:
        _calculator = MACECalculator(model_paths=MODEL_PATH,
                                     default_dtype=DTYPE, device=device)
    else:
        _calculator = mace_mp(model=MODEL, dispersion=DISPERSION,
                              default_dtype=DTYPE, device=device)
    return _calculator


def run(work_dir: str) -> float:
    """
    Relax the structure in work_dir with MACE and return total energy (eV).

    GmAte writes POSCAR to work_dir before calling this function.
    This function writes:
        CONTCAR  -- relaxed structure (VASP format)
        energy   -- total energy in eV (one line)
        log.mace -- optimizer log
    """
    from ase.io import read, write
    from ase.optimize import LBFGS

    atoms = read(os.path.join(work_dir, "POSCAR"), format="vasp")
    atoms.calc = _get_calculator()

    if RELAX_CELL:
        try:
            from ase.filters import FrechetCellFilter
            runner = FrechetCellFilter(atoms)
        except ImportError:
            from ase.constraints import ExpCellFilter
            runner = ExpCellFilter(atoms)
    else:
        runner = atoms

    opt = LBFGS(runner, logfile=os.path.join(work_dir, "log.mace"),
                trajectory=None)
    opt.run(fmax=FMAX, steps=MAXSTEP)

    energy = float(atoms.get_potential_energy())

    write(os.path.join(work_dir, "CONTCAR"), atoms, format="vasp", direct=True)
    with open(os.path.join(work_dir, "energy"), "w") as f:
        f.write(f"{energy}\n")

    return energy
