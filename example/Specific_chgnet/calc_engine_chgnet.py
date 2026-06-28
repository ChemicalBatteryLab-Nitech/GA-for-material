#!/usr/bin/env python
# coding: utf-8
"""
calc_engine_chgnet.py  --  CHGNet universal NNP engine for GmAte

Setup
-----
    pip install chgnet

Usage
-----
    cp example/Specific_chgnet/* Specific/
    # Edit Specific/inp_POSCAR.py (ions, ELEM)
    # Edit user settings below if needed
    python GmAte.py -ga

n_parallel (inp_ga.py)
----------------------
    n_parallel = CPU cores   (torch.set_num_threads(1) is set below)
    n_parallel = GPU count   (set USE_GPU = True below)
"""

import os

# ── user settings ─────────────────────────────────────────────────────────────
FMAX    = 0.05    # force convergence criterion (eV/Å)
MAXSTEP = 300     # max relaxation steps
USE_GPU = False   # True: use CUDA GPU
# ──────────────────────────────────────────────────────────────────────────────

_model = None   # loaded once per worker process


def _get_model():
    global _model
    if _model is not None:
        return _model
    import torch
    torch.set_num_threads(1)
    from chgnet.model import CHGNet
    _model = CHGNet.load()
    if USE_GPU and torch.cuda.is_available():
        _model = _model.cuda()
    return _model


def run(work_dir: str) -> float:
    """
    Relax the structure in work_dir with CHGNet and return total energy (eV).

    GmAte writes POSCAR to work_dir before calling this function.
    This function writes:
        CONTCAR     -- relaxed structure (VASP format)
        energy      -- total energy in eV (one line)
        log.chgnet  -- relaxation log
    """
    from chgnet.model.dynamics import StructOptimizer

    poscar = os.path.join(work_dir, "POSCAR")
    log    = os.path.join(work_dir, "log.chgnet")

    relaxer = StructOptimizer(
        model=_get_model(),
        use_device="cuda" if USE_GPU else "cpu",
    )

    # StructOptimizer accepts pymatgen Structure or ASE Atoms.
    # Passing the POSCAR path string works via pymatgen.
    from pymatgen.core import Structure
    structure = Structure.from_file(poscar)

    result = relaxer.relax(structure, fmax=FMAX, steps=MAXSTEP, logfile=log)

    final_structure = result["final_structure"]
    energy = float(result["trajectory"].energies[-1])

    final_structure.to(filename=os.path.join(work_dir, "CONTCAR"), fmt="poscar")
    with open(os.path.join(work_dir, "energy"), "w") as f:
        f.write(f"{energy}\n")

    return energy
