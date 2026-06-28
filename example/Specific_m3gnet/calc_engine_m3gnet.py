#!/usr/bin/env python
# coding: utf-8
"""
calc_engine_m3gnet.py  --  M3GNet universal NNP engine for GmAte

Setup
-----
    pip install matgl          # recommended (newer API)
    # or: pip install m3gnet   # legacy API (auto-detected)

Usage
-----
    cp example/Specific_m3gnet/* Specific/
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
FMAX       = 0.05     # force convergence criterion (eV/Å)
MAXSTEP    = 500      # max optimizer steps
ALGO       = "LBFGS"  # "LBFGS", "BFGS", or "FIRE"
RELAX_CELL = False    # True: relax cell + atoms (ISIF=3);  False: atoms only
USE_GPU    = False    # True: use CUDA GPU
# ──────────────────────────────────────────────────────────────────────────────

_calculator = None   # loaded once per worker process


def _get_calculator():
    global _calculator
    if _calculator is not None:
        return _calculator
    import torch
    torch.set_num_threads(1)
    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    try:
        # matgl (recommended)
        import matgl
        from matgl.ext.ase import PESCalculator
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        _calculator = PESCalculator(pot)
    except Exception:
        # legacy m3gnet
        from m3gnet.models import M3GNet, Potential, M3GNetCalculator
        pot = Potential(M3GNet.load())
        _calculator = M3GNetCalculator(potential=pot)
    return _calculator


def run(work_dir: str) -> float:
    """
    Relax the structure in work_dir with M3GNet and return total energy (eV).

    GmAte writes POSCAR to work_dir before calling this function.
    This function writes:
        CONTCAR  -- relaxed structure (VASP format)
        energy   -- total energy in eV (one line)
        log.m3g  -- optimizer log
    """
    from ase.io import read, write
    from ase.optimize import LBFGS, BFGS, FIRE

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

    _opt_cls = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}.get(ALGO, LBFGS)
    opt = _opt_cls(runner, logfile=os.path.join(work_dir, "log.m3g"),
                   trajectory=None)
    opt.run(fmax=FMAX, steps=MAXSTEP)

    energy = float(atoms.get_potential_energy())

    write(os.path.join(work_dir, "CONTCAR"), atoms, format="vasp", direct=True)
    with open(os.path.join(work_dir, "energy"), "w") as f:
        f.write(f"{energy}\n")

    return energy
