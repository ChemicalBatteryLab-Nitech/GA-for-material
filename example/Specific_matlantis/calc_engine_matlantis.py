#!/usr/bin/env python
# coding: utf-8
"""
calc_engine_matlantis.py  --  Matlantis / LightPFP engine for GmAte

Setup
-----
    pip install pfp-api-client matlantis-pfp

Usage
-----
    cp example/Specific_matlantis/* Specific/
    # Edit Specific/inp_POSCAR.py (ions, ELEM)
    # Edit user settings below if needed
    python GmAte.py -ga

n_parallel (inp_ga.py)
----------------------
    IMPORTANT: set n_parallel = 1
    The Matlantis API token allows only one concurrent session per account.
"""

import os

# ── user settings ─────────────────────────────────────────────────────────────
FMAX       = 0.05     # force convergence criterion (eV/Å)
MAXSTEP    = 300      # max optimizer steps
RELAX_CELL = True     # True: relax cell + atoms;  False: atoms only
# ──────────────────────────────────────────────────────────────────────────────

_calculator = None   # loaded once per worker process


def _get_calculator():
    global _calculator
    if _calculator is not None:
        return _calculator
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
    estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0)
    _calculator = ASECalculator(estimator)
    return _calculator


def run(work_dir: str) -> float:
    """
    Relax the structure in work_dir with Matlantis and return total energy (eV).

    GmAte writes POSCAR to work_dir before calling this function.
    This function writes:
        CONTCAR         -- relaxed structure (VASP format)
        energy          -- total energy in eV (one line)
        log.matlantis   -- optimizer log
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

    opt = LBFGS(runner, logfile=os.path.join(work_dir, "log.matlantis"),
                trajectory=None)
    opt.run(fmax=FMAX, steps=MAXSTEP)

    energy = float(atoms.get_potential_energy())

    write(os.path.join(work_dir, "CONTCAR"), atoms, format="vasp", direct=True)
    with open(os.path.join(work_dir, "energy"), "w") as f:
        f.write(f"{energy}\n")

    return energy
