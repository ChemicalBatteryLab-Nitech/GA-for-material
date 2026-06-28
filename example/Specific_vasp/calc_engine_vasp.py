#!/usr/bin/env python
# coding: utf-8
"""
calc_engine_vasp.py  --  VASP first-principles engine for GmAte

Requirements
------------
- VASP binary accessible via the command set in VASP_CMD below
- INCAR, KPOINTS, POTCAR in your Specific/ folder (POTCAR not provided here)

Copy this file, example/Specific_vasp/inp_POSCAR.py, INCAR, and KPOINTS
to your Specific/ folder, then:
  1. Place your POTCAR there as well.
  2. Set calc_engine = "calc_engine_vasp.py" in Specific/inp_POSCAR.py.

n_parallel guideline (inp_ga.py)
---------------------------------
For HPC/cluster use with a job scheduler:
    total cores = n_parallel × NCORE_PER_JOB
    where NCORE_PER_JOB is the MPI rank count used in VASP_CMD.

For a single workstation:
    Set n_parallel = 1 and use all cores inside a single VASP run.

temp_gene format (located at work_dir/temp_gene)
-------------------------------------------------
One line per ELEM group (NUM_OF_STRINGS lines total).
Each character maps to the corresponding element in ELEM (inp_POSCAR.py).

    ELEM = [["Li", "Al"]]
    temp_gene: "00110011"   → Li Li Al Al Li Li Al Al

GmAte writes POSCAR (from temp_gene) before calling run().
"""

import os
import shutil
import subprocess

# --- user settings -----------------------------------------------------------
# MPI launcher command examples (choose one or edit to match your environment):
#   VASP_CMD = "mpirun -np 8 vasp_std"          # OpenMPI
#   VASP_CMD = "mpiexec -n 8 vasp_std"          # MPICH / Intel MPI
#   VASP_CMD = "srun --ntasks=8 vasp_std"       # SLURM
#   VASP_CMD = "vasp_std"                        # single-core (no MPI)
VASP_CMD = "mpirun -np 8 vasp_std"   # ← edit

SPECIFIC_DIR = None    # None → auto-detect as ./Specific relative to project root
TIMEOUT_SEC = 30 * 60  # abort if VASP takes longer (seconds)
# -----------------------------------------------------------------------------


def _specific_dir(work_dir: str) -> str:
    if SPECIFIC_DIR is not None:
        return SPECIFIC_DIR
    # walk up from work_dir (sample001/) to project root, then into Specific/
    project_root = os.path.dirname(os.path.dirname(work_dir))
    return os.path.join(project_root, "Specific")


def run(work_dir: str) -> float:
    """Run VASP in work_dir and return E0 from OSZICAR (eV)."""
    spec = _specific_dir(work_dir)

    # Copy VASP input files from Specific/ (POSCAR is already in work_dir)
    for fname in ("INCAR", "KPOINTS", "POTCAR"):
        src = os.path.join(spec, fname)
        if os.path.isfile(src):
            shutil.copy(src, work_dir)

    # Clean up leftovers that can cause VASP to skip calculation
    for f in ("CHGCAR", "WAVECAR", "IBZKPT"):
        p = os.path.join(work_dir, f)
        if os.path.isfile(p):
            os.remove(p)

    proc = subprocess.run(
        VASP_CMD.split(),
        cwd=work_dir,
        timeout=TIMEOUT_SEC,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"VASP failed in {work_dir}\n"
            f"stderr: {proc.stderr.decode()[-500:]}"
        )

    energy = _read_oszicar_e0(os.path.join(work_dir, "OSZICAR"))

    with open(os.path.join(work_dir, "energy"), "w") as f:
        f.write(f"{energy}\n")

    return energy


def _read_oszicar_e0(oszicar_path: str) -> float:
    """Extract E0 from the last ionic step in OSZICAR."""
    with open(oszicar_path, "r") as f:
        lines = [l for l in f if l.strip()]
    # OSZICAR format: "N  E  dE  d eps  ncg  rms  E0"
    # Ionic-step lines start with an integer (step number)
    ionic = [l for l in lines if l.split()[0].isdigit()]
    if not ionic:
        raise ValueError(f"No ionic step found in {oszicar_path}")
    return float(ionic[-1].split()[6])  # E0 is the 7th field (index 6)
