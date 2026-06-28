# Calculation engine file name (located in the Specific/ directory).
calc_engine = "calc_engine_vasp.py"

# All element symbols present in the structure (including non-optimized sites).
ions = [["Li", "Al", "O"]]   # ← edit

# Candidate elements for each optimized site group (ELEM1, ELEM2, ...).
ELEM = [["Li", "Al"]]        # ← edit

# Files to archive after each evaluation.
# VASP produces CONTCAR (relaxed structure) and OSZICAR/OUTCAR (energy).
savefiles = ["POSCAR", "CONTCAR", "OSZICAR", "temp_gene"]

# Name of the file where calc_engine writes the energy value (for archiving).
output = "energy"
