# Calculation engine file name (located in the Specific/ directory).
calc_engine = "calc_engine_mace.py"

# All element symbols present in the structure (including non-optimized sites).
ions = [["Li", "Al", "O"]]   # ← edit

# Candidate elements for each optimized site group (ELEM1, ELEM2, ...).
# [[elements at ELEM1 sites], [elements at ELEM2 sites], ...]
ELEM = [["Li", "Al"]]        # ← edit

# Files to archive after each evaluation (CONTCAR = relaxed structure).
savefiles = ["POSCAR", "CONTCAR", "temp_gene"]

# Name of the file where calc_engine writes the energy value.
# Used only by make_savefiles(); the actual score is the return value of run().
output = "energy"
