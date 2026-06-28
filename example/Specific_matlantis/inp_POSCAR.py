# Calculation engine file name (located in the Specific/ directory).
calc_engine = "calc_engine_matlantis.py"

# All element symbols present in the structure (including non-optimized sites).
ions = [["Li", "Al", "O"]]   # ← edit

# Candidate elements for each optimized site group (ELEM1, ELEM2, ...).
ELEM = [["Li", "Al"]]        # ← edit

# Files to archive after each evaluation.
savefiles = ["POSCAR", "CONTCAR", "temp_gene"]

# Name of the file where calc_engine writes the energy value (for archiving).
output = "energy"

# NOTE: Set n_parallel = 1 in inp_ga.py.
# The Matlantis API token allows only one concurrent session per account.
