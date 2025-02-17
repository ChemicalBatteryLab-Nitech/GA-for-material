import warnings
import numpy as np
from pymatgen.ext.matproj import MPRester
from pymatgen.util.coord import pbc_diff
from pymatgen.core import Lattice, Structure, Molecule

 
import os
import ase
from ase.io import read, write
from ase.io import Trajectory
from ase.constraints import FixAtoms, FixedPlane, FixBondLength, UnitCellFilter, ExpCellFilter
from ase.optimize import LBFGS, BFGS, FIRE
from ase import Atoms
from ase import units


from m3gnet.models import M3GNet, M3GNetCalculator, Potential
from m3gnet.models import Relaxer

from ase.optimize import LBFGS, BFGS, FIRE
 
def main(input):

    global potver, outpos, outcif, infmax, static, maxstep, outenergy, outallenergy, mode, isif, algo, inpf
    potver=2.0
    outpos="CONTCAR"
    outcif="CONTCAR.cif"
    infmax=0.001
    static="F"
    maxstep=500
    outenergy="energy"
    outallenergy="log.energies"
    mode="relax"
    isif=3
    algo="LBFGS"
    inpf = "POSCAR"

    for i in input:
        print(i[0], i[1])
        globals()[i[0]] = i[1]

    if mode == "static":
        static = "T"

    with open ("log.m3g_input",'w') as f:
        print ("mode=",mode,file=f)
        print ("instruct=",inpf,file=f)
        print ("fmax=",infmax,file=f)
        print ("static=",static,file=f)
        print ("outpos=",outpos,file=f)
        print ("outcif=",outcif,file=f)
        print ("stepmax=",maxstep,file=f)
        print ("outenergy=",outenergy,file=f)
        print ("isif=",isif,file=f)
        print ("algo=",algo,file=f)


    print("input options:")
    fl=open ("log.m3g_input",'r')
    for flc in fl:
        print ("  ",flc.strip())

    try:
        os.system(f"rm {outallenergy}")
    except:
        pass
    # os.system("cp "+inpf+" POSCAR_in")
    poscar = Structure.from_file("POSCAR")
    atoms = read("POSCAR")

    if mode=="static":
        potential = Potential(M3GNet.load())
        calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)

        atoms.set_constraint()
        atoms.set_calculator(calculator)

        maxf = np.sqrt(((atoms.get_forces())**2).sum(axis=1).max())

        print("************* calculation starts *************")
        print("energy:{:.10f},  maxforce:{:.10f}".format(float(atoms.get_potential_energy()),float(maxf)))


        final_energy=atoms.get_potential_energy()
        fepa=float(final_energy.item()/len(atoms))

        with open (outenergy,'w') as f:
            print(final_energy.item(), file=f)

        with open (outenergy+"_per_atom",'w') as f:
            print (fepa, file=f)
 

    elif mode=="relax":
        potential = Potential(M3GNet.load())
        calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
        atoms.set_constraint()
        atoms.set_calculator(calculator)
        if isif == 3:
            ucf=ExpCellFilter(atoms)
            #ucf=UnitCellFilter(atoms)
        elif isif == 2:
            ucf=atoms
        maxf = np.sqrt(((atoms.get_forces())**2).sum(axis=1).max()) 

        print("************* calculation starts: structure relaxation  *************")
        print("ini   pot:{:.4f},maxforce:{:.4f}".format(float(atoms.get_potential_energy()),float(maxf)))

        if algo=="LBFGS":
            opt = LBFGS(ucf, trajectory="relax.traj",logfile=outallenergy)
        elif algo=="BFGS":
            opt = BFGS(ucf, trajectory="relax.traj",logfile=outallenergy)
        elif algo=="FIRE":
            opt = FIRE(ucf, trajectory="relax.traj",logfile=outallenergy)

        opt.run(fmax=infmax,steps=maxstep)

        if isif == 3:
            opt=opt.atoms

        final_energy = opt.get_potential_energy()

        ase.io.write(outpos, opt.atoms, format="vasp", direct=True)
        ase.io.write(outcif, opt.atoms, format="cif")

        fepa=float(final_energy.item()/len(atoms))

        with open (outenergy,'w') as f:
            print(final_energy.item(), file=f)

        with open (outenergy+"_per_atom",'w') as f:
            print (fepa, file=f)

        maxf = np.sqrt(((opt.atoms.get_forces())**2).sum(axis=1).max())
        print("fin   pot:{:.4f},maxforce:{:.4f}".format(float(opt.atoms.get_potential_energy()),float(maxf)))