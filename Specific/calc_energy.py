import os
import sys
import inp_POSCAR
import time
import glob

ref_poscar = "POSCAR_org"
ELEM = inp_POSCAR.ELEM
ions = inp_POSCAR.ions
runtype = inp_POSCAR.runtype
savefiles = inp_POSCAR.savefiles
output = inp_POSCAR.output
cpwd = os.getcwd()
cri_time = 30*60


if runtype == "m3g":
    calccode = "m3g.py instruct=POSCAR algo=FIRE 2>errorlog.m3g >log.m3g &"
    energy_file = "energy"
elif runtype == "m.py":
    calccode = "m.py -instruct=POSCAR"
    energy_file = "energy"
elif runtype == "vasp":
    energy_file = output
    vasp_code = "~/bin/vasp5404_gam_flowcl"  #Command to run vasp　#Please make changes according to your environment.
else:
    pass

def gene_to_POSCAR():
    with open(f"{cpwd}/temp_gene", "r") as f:
        strings = f.read().splitlines()
    with open(ref_poscar, "r") as f:
        all_lines = f.readlines()
    all_lines = [i.split() for i in all_lines]
    labels = all_lines[:8]
    lines = all_lines[8:]
    if "Direct" not in labels[-1] and "Cart" not in labels[-1]:
            print("Warning : POSCAR_org must be in VASP5 format.　A label with the name of the atom is required on line 6.")
            sys.exit()

    new_labels = labels.copy()
    label_info = []
    # print(labels[5])
    for i in range(len(labels[5])):
        if "ELEM" in labels[5][i]:
            flag = labels[5][i].lstrip("ELEM")
            if flag == "":
                flag = 1
            else:
                flag = int(flag)
            for xx, x in enumerate(ELEM[flag-1]):
                count = str(str(strings[flag-1]).count(str(xx)))
                if x != "Vac":
                    label_info.append([x, count])
        else:
            label_info.append([labels[5][i], labels[6][i]])
    label_info_sort = sorted(label_info, key=lambda x: str(x[0]))
    new_labels[5] = [i[0] for i in label_info_sort]
    new_labels[6] = [i[1] for i in label_info_sort]

    count_list = [0 for _ in range(len(strings))]
    for line in lines:
        if "ELEM" in line[3]:
            flag = line[3].lstrip("ELEM")
            if flag == "":
                flag = 1
            else:
                flag = int(flag)
            gen = int(strings[flag-1][count_list[flag-1]])
            count_list[flag-1] += 1
            line[3] = ELEM[flag-1][gen]
        else:
            pass
    lines_sort = sorted(lines, key= lambda x: str(x[3]))

    with open(f"{cpwd}/POSCAR", "w") as w:
        for i in new_labels:
            w.write(" ".join(i) + "\n")
        for i in lines_sort:
            if "Vac" not in i:
                w.write(" ".join(i) + "\n")

def calc_score():
    #calculate the score
    if os.path.isfile("wait") == False:
        print("no wait")
        sys.exit()
    os.system(f"mv {cpwd}/wait {cpwd}/running")
    gene_to_POSCAR()
    print("start calc")
    start_time = time.time()
    #test_score() #for test
    if runtype != "vasp":
        os.system(f"{calccode}")    
    else:
        os.system("rm CHG* WAV* IBZ* finish 2>err")
        os.system(vasp_code)
        os.system('vaspfiles -getenergy OUTCAR')
        #os.system('cp energy out.energy')
        os.system("tail OSZICAR -n1 > oszitail")
        with open("oszitail", "r") as f:
            E0 = f.readline().split()[4]
        os.system("rm oszitail")
        with open(output, "w") as w:
            w.write(E0)
        time.sleep(1)

    while True:
        if os.path.isfile(f"{cpwd}/{energy_file}") == True:
            print("calc finish")
            make_savefiles()
            os.system(f"mv {cpwd}/running {cpwd}/finish")
            os.system('vaspfiles -getenergy OUTCAR')
            break
        elif time.time() - start_time > cri_time:
            score = 0
            try:
                with open("log.m3g", "r") as f:
                    lines = f.read().splitlines()
                score = float(lines[-2].split()[1].split(",")[0].split(":")[1])
            except:
                pass
            if type(score) != int and type(score) != float:
                score = 0
            with open(f"{cpwd}/{energy_file}", "w") as w:
                w.write(f"{str(score)}\n")
            make_savefiles()
            os.system('vaspfiles -getenergy OUTCAR')
            os.system(f"rm {cpwd}/running")
            os.system(f"touch {cpwd}/finish")
            break
        else:
            pass    

def test_score():
    with open(f"{cpwd}/temp_gene", "r") as f:
        strings = f.read().splitlines()
    sc = 0
    time.sleep(1)
    for x in range(0, len(strings)):
        for i in range(0, len(strings[x])):
            sc += int(strings[x][i])*(i+1)
    sc = -sc
    with open(f"{cpwd}/{energy_file}", "w") as w:
        w.write(f"{str(sc)}\n")

def make_savefiles():
    with open(f"{cpwd}/flag", "r") as f:
        generation = f.readline().rstrip()
    for i in savefiles:
        if i == "log.m3g":
            try:
                os.system(f"mv {cpwd}/{i} {cpwd}/Save_{i}_{generation}")
            except:
                pass
        else:
            try:
                os.system(f"cp {cpwd}/{i} {cpwd}/Save_{i}_{generation}")  
            except:
                pass
    os.system(f"cp {cpwd}/{output} {cpwd}/Save_{output}_{generation}")

try:
    mode = sys.argv[1]
except:
    mode = "ga"

if __name__ == "__main__":
    if mode == "-gene2pos":
        gene_to_POSCAR()
    else:            
        calc_score()



