POPULATION = 24 #population par generation

NUM_OF_STRINGS = 2 #num of genes

MAX_GENERATION = 1000 #number of generations

SAVE = 3 #Number of the most stable genes that are unconditionally transmitted to the next generation

SURVIVAL_RATE = 0.6 #Percentage of individuals allowed to survive each generation

CR_2PT_RATE = 0.4 #Percentage of 2-point crossing

CR_UNI_RATE = 0.4 #Percentage of uniform crossing

CR_UNI_PB = 0.5 #Probability of occurrence of uniform crossover

MUTATION_PB = 0.02 #Probability of occurrence of mutation

STOP_CRITERIA = 100 #Stop condition when best does not change continuously

RESTART = False #if you want to restart =True

ELEMENT_FIX = True #if you want to fix the num of elements =True



select_mode = "ranking" #tournament, roulet, ranking


temp_gene = "temp_gene" #name of the temp gene file

eval_file = "energy"

njob = 6

