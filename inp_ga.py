"""
for Genetic Algolithm

"""
POPULATION = 24 #population par generation
NUM_OF_STRINGS = 1 #num of genes
MAX_GENERATION = 500 #number of generations
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
n_parallel = 6  # number of calculation processes to run concurrently


"""
for GA ML
"""
mlga = False #T:use ML, F:not use ML
mlopt1 = 6 #unicross the number of from gstring: ML = populations*ratio - mlopt
mlopt2 = 6 #twocross
mlopt3 = 3 #mutation
mlext1 = 200 #unicross ML candidate
mlext2 = 200 #twocross ML candidate
mlext3 = 200 #mutation ML candidate

outMLpred = "sort_label.out" #log: ML-selected gene strings (only when save_ml_log=True)
outMLreg = "test_rmse.out"  #log: training RMSE per generation (only when save_ml_log=True)
save_ml_log = True #True: write outMLpred and outMLreg; False: memory-only

MLoriginal = "SpecificML"
