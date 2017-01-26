from dataGenerator import *
# a python script for testing the algorithm generated a ranking with input mixing proportion
# change the corresponding parameter to output your results


# define the input population with size of user and protected group
user_N = 100
pro_N = 50
# define the input mixing proportion of generating algorithm
mixing_proportion = 0.5
# based on input population, generate a random ranking and index of protected group as input of generating function
test_ranking = [x for x in range(user_N)]
pro_index = [x for x in range(pro_N)]
# run the algorithm to generate a ranking
output_ranking = generateUnfairRanking(test_ranking, pro_index, mixing_proportion)
print "Successfully generate a ranking with input setting!"
print output_ranking


