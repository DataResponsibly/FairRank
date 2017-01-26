import measures 
import numpy as np
# a python script for testing fairness and accuracy measures 
# change the corresponding parameter to output your results

# constant for three fairness measures
KL_DIVERGENCE="rKL" # represent kl-divergence group fairness measure
ND_DIFFERENCE="rND" # represent normalized difference group fairness measure
RD_DIFFERENCE="rRD" # represent ratio difference group fairness measure


# define the input population with size of user and protected group
user_N = 100
pro_N = 50

# test fairness measures
# normalized fairness follow here  
# if this input population has been computed, then get from recorded maximum (stored in normalizer.txt)      
# else compute the normalizer of input population
max_rKL=measures.getNormalizer(user_N,pro_N,KL_DIVERGENCE) 
max_rND=measures.getNormalizer(user_N,pro_N,ND_DIFFERENCE)
max_rRD=measures.getNormalizer(user_N,pro_N,RD_DIFFERENCE)    

# # non-normalized fairness follow here 
# max_rKL= 1
# max_rND= 1
# max_rRD= 1
print "Finished fairness normalizer calculation!"

# based on input population, generate a random ranking and index of protected group as input of fairness measures
test_ranking = [x for x in range(user_N)]
pro_index = [x for x in range(pro_N)]

# define the cut point using in the computation of fairness measures
cut_point = 10

fair_rKL=measures.calculateNDFairness(test_ranking,pro_index,cut_point,KL_DIVERGENCE,max_rKL)
fair_rND=measures.calculateNDFairness(test_ranking,pro_index,cut_point,ND_DIFFERENCE,max_rND)
fair_rRD=measures.calculateNDFairness(test_ranking,pro_index,cut_point,RD_DIFFERENCE,max_rRD)

print "rKL of test ranking is ", str(fair_rKL)
print "rND of test ranking is ", str(fair_rND)
print "rKL of test ranking is ", str(fair_rRD)

# test accuracy measures
# define the ground truth test ranking for accuracy measures
ground_truth_scores = list(np.random.permutation(user_N))
estimate_scores = list(np.random.permutation(user_N))

# prepare different input for different accuracy measures   

# generate permutations of two score lists returned permutation of sorted id 
per_scores_input=sorted(range(len(ground_truth_scores)), key=lambda k: ground_truth_scores[k],reverse=True)
per_scores_hat=sorted(range(len(estimate_scores)), key=lambda k: estimate_scores[k],reverse=True)
# sort two scores list in descending order for computing score difference
sorted_score_hat = estimate_scores   
sorted_score_hat.sort(reverse=True)
sorted_inputscores = ground_truth_scores   
sorted_inputscores.sort(reverse=True)

# score difference   
acc_scoreDiff=measures.calculateScoreDifference(sorted_score_hat,sorted_inputscores) 
# position difference     
acc_posDiff=measures.calculatePositionDifference(per_scores_hat,per_scores_input) 
# kendall distance   
acc_kendallDis=measures.calculateKendallDistance(per_scores_hat,per_scores_input)      
# for spearman and pearson relation, use the negative value to minimize during optimization
# spearman distance  
acc_spearmanDis=measures.calculateSpearmanR(estimate_scores,ground_truth_scores) 
# pearson correlation 
acc_pearsonDis=measures.calculatePearsonC(estimate_scores,ground_truth_scores)

print "score difference between input two scores lists is ", str(acc_scoreDiff)
print "position difference between input two permutations is ", str(acc_posDiff)
print "kendall distance between input two permutations is ", str(acc_kendallDis)
print "spearman distance between input two scores lists is ", str(acc_spearmanDis)
print "pearson correlation between input two scores lists is ", str(acc_pearsonDis)