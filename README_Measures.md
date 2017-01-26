
# Usage of measures

### Import all functions in measures


```python
from measures import *
import numpy as np
```

#### Initialization

###### Specify some constant used in measures
- KL_DIVERGENCE represents kl-divergence difference fairness measure
- ND_DIFFERENCE represents normalized difference fairness measure
- RD_DIFFERENCE represents ratio difference fairness measure


```python
KL_DIVERGENCE = "rKL"
ND_DIFFERENCE = "rND"
RD_DIFFERENCE = "rRD"
```

##### Step 1: Specify the input population with size of user and protected group


```python
user_N = 100 
pro_N = 50
```

##### Step 2: Compute the normalizor of above input population


```python
# normalized fairness follow here  
# if this input population has been computed, then get from recorded maximum (stored in normalizer.txt)      
# else compute the normalizer of input population
max_rKL = getNormalizer(user_N,pro_N,KL_DIVERGENCE)  
max_rND = getNormalizer(user_N,pro_N,ND_DIFFERENCE)
max_rRD = getNormalizer(user_N,pro_N,RD_DIFFERENCE)
```

    Initialized a dataset description generator.
    

###### if want to skip the normalizor computation, execute the following cell.


```python
# non-normalized fairness follow here 
max_rKL = 1
max_rND = 1
max_rRD = 1
```

##### Step 3: Define the cut point of computation of fairness measures 


```python
cut_point = 10
```

#### Test three fairness meaures

##### Step 1: Generate a test ranking and related position of protected group


```python
test_ranking = [x for x in range(user_N)]
pro_index = [x for x in range(pro_N)]
```

##### Step 2: Compute three fairness measures for above test ranking


```python
fair_rKL = calculateNDFairness(test_ranking,pro_index,cut_point,KL_DIVERGENCE,max_rKL)
fair_rND = calculateNDFairness(test_ranking,pro_index,cut_point,ND_DIFFERENCE,max_rND)
fair_rRD = calculateNDFairness(test_ranking,pro_index,cut_point,RD_DIFFERENCE,max_rRD)

print "rKL of test ranking is ", str(fair_rKL)
print "rND of test ranking is ", str(fair_rND)
print "rKL of test ranking is ", str(fair_rRD)
```

    rKL of test ranking is  1.15901724032
    rND of test ranking is  0.660066334446
    rKL of test ranking is  1.08152800722
    

#### Test five accuracy meaures

##### Step 1: Generate a test score list and a ground truth score list


```python
ground_truth_scores = list(np.random.permutation(user_N))
estimate_scores = list(np.random.permutation(user_N))
```

##### Step 2: Rank the above score lists for accuracy computation


```python
# generate permutations of two score lists returned permutation of sorted id 
per_scores_input=sorted(range(len(ground_truth_scores)), key=lambda k: ground_truth_scores[k],reverse=True)
per_scores_hat=sorted(range(len(estimate_scores)), key=lambda k: estimate_scores[k],reverse=True)
# sort two scores list in descending order for computing score difference
sorted_score_hat = estimate_scores   
sorted_score_hat.sort(reverse=True)
sorted_inputscores = ground_truth_scores   
sorted_inputscores.sort(reverse=True)
```

##### Step 3: Compute the five accuracy measures for above ranked score lists


```python
# score difference   
acc_scoreDiff = calculateScoreDifference(sorted_score_hat,sorted_inputscores) 
# position difference     
acc_posDiff = calculatePositionDifference(per_scores_hat,per_scores_input) 
# kendall distance   
acc_kendallDis = calculateKendallDistance(per_scores_hat,per_scores_input)      
# for spearman and pearson relation, use the negative value to minimize during optimization
# spearman distance  
acc_spearmanDis = calculateSpearmanR(estimate_scores,ground_truth_scores) 
# pearson correlation 
acc_pearsonDis = calculatePearsonC(estimate_scores,ground_truth_scores)

print "score difference between input two scores lists is ", str(acc_scoreDiff)
print "position difference between input two permutations is ", str(acc_posDiff)
print "kendall distance between input two permutations is ", str(acc_kendallDis)
print "spearman distance between input two scores lists is ", str(acc_spearmanDis)
print "pearson correlation between input two scores lists is ", str(acc_pearsonDis)
```

    score difference between input two scores lists is  0.0
    position difference between input two permutations is  0.698
    kendall distance between input two permutations is  0.530101010101
    spearman distance between input two scores lists is  1.0
    pearson correlation between input two scores lists is  1.0
    
