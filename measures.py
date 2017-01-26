from __future__ import division
import numpy as np
import math
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import dataGenerator
# a python script define computation of fairness measures and accuracy measures
# test of this script can be found in testMeasures.py

KL_DIVERGENCE="rKL" # represent kl-divergence group fairness measure
ND_DIFFERENCE="rND" # represent normalized difference group fairness measure
RD_DIFFERENCE="rRD" # represent ratio difference group fairness measure
LOG_BASE=2 # log base used in logorithm function

NORM_CUTPOINT=10 # cut-off point used in normalizer computation
NORM_ITERATION=10 # max iterations used in normalizer computation
NORM_FILE="normalizer.txt" # externally text file for normalizers


def calculateNDFairness(_ranking,_protected_group,_cut_point,_gf_measure,_normalizer):
    """
        Calculate group fairness value of the whole ranking.
        Calls function 'calculateFairness' in the calculation.

        :param _ranking: A permutation of N numbers (0..N-1) that represents a ranking of N individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
        :param _protected_group: A set of identifiers from _ranking that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _cut_point: Cut range for the calculation of group fairness, e.g., 10, 20, 30,...
        :param _gf_measure:  Group fairness measure to be used in the calculation, 
                            one of 'rKL', 'rND', 'rRD'.
        :param _normalizer: The normalizer of the input _gf_measure that is computed externally for efficiency.
        :return: returns  fairness value of _ranking, a float, normalized to [0, 1]
    """
    
    # error handling for ranking and protected group
    dataGenerator.completeCheckRankingProperties(_ranking,_protected_group)
    # error handling for input type 
    if not isinstance( _cut_point, ( int, long ) ):
        raise TypeError("Input batch size must be an integer larger than 0")
    if not isinstance( _normalizer, (int, long, float, complex) ):
        raise TypeError("Input normalizer must be a number larger than 0")
    if not isinstance( _gf_measure, str ):
        raise TypeError("Input group fairness measure must be a string that choose from ['rKL', 'rND', 'rRD']")
    
    user_N=len(_ranking)
    pro_N=len(_protected_group)

    # error handling for input value
    if NORM_CUTPOINT > user_N:
        raise ValueError("Batch size should be less than input ranking's length")
    
    discounted_gf=0 #initialize the returned gf value
    for countni in range(user_N):
        countni=countni+1
        if(countni%_cut_point ==0):
            ranking_cutpoint=_ranking[0:countni]
            pro_cutpoint=set(ranking_cutpoint).intersection(_protected_group)

            gf=calculateFairness(ranking_cutpoint,pro_cutpoint,user_N,pro_N,_gf_measure)
            discounted_gf+=gf/math.log(countni+1,LOG_BASE) # log base -> global variable
            
            # make a call to compute, or look up, the normalizer; make sure to check that it's not 0!
            # generally, think about error handling

    if _normalizer==0:
        raise ValueError("Normalizer equals to zero")
    return discounted_gf/_normalizer



def calculateFairness(_ranking,_protected_group,_user_N,_pro_N,_gf_measure):
    """
        Calculate the group fairness value of input ranking.
        Called by function 'calculateNDFairness'.

        :param _ranking: A permutation of N numbers (0..N-1) that represents a ranking of N individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _protected_group: A set of identifiers from _ranking that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group
        :param _gf_measure: The group fairness measure to be used in calculation        
        :return: returns the value of selected group fairness measure of this input ranking
    """
      
    ranking_k=len(_ranking)
    pro_k=len(_protected_group)
    if _gf_measure==KL_DIVERGENCE: #for KL-divergence difference
        gf=calculaterKL(ranking_k,pro_k,_user_N,_pro_N)        
        
    elif _gf_measure==ND_DIFFERENCE:#for normalized difference
        gf=calculaterND(ranking_k,pro_k,_user_N,_pro_N)

    elif _gf_measure==RD_DIFFERENCE: #for ratio difference
        gf=calculaterRD(ranking_k,pro_k,_user_N,_pro_N)     

    return gf 

def calculaterKL(_ranking_k,_pro_k,_user_N,_pro_N):
    """
        Calculate the KL-divergence difference of input ranking        
        :param _ranking_k: A permutation of k numbers that represents a ranking of k individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _pro_k: A set of identifiers from _ranking_k that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group                
        :return: returns the value of KL-divergence difference of this input ranking
    """
    px=_pro_k/(_ranking_k)
    qx=_pro_N/_user_N
    if px==0 or px ==1: # manually set the value of extreme case to avoid error of math.log function 
        px=0.001
    if qx == 0 or qx ==1:
        qx=0.001
    return (px*math.log(px/qx,LOG_BASE)+(1-px)*math.log((1-px)/(1-qx),LOG_BASE))

def calculaterND(_ranking_k,_pro_k,_user_N,_pro_N):
    """
        Calculate the normalized difference of input ranking        
        :param _ranking_k: A permutation of k numbers that represents a ranking of k individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _pro_k: A set of identifiers from _ranking_k that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group                
        :return: returns the value of normalized difference of this input ranking
    """
    return abs(_pro_k/_ranking_k-_pro_N/_user_N)

def calculaterRD(_ranking_k,_pro_k,_user_N,_pro_N):
    """
        Calculate the ratio difference of input ranking        
        :param _ranking_k: A permutation of k numbers that represents a ranking of k individuals, 
                                e.g., [0, 3, 5, 2, 1, 4].  Each number is an identifier of an individual.
                                Stored as a python array.
                                Can be a total ranking of input data or a partial ranking of input data.
        :param _pro_k: A set of identifiers from _ranking_k that represent members of the protected group
                                e.g., [0, 2, 3].  Stored as a python array for convenience, order does not matter.
        :param _user_N: The size of input items 
        :param _pro_N: The size of input protected group                
        :return: returns the value of ratio difference of this input ranking
        # This version of rRD is consistent with poster of FATML instead of arXiv submission.
    """
    input_ratio=_pro_N/(_user_N-_pro_N)
    unpro_k=_ranking_k-_pro_k
    
    if unpro_k==0: # manually set the case of denominator equals zero
        current_ratio=0
    else:
        current_ratio=_pro_k/unpro_k

    min_ratio=min(input_ratio,current_ratio)
       
    return abs(min_ratio-input_ratio)

def getNormalizer(_user_N,_pro_N,_gf_measure):
    """
        Retrieve the normalizer of the current setting in external normalizer dictionary.
        If not founded, call function 'calculateNormalizer' to calculate the normalizer of input group fairness measure at current setting.
        Called separately from fairness computation for efficiency.

        :param _user_N: The total user number of input ranking
        :param _pro_N: The size of protected group in the input ranking        
        :param _gf_measure: The group fairness measure to be used in calculation
        
        :return: returns the maximum value of selected group fairness measure in _max_iter iterations
    """
    # read the normalizor dictionary that is computed externally for efficiency
    normalizer_dic=readNormalizerDictionary()

    # error handling for type  
    if not isinstance( _user_N, ( int, long ) ):
        raise TypeError("Input user number must be an integer")
    if not isinstance( _pro_N, ( int, long ) ):
        raise TypeError("Input size of protected group must be an integer")
    if not isinstance( _gf_measure, str ):
        raise TypeError("Input group fairness measure must be a string that choose from ['rKL', 'rND', 'rRD']")
    # error handling for value 
    if _user_N <=0:
        raise ValueError("Input a valud user number")
    if _pro_N <=0:
        raise ValueError("Input a valid protected group size")
    if _pro_N >= _user_N:
        raise ValueError("Input a valid protected group size")


    current_normalizer_key=str(_user_N)+","+str(_pro_N)+","+_gf_measure
    if current_normalizer_key in normalizer_dic.keys():
        normalizer=normalizer_dic[current_normalizer_key]
    else:
        normalizer=calculateNormalizer(_user_N,_pro_N,_gf_measure)           
    return float(normalizer)

def readNormalizerDictionary():
    """
        Retrieve recorded normalizer from external txt file that is computed external for efficiency.
        Normalizer file is a txt file that each row represents the normalizer of a combination of user number and protected group number.
        Has the format like this: user_N,pro_N,_gf_measure:normalizer
        Called by function 'getNormalizer'.

        :param : no parameter needed. The name of normalizer file is constant.     
        :return: returns normalizer dictionary computed externally.
    """
    try:
        with open(NORM_FILE) as f:
            lines = f.readlines()
    except EnvironmentError as e:
        print("Cannot find the normalizer txt file")
    
    
    normalizer_dic={}
    for line in lines:
        normalizer=line.split(":")
        normalizer_dic[normalizer[0]]=normalizer[1]
    return normalizer_dic

def calculateNormalizer(_user_N,_pro_N,_gf_measure):
    """
        Calculate the normalizer of input group fairness measure at input user and protected group setting.
        The function use two constant: NORM_ITERATION AND NORM_CUTPOINT to specify the max iteration and batch size used in the calculation.
        First, get the maximum value of input group fairness measure at different fairness probability.
        Run the above calculation NORM_ITERATION times.
        Then compute the average value of above results as the maximum value of each fairness probability.
        Finally, choose the maximum of value as the normalizer of this group fairness measure.
        
        :param _user_N: The total user number of input ranking
        :param _pro_N: The size of protected group in the input ranking 
        :param _gf_measure: The group fairness measure to be used in calculation 
        
        :return: returns the group fairness value for the unfair ranking generated at input setting

    """
    # set the range of fairness probability based on input group fairness measure
    if _gf_measure==RD_DIFFERENCE: # if the group fairness measure is rRD, then use 0.5 as normalization range
        f_probs=[0,0.5] 
    else:
        f_probs=[0,0.98] 
    avg_maximums=[] #initialize the lists of average results of all iteration
    for fpi in f_probs:
        iter_results=[] #initialize the lists of results of all iteration
        for iteri in range(NORM_ITERATION):
            input_ranking=[x for x in range(_user_N)]
            protected_group=[x for x in range(_pro_N)]
            # generate unfair ranking using algorithm
            unfair_ranking=dataGenerator.generateUnfairRanking(input_ranking,protected_group,fpi)    
            # calculate the non-normalized group fairness value i.e. input normalized value as 1
            gf=calculateNDFairness(unfair_ranking,protected_group,NORM_CUTPOINT,_gf_measure,1)
            iter_results.append(gf)
        avg_maximums.append(np.mean(iter_results))        
    return max(avg_maximums)

def calculateScoreDifference(_scores1,_scores2):
    """
        Calculate the average position-wise score difference
        between two sorted lists.  Lists are sorted in decreasing
        order of scores.  If lists are not sorted by descending- error.

        Only applied for two score lists with same size. 
        # check for no division by 0
        # check that each list is sorted in decreasing order of score

        :param _scores1: The first list of scores
        :param _scores2: The second list of scores         
        :return: returns the average score difference of two input score lists.

    """
    # error handling 
    if not isinstance(_scores1, (list, tuple, np.ndarray)) and not isinstance( _scores1, basestring ):
        raise TypeError("First score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_scores2, (list, tuple, np.ndarray)) and not isinstance( _scores2, basestring ):
        raise TypeError("Second score list must be a list-wise structure defined by '[]' symbol")
    
    if len(_scores1)*len(_scores2) ==0:
        raise ValueError("Input score lists should have length larger than 0")
        
    if not descendingOrderCheck(_scores1):
        raise ValueError("First score list is not ordered by descending order")
    if not descendingOrderCheck(_scores2):
        raise ValueError("Second score list is not ordered by descending order")

    user_N=min(len(_scores1),len(_scores2)) # get the minimum user number of two score lists
    score_diff = 0
    for xi in range(user_N):        
        score_diff+=abs(_scores1[xi]-_scores2[xi])         
    score_diff=score_diff/user_N
    return score_diff

def calculatePositionDifference(_perm1,_perm2):
    """
        Calculate the average position difference for each item, 
        between two permutations of the same items.
        CHECK THAT EACH list is a valid permutation
        CHECK that lists are of the same size

        :param _perm1: The first permutation
        :param _perm2: The second permutation         
        :return: returns the average position difference of two input score lists.

    """
    completePermutaionCheck(_perm1,_perm2)
    user_N=len(_perm1) # get the total user number of two score list

    position_diff = 0
    positions_perm1=[]
    for ui in range(user_N):
        for pi in range(len(_perm1)):
            if ui==_perm1[pi]:
                positions_perm1.append(pi)
    positions_perm2=[]
    for ui in range(user_N):
        for pi in range(len(_perm2)):
            if ui==_perm2[pi]:
                positions_perm2.append(pi)    
    
    for i in range(user_N):        
        position_diff+=abs(positions_perm1[i]-positions_perm2[i])
    # get the average value of position difference
    if(user_N%2==0):
        position_diff=(2*position_diff)/(user_N*user_N)
    else:
        position_diff=(2*position_diff)/(user_N*user_N-1)
    return position_diff

def calculateKendallDistance(_perm1,_perm2):
    """
        Calculate the kendall distance between two permutations of the same items.

        :param _perm1: The first permutation
        :param _perm2: The second permutation         
        :return: returns the kendall distance between two permutations.

    """
    completePermutaionCheck(_perm1,_perm2)
    user_N=len(_perm1)    
    swapped_pairs = 0
    for i in _perm1:
        for j in _perm1:
            if _perm1.index(i)>_perm1.index(j):
                if _perm2.index(i)<_perm2.index(j):
                    swapped_pairs += 1
            elif _perm1.index(i)<_perm1.index(j):
                if _perm2.index(i)>_perm2.index(j):
                    swapped_pairs += 1    
    return swapped_pairs/(user_N*(user_N-1))


def calculateSpearmanR(_scores1,_scores2):

    """
        Use function 'spearmanr' in scipy.stats package to calculate spearman correlation coefficient between two score lists.
        Only applied for two score lists with same size.

        :param _scores1: The first list of scores stored in a python list
        :param _scores2: The second list of scores stored in a python list        
        :return: returns the spearman correlation coefficient between two input score lists, ignore the p-value to test for non-correlation. 
    """
    # error handling for input type
    if not isinstance(_scores1, (list, tuple, np.ndarray)) and not isinstance( _scores1, basestring ):
        raise TypeError("First score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_scores2, (list, tuple, np.ndarray)) and not isinstance( _scores2, basestring ):
        raise TypeError("Second score list must be a list-wise structure defined by '[]' symbol")
    # error handling for input value
    if len(_scores1)*len(_scores2) ==0:
        raise ValueError("Input score lists should have length larger than 0")
    if len(_scores1) != len(_scores2):
        raise ValueError("Input two score lists should have same size")   
    
    return spearmanr(_scores1,_scores2)[0]

def calculatePearsonC(_scores1,_scores2):

    """
        Use function 'pearsonr' in scipy.stats package to calculate pearson correlation coefficient between two score lists.
        Only applied for two score lists with same size.
        
        :param _scores1: The first list of scores stored in a python list
        :param _scores2: The second list of scores stored in a python list        
        :return: returns the pearson correlation coefficient between two input score lists, ignore the p-value to test for non-correlation.
    """
    # error handling for input type
    if not isinstance(_scores1, (list, tuple, np.ndarray)) and not isinstance( _scores1, basestring ):
        raise TypeError("First score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_scores2, (list, tuple, np.ndarray)) and not isinstance( _scores2, basestring ):
        raise TypeError("Second score list must be a list-wise structure defined by '[]' symbol")
    # error handling for input value
    if len(_scores1)*len(_scores2) ==0:
        raise ValueError("Input score lists should have length larger than 0")
    if len(_scores1) != len(_scores2):
        raise ValueError("Input two score lists should have same size")   
    
    return pearsonr(_scores1,_scores2)[0]

# Functions for error handling
def descendingOrderCheck(_ordered_list):
    """
        Check whether the input list is ordered descending. 
        
        :param _ordered_list: The input list that is ordered by descending order         
        :return: returns true if order of _ordered_list is descending else returns false.

    """       
    return all(earlier >= later for earlier, later in zip(_ordered_list, _ordered_list[1:]))

def completePermutaionCheck(_perm1,_perm2):
    """
        Check the valid of two input permutations. 

        :param _perm1: The first permutation
        :param _perm2: The second permutation         
        :return: no returns. Raise error if founded.

    """
    
    if not isinstance(_perm1, (list, tuple, np.ndarray)) and not isinstance( _perm1, basestring ):
        raise TypeError("First permutation must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_perm2, (list, tuple, np.ndarray)) and not isinstance( _perm2, basestring ):
        raise TypeError("Second permutation must be a list-wise structure defined by '[]' symbol")
    
    # error handling for complete permutation
    if len(_perm1)*len(_perm2) ==0:
        raise ValueError("Input permutations should have length larger than 0")
        
    if len(set(_perm1)) < len(_perm1):
        raise ValueError("First permutation include repetitive items")
    if len(set(_perm2)) < len(_perm2):
        raise ValueError("Second permutation include repetitive items")    
    if len(_perm1) != len(_perm2):
        raise ValueError("Input permutations should have same size")