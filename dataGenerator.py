import random
import numpy as np
# a python script define algorithm to generate different rankings
# test of this script can be found in testDataGenerator.py

def generateUnfairRanking(_ranking,_protected_group,_fairness_probability):
    """
        An algorithm for generating rankings with varying degree of fairness.

        :param _ranking: A ranking
        :param _protected_group: The protected group
        :param _fairness_probability: The unfair degree, where 0 is most unfair (unprotected 
                       group ranked first) and 1 is fair (groups are mixed randomly 
                       in the output ranking)
        :return: returns a ranking that has the specified degree of unfairness w.r.t. 
                 the protected group
    """
    # error handling for ranking and protected group
    completeCheckRankingProperties(_ranking,_protected_group)

    if not isinstance( _fairness_probability, ( int, long, float, complex ) ):
        raise TypeError("Input fairness probability must be a number")
    # error handling for value
    if _fairness_probability > 1 or _fairness_probability < 0:
        raise ValueError("Input fairness probability must be a number in [0,1]")


    pro_ranking=[x for x in _ranking if x not in _protected_group] # partial ranking of protected member
    unpro_ranking=[x for x in _ranking if x in _protected_group] # partial ranking of unprotected member
    pro_ranking.reverse() #prepare for pop function to get the first element
    unpro_ranking.reverse()
    unfair_ranking=[]
    
    while(len(unpro_ranking)>0 and len(pro_ranking)>0):
        random_seed=random.random() # generate a random value in range [0,1]
        if random_seed<_fairness_probability:
            unfair_ranking.append(unpro_ranking.pop()) # insert protected group first
        else:
            unfair_ranking.append(pro_ranking.pop()) # insert unprotected group first
    
    if len(unpro_ranking)>0: # insert the remain unprotected member
        unpro_ranking.reverse()
        unfair_ranking=unfair_ranking+unpro_ranking        
    if len(pro_ranking)>0: # insert the remain protected member
        pro_ranking.reverse()
        unfair_ranking=unfair_ranking+pro_ranking
        
    if len(unfair_ranking)<len(_ranking): # check error for insertation
        print "Error!"
    return unfair_ranking
    
# Function for error handling
def completeCheckRankingProperties(_ranking,_protected_group):    
    """
        Check whether input ranking and protected group is valid.

        :param _ranking: A ranking
        :param _protected_group: The protected group
        
        :return: no returns. Raise errors if founded.
    """
    # error handling for input type
    if not isinstance(_ranking, (list, tuple, np.ndarray)) and not isinstance( _ranking, basestring ):
        raise TypeError("Input ranking must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_protected_group, (list, tuple, np.ndarray)) and not isinstance( _protected_group, basestring ):
        raise TypeError("Input protected group must be a list-wise structure defined by '[]' symbol")

    user_N=len(_ranking)
    pro_N=len(_protected_group)

    # error handling for input value
    if user_N <= 0: # check size of input ranking
        raise ValueError("Please input a valid ranking")
    if pro_N <= 0: # check size of input ranking
        raise ValueError("Please input a valid protected group whose length is larger than 0")
    
    if pro_N >= user_N: # check size of protected group
        raise ValueError("Please input a protected group with size less than total user")

    if len(set(_ranking)) != user_N: # check for repetition in input ranking
        raise ValueError("Please input a valid complete ranking")    

    if len(set(_protected_group)) != pro_N: # check repetition of protected group
        raise ValueError("Please input a valid protected group that have no repetitive members")
    
    if len(set(_protected_group).intersection(_ranking)) <=0: # check valid of protected group
        raise ValueError("Please input a valid protected group that is a subset of total user")  

    if len(set(_protected_group).intersection(_ranking)) != pro_N: # check valid of protected group
        raise ValueError("Please input a valid protected group that is a subset of total user")