from __future__ import division
import numpy as np
from numba.decorators import jit
import measures # import for accuracy measures
import utility # import for calculation of weighted scores

# a python script define optimization process
# test of this script can be found in testOptimization.py
# Part of optimization code refers from github https://github.com/zjelveh/learning-fair-representations/blob/master/lfr.py 

SCORE_DIVERGENCE="scoreDiff" # represent average score difference -ranking accuracy measure
POSITION_DIFFERENCE="positionDiff" # represent average position difference -ranking accuracy measure
KENDALL_DIS="kendallDis" # represent kendall distance -ranking accuracy measure
SPEARMAN_COR="spearmanDis" # represent spearman correlation -ranking accuracy measure
PEARSON_COR="pearsonDis" # represent pearson correlation -ranking accuracy measure


def calculateEvaluateRez(_rez,_data,_inputscores,_k,_accmeasure):
    """
        Calculate estimated scores of all input user and ranking accuracy of the corresponding ranking after optimization converged.
        :param _rez: The optimization parameter results of L-BFGS algorithm after converged
        :param _data: The input data, each row is a feature vector of one user
        :param _inputscores: The input scores of data that can be weighted scores or some score attributes
        :param _k: The number of clusters in the intermediate layer of neural network
        :param _accmeasure: The accuracy measure used in this function
        :return: returns the estimated scores and ranking accuracy of corresponding ranking
    """
    user_N,att_N=_data.shape

    # error handling for input type
    if not isinstance(_rez, (list, tuple, np.ndarray)) and not isinstance( _rez, basestring ):
        raise TypeError("Input parameter list must be a list-wise structure defined by '[]' symbol")
    if not isinstance(_inputscores, (list, tuple, np.ndarray)) and not isinstance( _inputscores, basestring ):
        raise TypeError("Input score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance( _k, ( int, long ) ):
        raise TypeError("Input k must be an integer")
    if not isinstance( _accmeasure, str ):
        raise TypeError("Input accuracy measure must be a string that \
            choose from ['scoreDiff', 'positionDiff', 'kendallDis', 'spearmanDis', 'pearsonDis'] defined in the begining of this file")
    
    # error handling for input value
    if user_N == 0:
        raise ValueError("Input data should not be empty")
    if att_N == 0:
        raise ValueError("Input data should have at least one attribute column")

    if len(_rez) == 0:
        raise ValueError("Input _rez should not be empty")
    if len(_inputscores) == 0:
        raise ValueError("Input estimated score list should not be empty")
    if _k == 0:
        raise ValueError("Input k must be an integer larger than 0")    

    # initialize the clusters
    clusters=np.matrix(_rez[0][(2 * att_N) + _k:]).reshape((_k, att_N))
    alpha1 = _rez[0][att_N : 2 * att_N]
    # get the distance between input user X and intermediate clusters Z
    dists_x = distances(_data, clusters, alpha1, user_N, att_N, _k)
    # compute the probability of each X maps to Z
    Mnk_x=M_nk(dists_x, user_N, _k)
    # get the estiamted scores and ranking accuracy
    scores_hat, ranking_accuracy = calculateEstimateY(Mnk_x, _inputscores, clusters, user_N, _k, _accmeasure)
    return scores_hat, ranking_accuracy

@jit
def distances(_X, _clusters, _alpha, _N, _P, _k): 
    """
        Calculate the distance between input X and clusters Z.
        :param _X: The input user feature vector 
        :param _clusters: The clusters in the intermediate Z
        :param _alpha: The weight of each attribute in the input X
        :param _N: The total user number in input X
        :param _P: The attribute number in input X
        :param _k: The number of clusters in the intermediate layer of neural network        
        :return: returns the distance matrix between X and Z.
    """
    dists = np.zeros((_N, _k))
    for i in range(_N):
        for p in range(_P):
            for j in range(_k):    
                dists[i, j] += (_X[i, p] - _clusters[j, p]) * (_X[i, p] - _clusters[j, p]) 
    return dists

@jit
def M_nk(_dists, _N, _k): 
    """
        Calculate the probability of input X maps to clusters Z.
        :param _dists: The distance matrix between X and Z 
        :param _clusters: The clusters in the intermediate Z
        :param _alpha: The weight of each attribute in the input X
        :param _N: The total user number in input X
        :param _P: The attribute number in input X
        :param _k: The number of clusters in the intermediate layer of neural network        
        :return: returns the probability mapping matrix between X and Z.
    """
    M_nk = np.zeros((_N, _k))
    exp = np.zeros((_N, _k))
    denom = np.zeros(_N)
    for i in range(_N):
        for j in range(_k):
            exp[i, j] = np.exp(-1 * _dists[i, j])
            denom[i] += exp[i, j]
        for j in range(_k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk
@jit    
def M_k(_M_nk, _N, _k): 
    # print(_M_nk, _N, _k)
    """
        Calculate the summed probability of all input users.
        :param _M_nk: The probability mapping matrix between X and Z 
        :param _N: The total user number in input X
        :param _k: The number of clusters in the intermediate layer of neural network        
        :return: returns the summed probability matrix of all users.
    """
    M_k = np.zeros(_k)

    for j in range(_k):
        for i in range(_N):
            M_k[j] += _M_nk[i, j]
        M_k[j] /= _N
    return M_k

@jit    
def x_n_hat(_X, _M_nk, _clusters, _N, _P, _k): 
    """
        Calculate the estimated X through clusters Z.
        :param _X: The input user feature vector 
        :param _M_nk: The probability mapping matrix between X and Z
        :param _clusters: The clusters in the intermediate Z
        :param _N: The total user number in input X
        :param _P: The attribute number in input X
        :param _k: The number of clusters in the intermediate layer of neural network        
        :return: returns the estimated X and loss between input X and estimated X.
    """
    x_n_hat = np.zeros((_N, _P))
    L_x = 0.0
    for i in range(_N):
        for p in range(_P):
            for j in range(_k):
                x_n_hat[i, p] += _M_nk[i, j] * _clusters[j, p]
            L_x += (_X[i, p] - x_n_hat[i, p]) * (_X[i, p] - x_n_hat[i, p])
    L_x=L_x/_N
    return x_n_hat, L_x

# @jit 
def calculateEstimateY(_M_nk_x, _inputscores, _clusters, _N, _k,_accmeasure):
    """
        Calculate the estimated score and ranking accuracy of corresponding ranking.
        :param _M_nk_x: The probability mapping matrix from input X and clusters Z 
        :param _inputscores: The input scores of all users
        :param _clusters: The clusters in the intermediate Z
        :param _N: The total user number in input X        
        :param _k: The number of clusters in the intermediate layer of neural network
        :param _accmeasure: The ranking accuracy measure used in this function        
        :return: returns the estimated X and loss between input X and estimated X.
    """
    score_hat = np.zeros(_N) # initialize the estimated scores
    # calculate estimate score of each user by mapping probability between X and Z     
    for ui in range(_N):
        score_hat_u = 0.0
        for ki in range(_k):
            score_hat_u += (_M_nk_x[ui,ki] * _clusters[ki])                         
        score_hat[ui] = utility.calculateWeightedScores(score_hat_u)
        
    ranking_loss = 0.0       
    
    score_hat=list(score_hat)    
    _inputscores=list(_inputscores)
    
    # generate permutations of two score lists returned permutation of sorted id
    per_scores_hat=sorted(range(len(score_hat)), key=lambda k: score_hat[k],reverse=True)
    per_scores_input=sorted(range(len(_inputscores)), key=lambda k: _inputscores[k],reverse=True)
    # sort the scores in descending order
    sorted_score_hat = score_hat   
    sorted_score_hat.sort(reverse=True)
    sorted_inputscores = _inputscores   
    sorted_inputscores.sort(reverse=True)

    if _accmeasure==SCORE_DIVERGENCE:
        L_y=measures.calculateScoreDifference(sorted_score_hat,sorted_inputscores) 
        ranking_loss = L_y

    elif _accmeasure==POSITION_DIFFERENCE:        
        L_y=measures.calculatePositionDifference(per_scores_hat,per_scores_input) 
        ranking_loss = L_y

    elif _accmeasure==KENDALL_DIS: 
        L_y=measures.calculateKendallDistance(per_scores_hat,per_scores_input) # kendall distance        
        ranking_loss = L_y
    
    # for spearman and pearson relation, use the negative value to minimize during optimization
    elif _accmeasure==SPEARMAN_COR:
        L_y=measures.calculateSpearmanR(score_hat,_inputscores)
        ranking_loss = -L_y 

    elif _accmeasure==PEARSON_COR:
        L_y=measures.calculatePearsonC(score_hat,_inputscores)
        ranking_loss=-L_y
    
    return score_hat, ranking_loss

def lbfgsOptimize(_params, _data, _pro_data, _unpro_data, 
        _inputscores, _accmeasure, _k, A_x = 0.01, A_y = 1, A_z = 100, results=0):
    
    """
        The function to run the optimization using l-bfgs algorithm.
        :param _params: The initialized optimization parameters
        :param _data: The input data of all users - X 
        :param _pro_data: The input data of protected group
        :param _unpro_data: The input data of unprotected group
        :param _inputscores: The scores of input users which can be a score attribute or summed score of all attributes
        :param _accmeasure: The ranking accuracy measure used in this function
        :param _k: The number of clusters in the intermediate layer of neural network        
        :param A_x: The super parameter - optimization weight for accuracy of reconstructing X
        :param A_y: The super parameter - optimization weight for ranking accuracy
        :param A_z: The super parameter - optimization weight for group fairness
        :param results: The flag of optimization, initialize to 0, update to 1 when optimization converged 
        :return: returns the estimated scores of all user and the probability mapping of protected and unprotected group if converged.
                 returns the last loss during optimization if optimization doesn't converge.
    """

    lbfgsOptimize.iters += 1 
    # get basic statistics
    user_N, att_N= _data.shape
    pro_N, pro_att_N = _pro_data.shape
    unpro_N, unpro_att_N = _unpro_data.shape

    # error handling for input type
    if not isinstance(_inputscores, (list, tuple, np.ndarray)) and not isinstance( _inputscores, basestring ):
        raise TypeError("Input score list must be a list-wise structure defined by '[]' symbol")
    if not isinstance( _k, ( int, long ) ):
        raise TypeError("Input k must be an integer")
    if not isinstance( _accmeasure, str ):
        raise TypeError("Input accuracy measure must be a string that \
            choose from ['scoreDiff', 'positionDiff', 'kendallDis', 'spearmanDis', 'pearsonDis'] defined in the begining of this file")
    
    # error handling for input value
    if user_N == 0:
        raise ValueError("Input data should not be empty")
    if (att_N *pro_att_N *unpro_att_N) == 0:
        raise ValueError("Input data, protected group data, and unprotected group data should have at least one attribute column")
    if att_N != pro_att_N:
        raise ValueError("Input protected group data '_pro_data' should have same size with '_data'")
    if att_N != unpro_att_N:
        raise ValueError("Input unprotected group data '_unpro_data' should have same size with '_data'")

    
    if len(_inputscores) == 0:
        raise ValueError("Input estimated score list should not be empty")
    if _k == 0:
        raise ValueError("Input k must be an integer larger than 0")


    # initialize parameters of neural network
    alpha0 = _params[:att_N]
    alpha1 = _params[att_N : 2 * att_N]
    w = _params[2 * att_N : (2 * att_N) + _k]
    # initialize the starting clusters
    clusters = np.matrix(_params[(2 * att_N) + _k:]).reshape((_k, att_N)) 
    # compute the distance from X to Z    
    dists_x = distances(_data, clusters, alpha1, user_N, att_N, _k)  
    M_nk_x = M_nk(dists_x, user_N, _k)    
    
   
    # based on the cluster centroid compute the distance of protected group and unprotected group
    pro_dists = distances(_pro_data, clusters, alpha1, pro_N, att_N, _k)
    unpro_dists = distances(_unpro_data, clusters, alpha0, unpro_N, att_N, _k)
       
    # compute the probability mapping from X to Z
    pro_M_nk = M_nk(pro_dists, pro_N, _k)
    unpro_M_nk = M_nk(unpro_dists, unpro_N, _k)
    
    # compute the summed probability of protected and unprotected group
    pro_M_k = M_k(pro_M_nk, pro_N, _k)
    unpro_M_k = M_k(unpro_M_nk, unpro_N, _k)
    # compute the mapping difference between protected group and unprotected group i.e. sub-loss of group fairness
    L_z = 0.0
    for j in range(_k):
        L_z += abs(pro_M_k[j] - unpro_M_k[j])
    
    # compute the estimated x hat from Z i.e. sub-loss of X
    pro_x_n_hat, L_x1 = x_n_hat(_pro_data, pro_M_nk, clusters, pro_N, att_N, _k)
    unpro_x_n_hat, L_x2 = x_n_hat(_unpro_data, unpro_M_nk, clusters, unpro_N, att_N, _k)
    L_x = L_x1 + L_x2
    
    # compute the estimated scores and ranking accuracy i.e. sub-loss of ranking Y

    estimate_scores, L_y = calculateEstimateY(M_nk_x, _inputscores, clusters, user_N, _k, _accmeasure)
    
    # generate the total loss    
    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    # print out the current loss after each 250 iterations
    if lbfgsOptimize.iters % 250 == 0:
        print(lbfgsOptimize.iters, criterion)
       
    if results:
        return estimate_scores, pro_M_nk, unpro_M_nk
    else:
        return criterion
# after each optimization, reset the iteration to zero
lbfgsOptimize.iters = 0

def initOptimization(_data,_k):
    """
        Initialize the parameter and bound of optimization.
        :param _data: The input data w.r.t X       
        :param _k: The number of clusters in the intermediate layer of neural network
        :return: returns the parameter vector and bound of optimization.
    """
    user_N,att_N=_data.shape

    # error handling for input type    
    if not isinstance( _k, ( int, long ) ):
        raise TypeError("Input k must be an integer")

    # error handling for input value
    if user_N == 0:
        raise ValueError("Input data should not be empty")
    if att_N == 0:
        raise ValueError("Input data should have at least one attribute column")
    if _k == 0:
        raise ValueError("Input k must be an integer larger than 0")    


    # initialize the parameter vector for neural network
    rez = np.random.uniform(size=_data.shape[1] * 2 + _k + _data.shape[1] * _k)
    # initialize the bound of optimization algorithm
    bnd = []
    for i, k2 in enumerate(rez):
        if i < _data.shape[1] * 2 or i >= _data.shape[1] * 2 + _k:
            bnd.append((None, None))
        else:
            bnd.append((0, 1))
    return rez, bnd