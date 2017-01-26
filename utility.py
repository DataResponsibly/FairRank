from __future__ import division
import csv
import numpy as np
# a python script define utility function i.e. format source data for computation
# test of this script can be found in testUtility.py

def transformCSVdata(_data_fn,_target_cols, _sensi_bound): 
    """
        Read pre-processed csv data and initialize data of protected and unprotected group.
        Csv data has no header and sensitve attribute is in the last column. 

        :param _data_fn: The file name of csv data      
        :param _target_cols: The target column ranked on, value from [0,col(_data_fn)-1] since sensitve attribute is on last column
                             If _target_cols equals to (col(_data_fn)-1), then use weighted summation of all attributes as target
        :param _sensi_bound: The value of sensitve attribute to use as protected group 
                             Applied for binary sensitve attribute
        :return: returns the data frame of all users, protected and unprotected group,
                 returns the scores of some attribute or summation of all attributes to rank on,
                 returns the index of protected group.
    """
    if not isinstance( _data_fn, str ):
        raise TypeError("Input file name must be a string which specify the path of input csv file")
    # read into the csv file
    header = True 
    dat = []
    
    try:        
        with open(_data_fn, 'rb') as f:
            rows = csv.reader(f)        
            for row in rows:
                if header:
                    header = False
                    continue
                dat.append([float(r) for r in row])
    except EnvironmentError as e:
        print("Cannot find the csv file")


    if not isinstance( _target_cols, ( int, long ) ):
        raise TypeError("Input target column must be an integer value from [0, col(data)-2], data is the input data in file _data_fn")
    if not isinstance( _sensi_bound, ( int, long ) ):
        raise TypeError("Input value of sensitive attribute must be an integer value can be 0 or 1")
    
    if len(dat) == 0:
        raise ValueError("Input file should not be empty")
    print('Finished reading csv!') 
    data = np.array(dat)
    user_N, att_N= data.shape

    if _sensi_bound > 1 or _sensi_bound < 0:
        raise ValueError("Input value of sensitive attribute must be an integer value can be 0 or 1")    
    if _target_cols > (att_N-1) or _target_cols < 0:
        raise ValueError("Input value of target column must be an integer value in range [0, col(data)-2], data is the input data in file _data_fn")

    # separate the sensitve attribute
    sensi_att = np.array(data[:,-1]).flatten()
    data = data[:,:-1] 
       
    # get the protected and unprotected group 
    pro_index = np.array(np.where(sensi_att ==_sensi_bound))[0].flatten()    
    unpro_index = np.array(np.where(sensi_att !=_sensi_bound))[0].flatten()

    pro_data = data[pro_index,:]
    unpro_data = data[unpro_index,:]
    
    # ranked on some attributes or scores by weighted summation of all attributes 
    _, att_N_after = data.shape

    if _target_cols==att_N_after:
        scores=calculateWeightedScores(data)
    else:
        scores=data[:,_target_cols]
    
    return data,scores,pro_data,unpro_data,pro_index

def calculateWeightedScores(_data): 
    """
        Calculate a list of scores by equally weighted summation of all the attributes in the _data.

        :param _data: The input data, each row is a feature vector of one user. Using dataframe to store.     
        :return: returns a score list.
    """
    user_N,att_N=_data.shape

    # error handling for input value
    if user_N == 0:
        raise ValueError("Input data should not be empty")
    if att_N == 0:
        raise ValueError("Input data should have at least one attribute column")
    # get the average weight for each attribute
    avg_weight=1.0*1/(att_N)
    
    weights=[]
    for ai in range(att_N):
        weights.append(avg_weight)
    weights_vector=np.array(weights).transpose()

    scores=np.dot(_data,weights_vector)
    if len(scores) != user_N:
        raise ValueError("Computation error appear in .dot opration")
    return scores 
