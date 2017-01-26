from __future__ import division
import numpy as np
import scipy.optimize as optim
import time
import optimization
import measures
import utility
# a python script for optimization. Can be run from command line by following command
# runOptimization input_fn target_att sensi_value k acc_measure cut_point output_fn
# input_fn represents the csv file stores the source data
# target_att represents the target attribute to rank on 
# sensi_value is the value of sentitive attribute represents the protected group
# k represents the size of intermediate layer of neural network
# acc_measure is choose from ["scoreDiff", "positionDiff", "kendallDis", "spearmanDis", "pearsonDis"]
# cut_point is the cut position of ranking to compute split fairness measures.
# output_fn represents the output file of optimization results

# test of this script can be found in testOptimization.py

# constant of opmitization script
KL_DIVERGENCE="rKL" # represent kl-divergence group fairness measure
ND_DIFFERENCE="rND" # represent normalized difference group fairness measure
RD_DIFFERENCE="rRD" # represent ratio difference group fairness measure

SCORE_DIVERGENCE="scoreDiff" # represent average score difference -ranking accuracy measure
POSITION_DIFFERENCE="positionDiff" # represent average position difference -ranking accuracy measure
KENDALL_DIS="kendallDis" # represent kendall distance -ranking accuracy measure
SPEARMAN_COR="spearmanDis" # represent spearman correlation -ranking accuracy measure
PEARSON_COR="pearsonDis" # represent pearson correlation -ranking accuracy measure

def main(_csv_fn,_target_col,_sensi_bound,_k,_accmeasure,_cut_point,_rez_fn):
    """
        Run the optimization process.
        Output evaluation results as csv file.
        Output results (accuracy, group fairness in op, values of group fairness measures) during optimization as txt files. 
        
        :param _csv_fn: The file name of input data stored in csv file
                        In csv file, one column represents one attribute of user
                        one row represents the feature vector of one user
        :param _target_col: The target attribute ranked on i.e. score of ranking
        :param _sensi_bound: The value of sensitve attribute to use as protected group, 0 or 1, usually 1 represnts belonging to protected group 
                             Applied for binary sensitve attribute
        :param _k: The number of clusters in the intermediate layer of neural network
        :param _accmeasure: The accuracy measure used in this function, one of constant string defined in this py file
        :param _cut_point: The cut off point of set-wise group fairness calculation
        :param _rez_fn: The file name to output optimization results
        :return: no returns.
    """        

    data,input_scores,pro_data,unpro_data,pro_index=utility.transformCSVdata(_csv_fn,_target_col,_sensi_bound)
    
    user_N = len(data)
    pro_N = len(pro_data)

	# get the maximum value first to run fast        
    max_rKL=measures.getNormalizer(user_N,pro_N,KL_DIVERGENCE) 
    max_rND=measures.getNormalizer(user_N,pro_N,ND_DIFFERENCE)
    max_rRD=measures.getNormalizer(user_N,pro_N,RD_DIFFERENCE)    
    
    print "Finished fairness normalizer calculation!"
    input_ranking=sorted(range(len(input_scores)), key=lambda k: input_scores[k],reverse=True)

    input_rKL=measures.calculateNDFairness(input_ranking,pro_index,_cut_point,KL_DIVERGENCE,max_rKL)
    input_rND=measures.calculateNDFairness(input_ranking,pro_index,_cut_point,ND_DIFFERENCE,max_rND)
    input_rRD=measures.calculateNDFairness(input_ranking,pro_index,_cut_point,RD_DIFFERENCE,max_rRD)


    # record the start time of optimization
    start_time = time.time()
    print "Starting optimization @ ",_k,"ACCM ",_accmeasure," time: ", start_time

    # initialize the optimization
    rez,bnd=optimization.initOptimization(data,_k) 
    
    optimization.lbfgsOptimize.iters=0                
    rez = optim.fmin_l_bfgs_b(optimization.lbfgsOptimize, x0=rez, disp=1, epsilon=1e-5, 
                   args=(data, pro_data, unpro_data, input_scores, _accmeasure, _k, 0.01,
                         1, 100, 0), bounds = bnd,approx_grad=True, factr=1e12, pgtol=1e-04,maxfun=15000, maxiter=15000)
    end_time = time.time()
    print "Ending optimization @ ",_k,"ACCM ",_accmeasure," time: ", end_time
    # evaluation after converged
    estimate_scores,acc_value=optimization.calculateEvaluateRez(rez,data,input_scores,_k,_accmeasure)
    estimate_ranking=sorted(range(len(estimate_scores)), key=lambda k: estimate_scores[k],reverse=True)
    # compute the value of fairness measure after converged
    eval_rKL=measures.calculateNDFairness(estimate_ranking,pro_index,_cut_point,KL_DIVERGENCE,max_rKL)
    eval_rND=measures.calculateNDFairness(estimate_ranking,pro_index,_cut_point,ND_DIFFERENCE,max_rND)
    eval_rRD=measures.calculateNDFairness(estimate_ranking,pro_index,_cut_point,RD_DIFFERENCE,max_rRD)
    
    # prepare the result line to write
    # initialize the outputted csv file
    result_fn=_rez_fn+".csv"
    with open(result_fn,'w') as mf:
        mf.write("UserN,pro_N,K,TargetAtt,AccMeasure,acc_value,rKL_input,rKL_converged,rND_input,rND_converged,rRD_input,rRD_converged,secondsSpent\n")
    rez_file=open(result_fn, 'a')
    
    rez_fline=str(user_N)+","+str(pro_N)+","+str(_k)+","+str(_target_col)+","+str(_accmeasure)+","+str(acc_value)+","+str(input_rKL)+","+str(eval_rKL)+","+str(input_rND)+","+str(eval_rND)+","+str(input_rRD)+","+str(eval_rRD)+","+str(end_time-start_time)+"\n"
    rez_file.write(rez_fline)
    rez_file.close()

if __name__ == "__main__":
    main()




            
            
            
            