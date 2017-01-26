from runSyntheticExp import *
# a python script for testing experiments of synthetic data 
# change the corresponding parameter to output your results

# define the size of input population and protected group
user_N = 100
pro_N = 50
# choose the fainress measure 
gf_measure = "rRD"

# define the cut point to compute split fairness measure
cut_point = 10

# define the output file name
output_fn = "Fairness_synthetic_"+gf_measure

# run the synthetic data experiments
main(user_N,pro_N,gf_measure,cut_point,output_fn)

print "Finished experiments on synthetic data"
print "Result stores in "+ output_fn+"_user"+str(user_N)+"_pro"+str(pro_N)+".csv"