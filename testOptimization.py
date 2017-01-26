from runOptimization import *
# a python script for testing optimization process with a input data
# change the corresponding parameter to output your results

# define a csv file stored input data
# this csv should have sensitive att located in last columns and value is 0 or 1.
fn = "GermanCredit_Age35.csv" 
# define target attribute to rank on if not will generate a equally weighted summation as score to rank on
# target attribute can be value from [0,col(fn)-1] while col(fn)-1 represents using weighted summation score as target attribute
target_att = 6 
# define which value of sentitive attribute represents the protected group
sensi_att = 1
# define the k value represents size of intermediate layer in optimization neural network. Higher K represents more accurate prediction. Also takes more time to converge.
opt_k = 4
# choose accuracy measure to use in the optimization. Choose from ["scoreDiff", "positionDiff", "kendallDis", "spearmanDis", "pearsonDis"].
acc_measure = "scoreDiff"
# define the cut point using in the computation of fairness measures
cut_split = 10
# define the output file of optimization results
out_fn = "testOptimization.csv"

# starting run the optimization
main(fn, target_att, sensi_att, opt_k, acc_measure, cut_split, out_fn)

print "Finished optimization"
print "OPT result stores in "+out_fn
