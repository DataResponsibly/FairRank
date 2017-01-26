from runRealDataExp import *
# a python script for testing experiments of real data sets
# change the corresponding parameter to output your results

# define the folder that includes all the real data csv
data_folder = "RealDatasets"
# define the output file name
output_fn = "Fairness_real_data"
# define which value of sentitive attribute represents the protected group
sensi_att = 1
# run the real data set experiments
main(data_folder,output_fn,sensi_att)

print "Finished experiments on real data sets"
print "Result stores in "+ output_fn+".csv"