from utility import *
# a python script to test utility python script
# change the corresponding parameter to output your results

# define the source csv which should have sensitive att located in last columns and value is 0 or 1.
fn = "GermanCredit_Age35.csv"
# define the target attribute used to rank on  
target_att = 1 
# define the value of sensitive attribute to represent protected group. In this case, 1 represents the protected group.
sensi_att = 1
 
# call the utility function to format source data for computation
transformed_data, scores, pro_data, unpro_data, pro_index= transformCSVdata(fn, target_att, sensi_att)

print "User Female"
print len(transformed_data), len(pro_data)
