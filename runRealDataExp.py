from __future__ import division
import pandas as pd
import measures
# a python script to compute fairness measures of real data sets
# can be run through command line by using command: runRealDataExp data_folder output_fn sensitive_att
# data_folder represents the folder name that stores all the data sets
# output_fn represents the output file name
# sensitive_att represents the value of sensitive attribute for protected group
# test of this script can be found in testRealdataExp.py

KL_DIVERGENCE="rKL" # represent kl-divergence group fairness measure
ND_DIFFERENCE="rND" # represent normalized difference group fairness measure
RD_DIFFERENCE="rRD" # represent ratio difference group fairness measure

def getAllGroupFairness(_data,_sensi_att,_target_att,_reverse_atts,_cut_point,_sensi_bound):
    """
        Run the calculation of all group fairness measures.      
        
        :param _data: The input data stored as data frame in pandas
        :param _sensi_att: The sensitve attribute in the input data
        :param _target_att: The target attribute ranked on
        :param _reverse_atts: The lists of attributes that lower is better
        :param _cut_point: The cut off point of set-wise group fairness calculation
        :param _sensi_bound: The value of sensitve attribute to use as protected group 
                             Applied for binary sensitve attribute
        :return: returns size of protected group, rKL, rND and rRD value.
    """

    # get the protected group
    pro_index=[idx for idx,row in _data.iterrows() if row[_sensi_att] == _sensi_bound]
    # remove sensitive attribute
    input_scores=_data[_target_att]
    if _target_att in _reverse_atts:
        input_ranking=sorted(range(len(input_scores)), key=lambda k: input_scores[k])
    else:
        input_ranking=sorted(range(len(input_scores)), key=lambda k: input_scores[k], reverse=True)
    # get the maximum value first to run faster
    user_N=len(input_scores)
    pro_N=len(pro_index)

    max_rKL=measures.getNormalizer(user_N,pro_N,KL_DIVERGENCE) 
    max_rND=measures.getNormalizer(user_N,pro_N,ND_DIFFERENCE)
    max_rRD=measures.getNormalizer(user_N,pro_N,RD_DIFFERENCE)
    
    gf_rKL=measures.calculateNDFairness(input_ranking, pro_index, _cut_point,KL_DIVERGENCE,max_rKL)
    gf_rND=measures.calculateNDFairness(input_ranking, pro_index, _cut_point,ND_DIFFERENCE,max_rND)
    gf_rRD=measures.calculateNDFairness(input_ranking, pro_index, _cut_point,RD_DIFFERENCE,max_rRD)
    
    return pro_N, gf_rKL, gf_rND, gf_rRD

# define the main function to define all the input parameters
def main(_data_folder,_rez_fn,_sensi_bound):
    """
        Run the group fairness experiments of all real data sets.
        Output group fairness results as csv file.        
        
        :param _data_folder: The file folder that stores all the csv data                
        :param _rez_fn: The output file name of group fairness results
        :param _sensi_bound: The value of sensitve attribute to use as protected group 
                             Applied for binary sensitve attribute
        :return: no returns.
    """
    # define all the real data set name list
    datasets=['ProPublica','GermanCredit']

    # define the sensitive attribute mapping dictionary
    sensi_atts_dic={}
    sensi_atts_dic['ProPublica']=['race','sex']
    sensi_atts_dic['GermanCredit']=['sex','age25','age35']
    # define the noweight attributes mapping dictionary
    noweight_atts_dic={}
    noweight_atts_dic['ProPublica,race']=[]
    noweight_atts_dic['ProPublica,sex']=[]
    noweight_atts_dic['GermanCredit,sex']=[]
    noweight_atts_dic['GermanCredit,age25']=[]
    noweight_atts_dic['GermanCredit,age35']=[]

    # define the reverse attributes mapping dictionary
    reverse_atts_dic={}
    reverse_atts_dic['ProPublica,race']=['priors_count','Violence_rawscore','Recidivism_rawscore']
    reverse_atts_dic['ProPublica,sex']=['priors_count','Violence_rawscore','Recidivism_rawscore']
    reverse_atts_dic['GermanCredit,sex']=[]
    reverse_atts_dic['GermanCredit,age25']=[]
    reverse_atts_dic['GermanCredit,age35']=[]

    # define the mapping dictionary of cut point
    cutpoint_dic={}
    cutpoint_dic['ProPublica']=10
    cutpoint_dic['GermanCredit']=10


    rez_fn=_rez_fn+".csv"
    with open(rez_fn,'w') as mf:
        mf.write("Dataset,User_N,SensitiveATT,Pro_N,Pro_percent,TargetAtt,rKL,rND,rRD\n")
    rez_file=open(rez_fn, 'a')
    # read all the real data into 
    for di in datasets:
        sensi_atts=sensi_atts_dic[di]
        cut_point=cutpoint_dic[di]
        for si in sensi_atts:
            current_fn=_data_folder+"/"+di+"_"+si+".csv"
            try:        
                data=pd.read_csv(current_fn)
            except EnvironmentError:
                print("Cannot find "+current_fn)
            
            print "Finishing computation of data: "+di+"_"+si
            reverse_atts=reverse_atts_dic[di+","+si]
            noweight_atts=[si]
            # get the target columns
            target_cols_df=data.loc[:,data.columns.difference(noweight_atts)] 
            target_cols=list(target_cols_df)
            
            for ti in target_cols:
                pro_N,gf_rKL,gf_rND,gf_rRD=getAllGroupFairness(data,si,ti,reverse_atts,cut_point,_sensi_bound)
                user_N=len(data)
                pro_percent=round(pro_N*100/user_N)
                rez_fline=di+","+str(user_N)+","+si+","+str(pro_N)+","+str(pro_percent)+","+ti+","+str(gf_rKL)+","+str(gf_rND)+","+str(gf_rRD)+"\n"
                rez_file.write(rez_fline)
    rez_file.close()        



if __name__ == "__main__":
    main()