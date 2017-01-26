We includes two real data sets here: German credit data and ProPublic data.
German credit data is originally available in UCI. Link is https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data).
ProPublic data is originally described in https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm. Data can be found in here https://github.com/propublica/compas-analysis.

Data sets uploaded here only include some manually choosed attributes from original one. We binarized the sentitive attributes like age and sex by different thresholds for these two data sets.

For example, for sensitive attribute age, we use three threshold (age<25, age<35) to transform the age into binary attribute. When the age is smaller than 25, the binarized age is 1 and 0 otherwise. We used age<25, age<35 and sex=female to binarize German credit data. The output of these three threshold are GermanCredit_age25, GermanCredit_age35 and GermanCredit_sex.

For ProPulica, we use race=black and sex=female to binarize the sensitive attributes. The output of these two threshold are ProPublica_race and ProPublica_sex. 


Above processed data is used in experiments of poster_FATML16 and arXiv_FATML16 (both can be found in docs).
