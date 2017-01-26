As part of the FairRank package, we include two real datasets, German
Credit
(https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
and ProPublica's COMPAS dataset
(https://github.com/propublica/compas-analysis).

<<<<<<< HEAD
Data sets uploaded here only include some manually choosed attributes from original one. We binarized the sentitive attributes like age and sex by different thresholds for these two data sets.

For example, for sensitive attribute age, we use three threshold (age<25, age<35) to transform the age into binary attribute. When the age is smaller than 25, the binarized age is 1 and 0 otherwise. We used age<25, age<35 and sex=female to binarize German credit data. The output of these three threshold are GermanCredit_age25, GermanCredit_age35 and GermanCredit_sex.

For ProPulica, we use race=black and sex=female to binarize the sensitive attributes. The output of these two threshold are ProPublica_race and ProPublica_sex. 


Above processed data is used in experiments of poster_FATML16 and arXiv_FATML16 (both can be found in docs).
=======
Our version of the datasets includes a subset of manually choosen
attributes.  Further, we binarized the sensitive attributes on
different thresholds.

The processed datasets were used in experiments as reported in
https://arxiv.org/abs/1610.08559.
>>>>>>> 71677934cca3883d04c12090a6cb861883225dd9
