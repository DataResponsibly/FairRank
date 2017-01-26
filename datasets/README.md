Overview of the datasets
------------------------

As part of the FairRank package, we include two real datasets, German
Credit
(https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
and ProPublica's COMPAS dataset
(https://github.com/propublica/compas-analysis).

Explanation of different version of dataset
--------------------------------------------
Data sets uploaded here only include some manually choosed attributes from original one. We binarized the sentitive attributes like age and sex by different thresholds for these two data sets.

For example, for sensitive attribute age, we use three threshold (age<25, age<35) to transform the age into binary attribute. When age is smaller than 25, the binarized age is 1 and 0 otherwise. We use age<25, age<35 and sex=female to binarize age and sex. The output of these three threshold are GermanCredit_age25.csv, GermanCredit_age35.csv and GermanCredit_sex.csv.

For ProPulica, we use race=black and sex=female to binarize the sensitive attributes. The output of these two threshold are ProPublica_race.csv and ProPublica_sex.csv. 

Usage of datasets
-----------------
The processed datasets were used in experiments as reported in https://arxiv.org/abs/1610.08559.

