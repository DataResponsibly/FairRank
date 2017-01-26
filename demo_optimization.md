
# Usage of optimization and runOptimization

### Import all functions in runOptimization
- runOptimization provide the all the functions of optimization and interface to run optimization


```python
from runOptimization import *
```

#### Initialization

##### Step 1: Specify a input data stored in csv file


```python
fn = "GermanCredit_Age35.csv" 
```

##### Step 2: Specify the target attribute to rank on
- If specify target attribute as col(fn)-1, it will generate a score for each user by summing all attributes with equally weight to rank on. col(fn) is the number of attributes in the input data.
- Target attribute can be value from [0,col(fn)-1] while col(fn)-1 represents using weighted summation score as target attribute


```python
target_att = 6 
```

##### Step 3: Specify which value of sentitive attribute represents the protected group


```python
sensi_att = 1
```

##### Step 4: Specify size of K
- K represents size of intermediate layer in optimization neural network. Higher K represents more accurate prediction. Also takes more time to converge.


```python
opt_k = 4
```

##### Step 5: Choose the accuracy measure will be used in the optimization process
- Choose from ["scoreDiff", "positionDiff", "kendallDis", "spearmanDis", "pearsonDis"].
- scoreDiff represents the score difference between two rankings.
- positionDiff represents the rank position difference between two rankings.
- kendallDis represents the kendall distance between two rankings.
- spearmanDis represents spearman correlation between two rankings. Only used the correlation and ignore the p-value in this case.
- pearsonDis represents pearson correlation between two rankings. Only used the correlation and ignore the p-value in this case.


```python
acc_measure = "scoreDiff"
```

##### Step 6: Set the cut point at where to compute the fairness measure


```python
cut_point = 10
```

##### Step 7: Specify the file to output optimization results


```python
out_fn = "testOptimization.csv"
```

#### Test optimization process using above setting


```python
main(fn, target_att, sensi_att, opt_k, acc_measure, cut_point, out_fn)
print "Finished optimization"
print "OPT result stores in "+ out_fn
```

    Finished reading csv!
    Finished fairness normalizer calculation!
    Starting optimization @  4 ACCM  scoreDiff  time:  1485470812.1
    (250, 1.0368499278231544)
    (500, 0.54193378842745721)
    (750, 0.42883044927988201)
    (1000, 0.21465836597579119)
    (1250, 0.20320949600353819)
    (1500, 0.20011416240307803)
    (1750, 0.19853254631944087)
    Ending optimization @  4 ACCM  scoreDiff  time:  1485470993.81
    Finished optimization
    OPT result stores in testOptimization.csv
    
