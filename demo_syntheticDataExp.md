
# Usage of runSyntheticExp

### Import all functions in runSyntheticExp


```python
from runSyntheticExp import *
```

#### Initialization

##### Step 1: Specify the input population with size of user and protected group


```python
user_N = 100
pro_N = 50
```

##### Step 2: Choose the fairness measure will be used in the experiment
- Choose from ["rKL", "rND", "rRD"].
- rKL represents KL-divergence fairness measure.
- rND represents normalized difference fairness measure.
- rRD represents ratio difference fairness measure.


```python
gf_measure = "rKL"
```

##### Step 3: Set the cut point at where to compute the fairness measure


```python
cut_point = 10
```

##### Step 4: Specify the file to output optimization results


```python
output_fn = "Fairness_synthetic_"+gf_measure
```

#### Run fairness measure expetiments of synthetic data 


```python
main(user_N,pro_N,gf_measure,cut_point,output_fn)

print "Finished experiments on synthetic data"
print "Result stores in "+ output_fn+"_user"+str(user_N)+"_pro"+str(pro_N)+".csv"
```

    Finished mixing proportion  0.0
    Finished mixing proportion  0.1
    Finished mixing proportion  0.2
    Finished mixing proportion  0.3
    Finished mixing proportion  0.4
    Finished mixing proportion  0.5
    Finished mixing proportion  0.6
    Finished mixing proportion  0.7
    Finished mixing proportion  0.8
    Finished mixing proportion  0.9
    Finished mixing proportion  0.98
    Finished experiments on synthetic data
    Result stores in Fairness_synthetic_rKL_user100_pro50.csv
    
