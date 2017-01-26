
# Usage of utility

### Import all functions in utility


```python
from utility import *
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
target_att = 1 
```

##### Step 3: Specify which value of sentitive attribute represents the protected group


```python
sensi_att = 1
```

#### Test transform source data function


```python
transformed_data, scores, pro_data, unpro_data, pro_index= transformCSVdata(fn, target_att, sensi_att)
print "Read into:"
print "User Female"
print len(transformed_data), len(pro_data)
```

    Finished reading csv!
    Read into:
    User Female
    1000 548
    
