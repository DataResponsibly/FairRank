
# Usage of dataGenerator

### Import all functions in dataGenerator


```python
from dataGenerator import *
```

#### Initialization

##### Step 1: Specify the input population with size of user and protected group


```python
user_N = 100 
pro_N = 50
```

##### Step 2: Specify the input mixing proportion of generating algorithm


```python
mixing_proportion = 0.5
```

##### Step 3: Generate a random ranking and position of protected group to initiate the algorithm


```python
test_ranking = [x for x in range(user_N)]
pro_index = [x for x in range(pro_N)]
```

#### Test ranking-generator algorithm


```python
output_ranking = generateUnfairRanking(test_ranking, pro_index, mixing_proportion)
print "Successfully generate a ranking with input setting!"
```

    Successfully generate a ranking with input setting!
    
