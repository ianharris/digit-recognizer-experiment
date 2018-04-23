
# Overview

This notebook is to split the training data set from Kaggle's digit recognizer challenge into a new, smaller training set and a validation set. These new data sets will be used in hyper-parameter tuning and CNN architecture exploration with a view to understanding the optimal CNN architecture for the problem


```python
import pandas as pd
import numpy as np
import random
import math
```


```python
# read the data in
data = pd.read_csv('datasets/train.csv')
```


```python
# print out the label value counts to see how many occurrences of each label exist in the database
data.label.value_counts()
```




    1    4684
    7    4401
    3    4351
    9    4188
    2    4177
    6    4137
    0    4132
    4    4072
    8    4063
    5    3795
    Name: label, dtype: int64




```python
# create a list of lists - each sub-list will contain all the indices corresponding to a single label 
# in the input training set
indarray = []
for i in range(0,10):
    indarray.append(data.index[data.label == i].tolist())
```


```python
# print out the length of the index lists as a check that the data has been divided correctly
for i in range(len(indarray)):
    print(len(indarray[i]))
```

    4132
    4684
    4177
    4351
    4072
    3795
    4137
    4401
    4063
    4188



```python
# create training and validation subsets of the indarray (index array)
trainind = []
validationind = []
for i in range(len(indarray)):
    # calculate '10%'
    tenpercent = math.ceil(len(indarray[i]) * 0.1)
    # form the validation index and train index arrays
    validationind.append(indarray[i][0:tenpercent])
    trainind.append(indarray[i][tenpercent:])
```


```python
# form the training data set
idx = np.concatenate(trainind[:])
random.shuffle(idx) # shuffle the indices so that image labels are not grouped in the output set
train = data.iloc[idx]
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 37794 entries, 37078 to 28358
    Columns: 785 entries, label to pixel783
    dtypes: int64(785)
    memory usage: 226.6 MB



```python
idx = np.concatenate(validationind[:])
random.shuffle(idx) # shuffle the indices so that image labels are not grouped in the output set
validation = data.iloc[idx]
```


```python
validation.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4206 entries, 2608 to 1405
    Columns: 785 entries, label to pixel783
    dtypes: int64(785)
    memory usage: 25.2 MB



```python
# write out all data
train.to_csv('datasets/train-exploration.csv', index=False)
validation.to_csv('datasets/validation-exploration.csv', index=False)
```
