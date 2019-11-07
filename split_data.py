"""
    3.1 Create train/validation/test splits
 tab separated value, -> array.
 text(string), label(binary 0 or 1, string) 0 = objective.
 two labels - two sets.
 three sets: training 0.64, validation 0.16, test 0.20
 method: /2 -> /3 -> concat obj and sub samples.
 save into three tsv files.
 print out the number of o and s in each train,valid,test set.

 key dattatypes: labelssplit-> ndarray trainvalidtestsplit and savetotsv

 stratification by torchtext.data.Dataset (stratification = keep class ratio same as before splitting.

 a = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1])
 np.bincount(a)
 Out: array([8, 4])

 overfit.tsv only 50 training samples with 25 o and 25 s.

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torchtext import data
import torchtext

data = pd.read_csv("data/data.tsv", delimiter="\t")

print("shape:\n", data.shape)
print("columns name:\n", data.columns)
print("view of first 5 rows:\n", data.head())
print(data["label"].value_counts())

print(type(data))
npdata = data.to_numpy()
print(type(npdata))
random_seed = 3
A, test = train_test_split(npdata, test_size=0.2, random_state=random_seed, stratify=npdata[:,1])
print(A.shape)
print(test.shape)
print(test)
labels = A[:,1]
print(labels)
o = 0
s = 0
for i in labels:
    if i == 1:
        s+=1
    else:
        o+=1

print(o,s)





