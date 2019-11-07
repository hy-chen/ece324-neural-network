"""
    3.1 Create train/validation/test splits

 overfit.tsv only 50 training samples with 25 o and 25 s.

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torchtext import data
import torchtext
random_seed = 3

data = pd.read_csv("data/data.tsv", delimiter="\t")

npdata = data.to_numpy()

train_valid, test = train_test_split(npdata, test_size=0.2, random_state=random_seed, stratify=npdata[:,1])
train, valid = train_test_split(train_valid, test_size=0.2, random_state=random_seed, stratify=train_valid[:,1])
_, overfit = train_test_split(npdata, test_size=50/10000, random_state=random_seed, stratify=npdata[:,1])


train = pd.DataFrame(train, columns=["text", "label"])
valid = pd.DataFrame(valid, columns=["text", "label"])
test = pd.DataFrame(test, columns=["text", "label"])
overfit = pd.DataFrame(overfit, columns=["text", "label"])

train.to_csv("data/train.tsv", "\t", header=True, index=False)
valid.to_csv("data/validation.tsv", "\t", header=True, index=False)
test.to_csv("data/test.tsv", "\t", header=True, index=False)
overfit.to_csv("data/overfit.tsv", "\t", header=True, index=False)

for i in [train,valid,test,overfit]:
    print(i["label"].value_counts())



