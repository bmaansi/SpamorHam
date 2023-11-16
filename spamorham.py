import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from ucimlrepo import fetch_ucirepo 
# from IPython.display import display

# Check the version of TensorFlow Decision Forests
# print("Found TensorFlow Decision Forests v" + tfdf.__version__)


  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
  
# metadata 
# print(spambase.metadata) 
  
# variable information 
# print(spambase.variables) 

# df = pd.DataFrame(y,
#                     columns = spambase.names)



# test = pd.DataFrame

# print(df.to_markdown())
print(y)
print(X)

# Split the dataset into a training and a testing dataset.

label = "word_freq_make"

classes = X[label].unique().tolist()
print(f"Label classes: {classes}")

X[label] = X[label].map(classes.index)


def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(X)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)
# %set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)

# print("Found TensorFlow Decision Forests v" + tf.__version__)

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")