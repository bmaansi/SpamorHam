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
# print(df)
label = "Class"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(y, label=label)

# %set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)