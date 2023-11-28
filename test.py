import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from IPython.display import display

# Download the dataset
# wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("spambase.csv")

# Display the first 3 examples.
dataset_df.head(3)

# Encode the categorical labels as integers.
#
# Details:
# This stage is necessary if your classification label is represented as a
# string since Keras expects integer classification labels.
# When using `pd_dataframe_to_tf_dataset` (see below), this step can be skipped.

# Name of the label column.
label = "spam"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

# Split the dataset into a training and a testing dataset.

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# %set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

model_1.save("/tmp/my_saved_model")

tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)

# %set_cell_height 300
model_1.summary()

# The input features
model_1.make_inspector().features()

# The feature importances
model_1.make_inspector().variable_importances()

model_1.make_inspector().evaluation()

#%set_cell_height 150
model_1.make_inspector().training_logs()

import matplotlib.pyplot as plt

logs = model_1.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

#plt.show()
tfdf.keras.get_all_models()

feature_1 = tfdf.keras.FeatureUsage(name="word_freq_make")
feature_2 = tfdf.keras.FeatureUsage(name="word_freq_address")
feature_3 = tfdf.keras.FeatureUsage(name="word_freq_all")
feature_4 = tfdf.keras.FeatureUsage(name="word_freq_3d")
feature_5 = tfdf.keras.FeatureUsage(name="word_freq_our")
feature_6 = tfdf.keras.FeatureUsage(name="word_freq_over")
feature_7 = tfdf.keras.FeatureUsage(name="word_freq_remove")
feature_8 = tfdf.keras.FeatureUsage(name="word_freq_internet")
feature_9 = tfdf.keras.FeatureUsage(name="word_freq_order")
feature_10 = tfdf.keras.FeatureUsage(name="word_freq_mail")
feature_11 = tfdf.keras.FeatureUsage(name="word_freq_receive")
feature_12 = tfdf.keras.FeatureUsage(name="word_freq_will")
feature_13 = tfdf.keras.FeatureUsage(name="word_freq_people")
feature_14 = tfdf.keras.FeatureUsage(name="word_freq_report")
feature_15 = tfdf.keras.FeatureUsage(name="word_freq_addresses")
feature_16 = tfdf.keras.FeatureUsage(name="word_freq_free")
feature_17 = tfdf.keras.FeatureUsage(name="word_freq_business")
feature_18 = tfdf.keras.FeatureUsage(name="word_freq_email")
feature_19 = tfdf.keras.FeatureUsage(name="word_freq_you")
feature_20 = tfdf.keras.FeatureUsage(name="word_freq_credit")
feature_21 = tfdf.keras.FeatureUsage(name="word_freq_your")
feature_22 = tfdf.keras.FeatureUsage(name="word_freq_font")
feature_23 = tfdf.keras.FeatureUsage(name="word_freq_000")
feature_24 = tfdf.keras.FeatureUsage(name="word_freq_money")
feature_25 = tfdf.keras.FeatureUsage(name="word_freq_hp")
feature_26 = tfdf.keras.FeatureUsage(name="word_freq_hpl")
feature_27 = tfdf.keras.FeatureUsage(name="word_freq_george")
feature_28 = tfdf.keras.FeatureUsage(name="word_freq_650")
feature_29 = tfdf.keras.FeatureUsage(name="word_freq_lab")
feature_30 = tfdf.keras.FeatureUsage(name="word_freq_labs")
feature_31 = tfdf.keras.FeatureUsage(name="word_freq_telnet")
feature_32 = tfdf.keras.FeatureUsage(name="word_freq_857")
feature_33 = tfdf.keras.FeatureUsage(name="word_freq_data")
feature_34 = tfdf.keras.FeatureUsage(name="word_freq_415")
feature_35 = tfdf.keras.FeatureUsage(name="word_freq_85")
feature_36 = tfdf.keras.FeatureUsage(name="word_freq_technology")
feature_37 = tfdf.keras.FeatureUsage(name="word_freq_1999")
feature_38 = tfdf.keras.FeatureUsage(name="word_freq_parts")
feature_39 = tfdf.keras.FeatureUsage(name="word_freq_pm")
feature_40 = tfdf.keras.FeatureUsage(name="word_freq_direct")
feature_41 = tfdf.keras.FeatureUsage(name="word_freq_cs")
feature_42 = tfdf.keras.FeatureUsage(name="word_freq_meeting")
feature_43 = tfdf.keras.FeatureUsage(name="word_freq_original")
feature_44 = tfdf.keras.FeatureUsage(name="word_freq_project")
feature_45 = tfdf.keras.FeatureUsage(name="word_freq_re")
feature_46 = tfdf.keras.FeatureUsage(name="word_freq_edu")
feature_47 = tfdf.keras.FeatureUsage(name="word_freq_table")
feature_48 = tfdf.keras.FeatureUsage(name="word_freq_conference")
feature_49 = tfdf.keras.FeatureUsage(name="char_freq_semicolon")
feature_50 = tfdf.keras.FeatureUsage(name="char_freq_parenthesis")
feature_51 = tfdf.keras.FeatureUsage(name="char_freq_bracket")
feature_52 = tfdf.keras.FeatureUsage(name="char_freq_exclamation")
feature_53 = tfdf.keras.FeatureUsage(name="char_freq_dollarsign")
feature_54 = tfdf.keras.FeatureUsage(name="char_freq_hash")
feature_55 = tfdf.keras.FeatureUsage(name="capital_run_length_average")
feature_56 = tfdf.keras.FeatureUsage(name="capital_run_length_longest")
feature_57 = tfdf.keras.FeatureUsage(name="capital_run_length_total")


#update

all_features = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9,feature_10, feature_11,
feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23,
feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31, feature_32, feature_33, feature_34, feature_35, 
feature_36, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47,
feature_48, feature_49, feature_50, feature_51, feature_52, feature_53, feature_54, feature_55, feature_56, feature_57] 

# Note: This model is only trained with two features. It will not be as good as
# the one trained on all features.

model_2 = tfdf.keras.RandomForestModel(features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))