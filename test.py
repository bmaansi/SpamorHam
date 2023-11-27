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

# import matplotlib.pyplot as plt

# logs = model_1.make_inspector().training_logs()

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
# plt.xlabel("Number of trees")
# plt.ylabel("Accuracy (out-of-bag)")

# plt.subplot(1, 2, 2)
# plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
# plt.xlabel("Number of trees")
# plt.ylabel("Logloss (out-of-bag)")

# plt.show()

tfdf.keras.get_all_models()

# help works anywhere.
#help(tfdf.keras.RandomForestModel)

# ? only works in ipython or notebooks, it usually opens on a separate panel.
tfdf.keras.RandomForestModel

feature_1 = tfdf.keras.FeatureUsage(name="word_freq_make")
feature_2 = tfdf.keras.FeatureUsage(name="word_freq_address")
feature_3 = tfdf.keras.FeatureUsage(name="word_freq_all")
feature_4 = tfdf.keras.FeatureUsage(name="word_freq_our")
feature_5 = tfdf.keras.FeatureUsage(name="word_freq_over")
feature_6 = tfdf.keras.FeatureUsage(name="word_freq_remove")
feature_7 = tfdf.keras.FeatureUsage(name="word_freq_internet")
feature_8 = tfdf.keras.FeatureUsage(name="word_freq_order")
feature_9 = tfdf.keras.FeatureUsage(name="word_freq_mail")
feature_10 = tfdf.keras.FeatureUsage(name="word_freq_receive")
feature_11 = tfdf.keras.FeatureUsage(name="word_freq_will")
feature_12 = tfdf.keras.FeatureUsage(name="word_freq_people")
feature_13 = tfdf.keras.FeatureUsage(name="word_freq_report")
feature_14 = tfdf.keras.FeatureUsage(name="word_freq_addresses")
feature_15 = tfdf.keras.FeatureUsage(name="word_freq_free")
feature_16 = tfdf.keras.FeatureUsage(name="word_freq_business")
feature_17 = tfdf.keras.FeatureUsage(name="word_freq_email")
feature_18 = tfdf.keras.FeatureUsage(name="word_freq_you")
feature_19 = tfdf.keras.FeatureUsage(name="word_freq_credit")
feature_20 = tfdf.keras.FeatureUsage(name="word_freq_your")
feature_21 = tfdf.keras.FeatureUsage(name="word_freq_font")
feature_22 = tfdf.keras.FeatureUsage(name="word_freq_000")
feature_23 = tfdf.keras.FeatureUsage(name="word_freq_money")
feature_24 = tfdf.keras.FeatureUsage(name="word_freq_hp")
feature_25 = tfdf.keras.FeatureUsage(name="word_freq_hpl")
feature_26 = tfdf.keras.FeatureUsage(name="word_freq_george")
feature_27 = tfdf.keras.FeatureUsage(name="word_freq_650")



all_features = [feature_1, feature_2, feature_3, feature_4]

# Note: This model is only trained with two features. It will not be as good as
# the one trained on all features.

model_2 = tfdf.keras.GradientBoostedTreesModel(
    features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))


# feature_1 = tfdf.keras.FeatureUsage(name="word_freq_all", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
# feature_2 = tfdf.keras.FeatureUsage(name="word_freq_3d")
# feature_3 = tfdf.keras.FeatureUsage(name="word_freq_our")
# all_features = [feature_1, feature_2, feature_3]

# model_3 = tfdf.keras.GradientBoostedTreesModel(features=all_features, exclude_non_specified_features=True)
# model_3.compile( metrics=["accuracy"])

# model_3.fit(train_ds, validation_data=test_ds)